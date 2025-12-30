#!/usr/bin/env python3

import threading
import requests
import binascii
import hashlib
import random
import socket
import time
import json
import sys
import os
import curses
import argparse
import signal
import subprocess  # for accurate temp

# ======================  diff_to_target ======================
def diff_to_target(diff):
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    target_int = diff1 // int(diff)
    return format(target_int, '064x')

# ======================  CPU TEMPERATURE ======================
def get_cpu_temp():
    try:
        result = subprocess.check_output(["sensors"], text=True)
        temps = []
        for line in result.splitlines():
            if 'Tctl' in line or 'Tdie' in line or 'edge' in line or 'junction' in line:
                match = re.search(r'[\+\-]?[\d\.]+°C', line)
                if match:
                    temps.append(float(match.group(0).replace('°C', '').strip()))
        if temps:
            avg = sum(temps) / len(temps)
            max_temp = max(temps)
            return f"{avg:.1f}°C (avg) / {max_temp:.1f}°C (max)"
    except:
        pass
    return "N/A"

# ======================  FETCH LIVE BRAIINS DATA ======================
def fetch_live_data():
    try:
        hr_data = requests.get("https://api.blockchain.info/charts/hash-rate?timespan=1days&format=json", timeout=5).json()
        network_hr = f"{hr_data['values'][-1]['y'] / 1e6:.2f} EH/s" if hr_data else "N/A"

        diff_data = requests.get("https://api.blockchain.info/charts/difficulty?timespan=1days&format=json", timeout=5).json()
        difficulty = f"{diff_data['values'][-1]['y'] / 1e12:.2f} T" if diff_data else "N/A"

        fees_data = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=5).json()
        block_fees = f"{fees_data['hourFee']} SAT/vB" if fees_data else "N/A"

        return [
            f"Network HR   : {network_hr}",
            "Avg 30d HR   : N/A",
            f"Difficulty   : {difficulty}",
            f"Block Fees   : {block_fees}",
            "Hash Value   : ~0.0004 BTC/PH/Day",
            "Hash Price   : ~$37/PH/Day",
            "Profit Ex.   : N/A"
        ]
    except Exception:
        return ["Live Data: Offline"] * 7

LIVE_DATA = fetch_live_data()

# ======================  GLOBALS ======================
fShutdown = False
hashrates = [0] * 256
accepted = rejected = 0
accepted_timestamps = []
rejected_timestamps = []
lock = threading.Lock()

# job data
job_id = prevhash = coinb1 = coinb2 = None
merkle_branch = []
version = nbits = ntime = None
extranonce1 = "00000000"
extranonce2 = "00000000"
extranonce2_size = 4  # automatic from pool
sock = None
target = None
pool_diff = 429496729600000  # 15-digit fallback

# Global log lines for display
log_lines = []
max_log = 40

# Last error time
last_error_time = 0

# ======================  LOGGER ======================
def logg(msg):
    timestamp = int(time.time() * 100000)
    prefixed_msg = f"₿ ({timestamp}) {msg}"
    log_lines.append(prefixed_msg)

logg("minerAlfa2 starting...")

# ======================  CONFIG ======================
BRAIINS_HOST = 'stratum.braiins.com'
BRAIINS_PORT = 3333

num_cores = os.cpu_count()
num_threads = num_cores * 4  # Threadripper max

# ======================  SIGNAL ======================
def signal_handler(sig, frame):
    global fShutdown
    fShutdown = True
    logg("\n[!] Shutting down...")

signal.signal(signal.SIGINT, signal_handler)

# ======================  SUBMIT SHARE ======================
def submit_share(nonce):
    current_time = time.time()
    if current_time - last_error_time < 0.5:
        return

    payload = {
        "id": None,
        "method": "mining.submit",
        "params": [user, job_id, extranonce2, ntime, f"{nonce:08x}"]
    }
    try:
        logg(f"stratum_api: tx: {json.dumps(payload)}")
        sock.sendall((json.dumps(payload) + "\n").encode())
        resp = sock.recv(1024).decode().strip()
        logg(f"stratum_task: rx: {resp}")
        if "true" in resp.lower():
            logg("stratum_task: message result accepted")
            global accepted
            with lock:
                accepted += 1
            accepted_timestamps.append(time.time())
            log_lines.append("*** SHARE ACCEPTED ***")
            log_lines.append(f"Nonce: {nonce:08x}")
            log_lines.append(f"Time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
            curses.beep()
        else:
            logg("[!] Share rejected")
            global rejected
            with lock:
                rejected += 1
            rejected_timestamps.append(time.time())
    except Exception as e:
        current_time = time.time()
        if current_time - last_error_time > 10:
            logg(f"[!] Submit error: {e}")
            last_error_time = current_time

# ======================  MINING LOOP ======================
def bitcoin_miner(thread_id):
    global job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce2, pool_diff

    last_job_id = None
    hashes_done = 0
    last_report = time.time()

    header_bytes = b''
    nonce = 0

    while not fShutdown:
        if job_id != last_job_id:
            last_job_id = job_id
            if None in (nbits, version, prevhash, ntime):
                time.sleep(0.5)
                continue

            logg(f"create_jobs_task: New Work Dequeued {job_id}")

            extranonce2 = "0" * (extranonce2_size * 2)
            nonce = random.randint(0, 0xffffffff)

            header_static = version + prevhash + coinb1 + extranonce1 + extranonce2 + coinb2 + ntime + nbits
            header_bytes = binascii.unhexlify(header_static)

            network_target = (nbits[2:] + '00' * (int(nbits[:2],16) - 3)).zfill(64)

            share_target = target if target else diff_to_target(pool_diff)

        for _ in range(2000000):
            nonce = (nonce - 1) & 0xffffffff
            h = hashlib.sha256(hashlib.sha256(header_bytes + nonce.to_bytes(4,'little')).digest()).digest()
            h_hex = binascii.hexlify(h[::-1]).decode()

            hashes_done += 1

            if h_hex < share_target:
                diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
                share_diff = diff1 / int(h_hex, 16)
                logg(f"asic_result: Nonce difficulty {share_diff:.2f} of {pool_diff}.")
                submit_share(nonce)

            if hashes_done % 50000 == 0:
                now = time.time()
                elapsed = now - last_report
                if elapsed > 0:
                    hr = int((50000 * 4) / elapsed)
                    hashrates[thread_id] = hr
                last_report = now

        extranonce2_int = int(extranonce2, 16)
        extranonce2_int += 1
        extranonce2 = f"{extranonce2_int:0{extranonce2_size * 2}x}"

# ======================  STRATUM ======================
def stratum_worker():
    global sock, job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce1, extranonce2_size, pool_diff

    while not fShutdown:
        try:
            s = socket.socket()
            s.settimeout(30)
            s.connect((BRAIINS_HOST, BRAIINS_PORT))
            sock = s

            s.sendall(b'{"id":1,"method":"mining.subscribe","params":["minerAlfa2.py/1.0"]}\n')

            auth = {"id":2,"method":"mining.authorize","params":[user,password]}
            s.sendall((json.dumps(auth)+"\n").encode())

            buf = b""
            while not fShutdown:
                data = s.recv(4096)
                if not data:
                    logg("[!] Connection lost – reconnecting...")
                    break
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    if not line.strip(): continue
                    msg = json.loads(line)
                    logg(f"stratum_task: rx: {json.dumps(msg)}")
                    if "result" in msg and msg["id"] == 1:
                        extranonce1 = msg["result"][1]
                        extranonce2_size = msg["result"][2]
                        logg(f"Subscribed – extranonce1: {extranonce1}, size: {extranonce2_size}")
                    elif msg.get("method") == "mining.notify":
                        params = msg["params"]
                        job_id = params[0]
                        prevhash = params[1]
                        coinb1 = params[2]
                        coinb2 = params[3]
                        merkle_branch = params[4]
                        version = params[5]
                        nbits = params[6]
                        ntime = params[7]
                        logg(f"create_jobs_task: New Work Dequeued {job_id}")
                    elif msg.get("method") == "mining.set_difficulty":
                        pool_diff = msg["params"][0]
                        target = diff_to_target(pool_diff)
                        logg(f"[*] Difficulty set to {pool_diff}")
        except Exception as e:
            logg(f"[!] Stratum error: {e} – reconnecting in 10s...")
            time.sleep(10)

# ======================  DISPLAY ======================
def display_worker():
    global log_lines
    stdscr = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN,  curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN,   curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.noecho(); curses.cbreak(); stdscr.keypad(True)

    try:
        while not fShutdown:
            stdscr.clear()
            screen_height, screen_width = stdscr.getmaxyx()

            now = time.time()
            a_min = sum(1 for t in accepted_timestamps if now-t<60)
            r_min = sum(1 for t in rejected_timestamps if now-t<60)

            cpu_temp = get_cpu_temp()

            stdscr.addstr(0, max(0, screen_width - 20), "Ctrl+C to quit", curses.color_pair(3))
            title = "Bitcoin Miner (CPU) - Braiins Pool"
            stdscr.addstr(2, 0, title, curses.color_pair(4)|curses.A_BOLD)

            try:
                block_height = requests.get('https://blockchain.info/q/getblockcount', timeout=3).text
            except:
                block_height = "???"
            stdscr.addstr(4, 0, f"Block height : ~{block_height}", curses.color_pair(3))
            stdscr.addstr(5, 0, f"Hashrate     : {sum(hashrates):,} H/s", curses.color_pair(1))
            stdscr.addstr(6, 0, f"CPU Temp     : {cpu_temp}", curses.color_pair(3))
            stdscr.addstr(7, 0, f"Threads      : {num_threads}", curses.color_pair(3))
            stdscr.addstr(8, 0, f"Shares       : {accepted} accepted / {rejected} rejected")
            stdscr.addstr(9, 0, f"Last minute  : {a_min} acc / {r_min} rej")

            right_x = max(50, screen_width // 2 + 5)
            stdscr.addstr(4, right_x, "Braiins Live Data", curses.color_pair(4)|curses.A_BOLD)
            for i, line in enumerate(LIVE_DATA):
                if 5 + i < 11:
                    stdscr.addstr(5 + i, right_x, line, curses.color_pair(3))

            stdscr.addstr(11, 0, "─" * (screen_width - 1), curses.color_pair(3))

            start_y = 12
            for i, line in enumerate(log_lines[-max_log:]):
                if start_y + i >= screen_height:
                    break
                stdscr.addstr(start_y + i, 0, line[:screen_width-1], curses.color_pair(5))

            stdscr.refresh()
            time.sleep(1)
    finally:
        curses.endwin()

# ======================  MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Braiins Pool SHA-256 CPU Miner - minerAlfa2")
    parser.add_argument("--username", type=str, required=True, help="Braiins username")
    parser.add_argument("--worker", type=str, default="cpu002", help="Worker name")
    args = parser.parse_args()

    user = f"{args.username}.{args.worker}"
    password = "x"

    hashrates = [0] * num_threads

    threading.Thread(target=stratum_worker, daemon=True).start()
    time.sleep(5)

    for i in range(num_threads):
        threading.Thread(target=bitcoin_miner, args=(i,), daemon=True).start()

    threading.Thread(target=display_worker, daemon=True).start()

    logg("[*] minerAlfa2 running – press Ctrl+C to stop")
    try:
        while not fShutdown:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    fShutdown = True
    time.sleep(2)
    logg("[*] Shutdown complete")
