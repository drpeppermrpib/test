#!/usr/bin/env python3

import threading
import requests
import binascii
import hashlib
import logging
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
import re  # for parsing ckpool stats

# ======================  diff_to_target ======================
def diff_to_target(diff):
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    target_int = diff1 // int(diff)
    return format(target_int, '064x')

# ======================  CPU TEMPERATURE (accurate for HiveOS/AMD) ======================
def get_cpu_temp():
    # HiveOS/AMD: use 'sensors' for Tctl/Tdie
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

    # Fallback thermal zones
    temps = []
    for zone in range(20):
        path = f"/sys/class/thermal/thermal_zone{zone}/temp"
        if os.path.exists(path):
            try:
                with open(path) as f:
                    temp = int(f.read().strip()) / 1000
                    temps.append(temp)
            except:
                pass
    if temps:
        avg = sum(temps) / len(temps)
        max_temp = max(temps)
        return f"{avg:.1f}°C (avg) / {max_temp:.1f}°C (max)"
    return "N/A"

# ======================  CKPOOL STATS (updated regex for current page) ======================
def get_ckpool_stats():
    url = "https://solostats.ckpool.org/users/bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e"
    try:
        r = requests.get(url, timeout=10)
        text = r.text
        hashrate = re.search(r'Hashrate</td><td>([^<]+)', text)
        last_share = re.search(r'Last Share</td><td>([^<]+)', text)
        best_share = re.search(r'Best Share</td><td>([^<]+)', text)
        shares = re.search(r'Shares</td><td>([^<]+)', text)

        return {
            "hashrate": hashrate.group(1).strip() if hashrate else "N/A",
            "last_share": last_share.group(1).strip() if last_share else "N/A",
            "best_share": best_share.group(1).strip() if best_share else "N/A",
            "shares": shares.group(1).strip() if shares else "N/A"
        }
    except:
        return {"hashrate": "N/A", "last_share": "N/A", "best_share": "N/A", "shares": "N/A"}

# ======================  GLOBALS ======================
fShutdown = False
hashrates = []
accepted = rejected = 0
accepted_timestamps = []
rejected_timestamps = []
lock = threading.Lock()

# job data
job_id = prevhash = coinb1 = coinb2 = None
merkle_branch = version = nbits = ntime = None
extranonce1 = extranonce2 = extranonce2_size = None
sock = None
target = None
mode = "solo"
host = port = user = password = None

# Global log lines for display
log_lines = []
max_log = 15

# ======================  LOGGER (LV06 style with ₿ timestamp) ======================
def logg(msg):
    timestamp = int(time.time() * 100000)  # mimic LV06 timestamp
    prefixed_msg = f"₿ ({timestamp}) {msg}"
    sys.stdout.write(prefixed_msg + "\n")
    sys.stdout.flush()
    log_lines.append(prefixed_msg)
    try:
        logging.basicConfig(level=logging.INFO, filename="miner.log",
                            format='%(asctime)s %(message)s', force=True)
        logging.info(prefixed_msg)
    except:
        pass

logg("Miner starting...")

# ======================  CONFIG ======================
SOLO_HOST = 'solo.ckpool.org'
SOLO_PORT = 3333
SOLO_ADDRESS = 'bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e'

POOL_HOST = 'ss.antpool.com'
POOL_PORT = 3333
POOL_WORKER = 'Xk2000.001'
POOL_PASSWORD = 'x'

num_cores = os.cpu_count()
num_threads = num_cores * 2  # Hyper-threading helps for hashlib

# ======================  SIGNAL ======================
def signal_handler(sig, frame):
    global fShutdown
    fShutdown = True
    logg("\n[!] Shutting down...")

signal.signal(signal.SIGINT, signal_handler)

# ======================  MERKLE ROOT ======================
def calculate_merkle_root():
    global extranonce2
    if None in (coinb1, extranonce1, extranonce2, coinb2, merkle_branch):
        return "0" * 64
    coinbase = coinb1 + extranonce1 + extranonce2 + coinb2
    h = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()
    for b in merkle_branch:
        h = hashlib.sha256(hashlib.sha256(h + binascii.unhexlify(b)).digest()).digest()
    return binascii.hexlify(h).decode()[::-1]

# ======================  SUBMIT SHARE (LV06 style logs) ======================
def submit_share(nonce):
    payload = {
        "id": 1,
        "method": "mining.submit",
        "params": [user, job_id, extranonce2, ntime, f"{nonce:08x}"]
    }
    try:
        logg(f"stratum_api: tx: {json.dumps(payload)}")
        sock.sendall((json.dumps(payload) + "\n").encode())
        resp = sock.recv(1024).decode().strip()
        logg(f"stratum_task: rx: {resp}")
        logg("stratum_task: message result accepted" if "true" in resp.lower() else "[!] Share rejected")

        if "true" in resp.lower():
            global accepted
            with lock:
                accepted += 1
            accepted_timestamps.append(time.time())
            sys.stdout.write("\n" + "="*60 + "\n")
            sys.stdout.write("*** SHARE ACCEPTED ***\n")
            sys.stdout.write(f"Nonce: {nonce:08x}\n")
            sys.stdout.write(f"Time : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            sys.stdout.write("="*60 + "\n")
            sys.stdout.flush()
            sys.stdout.write("\a")
            sys.stdout.flush()
        else:
            global rejected
            with lock:
                rejected += 1
            rejected_timestamps.append(time.time())
    except BrokenPipeError:
        logg("[!] Broken pipe – connection lost")
    except Exception as e:
        logg(f"[!] Submit failed: {e}")

# ======================  MINING LOOP (optimized, LV06 logs, nonce shuffling) ======================
def bitcoin_miner(thread_id):
    global nbits, version, prevhash, ntime, target, extranonce2

    last_job_id = None

    hashes_done = 0
    last_report = time.time()

    while not fShutdown:
        if job_id != last_job_id:
            last_job_id = job_id
            if None in (nbits, version, prevhash, ntime):
                time.sleep(0.5)
                continue

            logg(f"create_jobs_task: New Work Dequeued {job_id}")

            # Reset extranonce2 for new job (ckpool allows client to choose)
            extranonce2 = "00" * extranonce2_size

            # Shuffle starting nonce
            nonce = random.randint(0, 0xffffffff)

        header_static = version + prevhash + coinb1 + extranonce1 + extranonce2 + coinb2 + ntime + nbits
        header_bytes = binascii.unhexlify(header_static)

        # Network (block) target
        network_target = (nbits[2:] + '00' * (int(nbits[:2],16) - 3)).zfill(64)

        # Low share target for solo to show live hashrate
        share_target = diff_to_target(128)

        for _ in range(500000):
            nonce = (nonce - 1) & 0xffffffff
            h = hashlib.sha256(hashlib.sha256(header_bytes + nonce.to_bytes(4,'little')).digest()).digest()
            h_hex = binascii.hexlify(h[::-1]).decode()

            hashes_done += 1

            if h_hex < share_target:
                diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
                share_diff = diff1 / int(h_hex, 16)
                logg(f"asic_result: Nonce difficulty {share_diff:.2f} of 371.")
                is_block = h_hex < network_target
                submit_share(nonce)
                if is_block:
                    sys.stdout.write("\n" + "="*80 + "\n")
                    sys.stdout.write("█" + " "*78 + "█\n")
                    sys.stdout.write("█" + " "*28 + "BLOCK SOLVED!!!" + " "*33 + "█\n")
                    sys.stdout.write("█" + " "*78 + "█\n")
                    sys.stdout.write(f"█  Nonce      : {nonce:08x}\n")
                    sys.stdout.write(f"█  Time      : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    sys.stdout.write("█" + " "*78 + "█\n")
                    sys.stdout.write("="*80 + "\n")
                    sys.stdout.flush()
                    sys.stdout.write("\a" * 5)
                    sys.stdout.flush()

            if hashes_done % 100000 == 0:
                now = time.time()
                elapsed = now - last_report
                if elapsed > 0:
                    hr = int(100000 / elapsed)
                    with lock:
                        hashrates[thread_id] = hr
                last_report = now

        # When nonce wraps, increment extranonce2 to avoid duplicate work
        extranonce2_int = int(extranonce2, 16)
        extranonce2_int += 1
        extranonce2 = f"{extranonce2_int:0{extranonce2_size*2}x}"

# ======================  STRATUM (LV06 style logs, reconnection) ======================
def stratum_worker():
    global sock, job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce1, extranonce2_size

    while not fShutdown:
        try:
            s = socket.socket()
            s.connect((host, port))
            sock = s

            # Subscribe
            s.sendall(b'{"id":1,"method":"mining.subscribe","params":[]}\n')
            buf = b""
            while b'\n' not in buf:
                data = s.recv(4096)
                buf += data
            line, buf = buf.split(b'\n', 1)
            msg = json.loads(line)
            logg(f"stratum_task: rx: {json.dumps(msg)}")
            extranonce1 = msg["result"][1]
            extranonce2_size = msg["result"][2]
            logg(f"Subscribed – extranonce1: {extranonce1}, size: {extranonce2_size}")

            # Authorize
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
                    if msg.get("method") == "mining.notify":
                        (job_id, prevhash, coinb1, coinb2,
                         merkle_branch, version, nbits, ntime, _) = msg["params"]
                        logg(f"create_jobs_task: New Work Dequeued {job_id}")
                    elif msg.get("method") == "mining.set_difficulty":
                        target = diff_to_target(msg["params"][0])
                        logg(f"[*] Difficulty set to {msg['params'][0]}")
        except Exception as e:
            logg(f"[!] Stratum error: {e} – reconnecting in 10s...")
            time.sleep(10)

# ======================  DISPLAY ======================
def display_worker():
    global log_lines
    stdscr = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN,  curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED,    curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN,   curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.noecho(); curses.cbreak(); stdscr.keypad(True)

    try:
        while not fShutdown:
            stdscr.clear()
            screen_height, screen_width = stdscr.getmaxyx()

            now = time.time()
            with lock:
                a_min = sum(1 for t in accepted_timestamps if now-t<60)
                r_min = sum(1 for t in rejected_timestamps if now-t<60)

            cpu_temp = get_cpu_temp()

            # Top right: Ctrl+C to quit
            stdscr.addstr(0, max(0, screen_width - 20), "Ctrl+C to quit", curses.color_pair(3))

            # Title
            title = f"Bitcoin {mode.upper()} Miner (CPU)"
            stdscr.addstr(2, 0, title, curses.color_pair(4)|curses.A_BOLD)

            # Static stats
            try:
                block_height = requests.get('https://blockchain.info/q/getblockcount',timeout=3).text
            except:
                block_height = "???"
            stdscr.addstr(4, 0, f"Block height : ~{block_height}", curses.color_pair(3))
            stdscr.addstr(5, 0, f"Hashrate     : {sum(hashrates):,} H/s", curses.color_pair(1))
            stdscr.addstr(6, 0, f"CPU Temp     : {cpu_temp}", curses.color_pair(3))
            stdscr.addstr(7, 0, f"Threads      : {num_threads}", curses.color_pair(3))
            stdscr.addstr(8, 0, f"Shares       : {accepted} accepted / {rejected} rejected")
            stdscr.addstr(9, 0, f"Last minute  : {a_min} acc / {r_min} rej")

            # ckpool stats under last minute
            stats = get_ckpool_stats()
            stdscr.addstr(11, 0, f"ckpool Hashrate : {stats['hashrate']}", curses.color_pair(1))
            stdscr.addstr(12, 0, f"Last Share      : {stats['last_share']}", curses.color_pair(3))
            stdscr.addstr(13, 0, f"Best Share      : {stats['best_share']}", curses.color_pair(1))
            stdscr.addstr(14, 0, f"Total Shares    : {stats['shares']}", curses.color_pair(3))

            # Yellow line
            stdscr.addstr(16, 0, "─" * (screen_width - 1), curses.color_pair(3))

            # Scrolling log area (stable)
            start_y = 17
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["solo","pool"], default="solo")
    parser.add_argument("--worker", type=str, default="002", help="Worker name (e.g., 001 for LV06, 002 for Python miner)")
    args = parser.parse_args()

    mode = args.mode
    worker = args.worker
    if mode == "solo":
        host, port, user, password = SOLO_HOST, SOLO_PORT, f"{SOLO_ADDRESS}.{worker}", "x"
    else:
        host, port, user, password = POOL_HOST, POOL_PORT, POOL_WORKER, POOL_PASSWORD

    # Initialize shared hashrates list
    hashrates = [0] * num_threads

    # Start stratum
    threading.Thread(target=stratum_worker, daemon=True).start()
    time.sleep(3)

    # Start mining threads
    for i in range(num_threads):
        threading.Thread(target=bitcoin_miner, args=(i,), daemon=True).start()

    # Display
    threading.Thread(target=display_worker, daemon=True).start()

    logg("[*] Miner running – press Ctrl+C to stop")
    try:
        while not fShutdown:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    fShutdown = True
    time.sleep(2)
    logg("[*] Shutdown complete")
