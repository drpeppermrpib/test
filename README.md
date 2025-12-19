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
import re  # for parsing ckpool stats (unused but kept)

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

# ======================  GLOBALS ======================
fShutdown = False
hashrates = [0] * 192  # fixed size for stability
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
pool_diff = 10000  # default, updated from set_difficulty
host = port = user = password = None

# Global log lines for display
log_lines = []
max_log = 15

# ======================  LOGGER (LV06 style with ₿ timestamp) ======================
def logg(msg):
    timestamp = int(time.time() * 100000)  # mimic LV06 timestamp
    prefixed_msg = f"₿ ({timestamp}) {msg}"
    log_lines.append(prefixed_msg)

logg("Miner starting...")

# ======================  CONFIG ======================
PROHASHING_HOST = 'us.mining.prohashing.com'  # change to 'eu' or 'asia' if timeout
PROHASHING_PORT = 3335

num_cores = os.cpu_count()
num_threads = num_cores * 4  # heavier load for Threadripper

# ======================  SIGNAL ======================
def signal_handler(sig, frame):
    global fShutdown
    fShutdown = True
    logg("\n[!] Shutting down...")

signal.signal(signal.SIGINT, signal_handler)

# ======================  MERKLE ROOT ======================
def calculate_merkle_root():
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
            log_lines.append("*** SHARE ACCEPTED ***")
            log_lines.append(f"Nonce: {nonce:08x}")
            log_lines.append(f"Time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
            curses.beep()
        else:
            global rejected
            with lock:
                rejected += 1
            rejected_timestamps.append(time.time())
    except BrokenPipeError:
        logg("[!] Broken pipe – connection lost")
    except Exception as e:
        logg(f"[!] Submit failed: {e}")

# ======================  MINING LOOP (optimized, LV06 logs, nonce shuffling, heavier load) ======================
def bitcoin_miner(thread_id):
    global nbits, version, prevhash, ntime, target, pool_diff

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

            header_static = version + prevhash + calculate_merkle_root() + ntime + nbits
            header_bytes = binascii.unhexlify(header_static)

            # Network (block) target
            network_target = (nbits[2:] + '00' * (int(nbits[:2],16) - 3)).zfill(64)

            # Use pool difficulty for share target (fluctuates)
            share_target = target if target else diff_to_target(128)

            # Shuffle starting nonce
            nonce = random.randint(0, 0xffffffff)

        for _ in range(500000):
            nonce = (nonce - 1) & 0xffffffff
            h = hashlib.sha256(hashlib.sha256(header_bytes + nonce.to_bytes(4,'little')).digest()).digest()
            h_hex = binascii.hexlify(h[::-1]).decode()

            hashes_done += 1

            if h_hex < share_target:
                diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
                share_diff = diff1 / int(h_hex, 16)
                logg(f"asic_result: Nonce difficulty {share_diff:.2f} of {pool_diff}")
                is_block = h_hex < network_target
                submit_share(nonce)
                if is_block:
                    log_lines.append("*** BLOCK SOLVED!!! ***")
                    log_lines.append(f"Nonce: {nonce:08x}")
                    log_lines.append(f"Time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    curses.flash()
                    curses.beep()
                    curses.beep()
                    curses.beep()
                    curses.beep()
                    curses.beep()

            if hashes_done % 100000 == 0:
                now = time.time()
                elapsed = now - last_report
                if elapsed > 0:
                    hr = int(100000 / elapsed)
                    with lock:
                        hashrates[thread_id] = hr
                last_report = now

        # Increment extranonce2 when nonce wraps
        extranonce2_int = int(extranonce2, 16)
        extranonce2_int += 1
        extranonce2 = f"{extranonce2_int:0{extranonce2_size*2}x}"

# ======================  STRATUM (LV06 style logs, reconnection) ======================
def stratum_worker():
    global sock, job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce1, extranonce2_size, pool_diff

    while not fShutdown:
        try:
            s = socket.socket()
            s.settimeout(30)  # longer timeout
            s.connect((host, port))
            sock = s

            s.sendall(b'{"id":1,"method":"mining.subscribe","params":[]}\n')

            # Handle subscribe response
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
                        pool_diff = int(msg["params"][0])
                        target = diff_to_target(pool_diff)
                        logg(f"[*] Difficulty set to {pool_diff}")
        except socket.timeout:
            logg("[!] Connection timed out – reconnecting in 10s...")
            time.sleep(10)
        except Exception as e:
            logg(f"[!] Stratum error: {e} – reconnecting in 10s...")
            time.sleep(10)

# ======================  DISPLAY (stable top bar, scrolling logs) ======================
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
            a_min = sum(1 for t in accepted_timestamps if now-t<60)
            r_min = sum(1 for t in rejected_timestamps if now-t<60)

            cpu_temp = get_cpu_temp()

            # Top right: Ctrl+C to quit
            stdscr.addstr(0, max(0, screen_width - 20), "Ctrl+C to quit", curses.color_pair(3))

            # Title
            title = "Bitcoin Miner (CPU) - ProHashing"
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

            # Yellow line
            stdscr.addstr(11, 0, "─" * (screen_width - 1), curses.color_pair(3))

            # Scrolling log area (stable)
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
    parser = argparse.ArgumentParser(description="ProHashing SHA-256 CPU Miner")
    parser.add_argument("--username", type=str, required=True, help="ProHashing username")
    parser.add_argument("--worker", type=str, default="001", help="Worker name (e.g., 001)")
    parser.add_argument("--group", type=str, default="kx2000", help="Group name (e.g., kx2000)")
    parser.add_argument("--wattage", type=int, default=1000, help="Wattage for tracking (e.g., 1000)")
    parser.add_argument("--price", type=float, default=1.45, help="Electricity price $/kWh (e.g., 1.45)")
    parser.add_argument("--server", type=str, default="us", help="Server region (us, eu, asia)")
    args = parser.parse_args()

    # Construct ProHashing password string
    password = f"a=sha-256,c=bitcoin,w={args.wattage},p={args.price},n={args.worker},o={args.group}"

    server_map = {
        "us": "us.mining.prohashing.com",
        "eu": "eu.mining.prohashing.com",
        "asia": "asia.mining.prohashing.com"
    }
    host = server_map.get(args.server.lower(), PROHASHING_HOST)
    port = PROHASHING_PORT
    user = args.username

    # Fixed size hashrates list for stability
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
