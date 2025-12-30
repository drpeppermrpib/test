#!/usr/bin/env python3

# alfa5.py - Braiins Pool CPU Miner (Inspired by Pymmdrza/SoloMinerV2 style)
# Clean, colorful curses UI with live stats, connection status, and gradual thread ramp-up
# Submits correctly to Braiins Pool (Stratum V1)
# Automatic extranonce2_size, live pool_diff, stable connection
# Gradual startup: 1 core → full (cores * 8) over 30 seconds

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
import subprocess

# ======================  diff_to_target ======================
def diff_to_target(diff):
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    target_int = diff1 // int(diff)
    return format(target_int, '064x')

# ======================  CPU TEMPERATURE (fixed for HiveOS/AMD) ======================
def get_cpu_temp():
    try:
        # Primary: sensors command (HiveOS standard)
        result = subprocess.check_output(["sensors"], text=True)
        for line in result.splitlines():
            if 'Tdie' in line or 'Tctl' in line:
                temp = line.split(':')[1].strip().split(' ')[0]
                return f"{temp} (Tdie/Tctl)"
    except:
        pass

    # Fallback: thermal zones
    try:
        for zone in range(20):
            path = f"/sys/class/thermal/thermal_zone{zone}/type"
            if os.path.exists(path):
                with open(path) as f:
                    zone_type = f.read().strip()
                if 'cpu' in zone_type.lower() or 'core' in zone_type.lower():
                    with open(f"/sys/class/thermal/thermal_zone{zone}/temp") as f:
                        temp = int(f.read().strip()) / 1000
                        return f"{temp:.1f}°C ({zone_type})"
    except:
        pass
    return "N/A"

# ======================  GLOBALS ======================
fShutdown = False
hashrates = [0] * 512
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
extranonce2_size = 4
sock = None
target = None
pool_diff = 0  # will be set by pool

# log lines
log_lines = []
max_log = 40

# connection status
connected = False

# ======================  LOGGER ======================
def logg(msg):
    timestamp = time.strftime("%H:%M:%S")
    prefixed_msg = f"[{timestamp}] {msg}"
    log_lines.append(prefixed_msg)

logg("alfa5.py starting...")

# ======================  CONFIG ======================
BRAIINS_HOST = 'stratum.braiins.com'
BRAIINS_PORT = 3333

num_cores = os.cpu_count()
max_threads = num_cores * 8  # final target for Threadripper
current_threads = 0

# ======================  SIGNAL ======================
def signal_handler(sig, frame):
    global fShutdown
    fShutdown = True
    logg("Shutting down...")

signal.signal(signal.SIGINT, signal_handler)

# ======================  SUBMIT SHARE ======================
def submit_share(nonce):
    payload = {
        "id": None,
        "method": "mining.submit",
        "params": [user, job_id, extranonce2, ntime, f"{nonce:08x}"]
    }
    try:
        sock.sendall((json.dumps(payload) + "\n").encode())
        resp = sock.recv(1024).decode().strip()
        logg(f"Submitted share → {resp}")
        if "true" in resp.lower():
            global accepted
            with lock:
                accepted += 1
            accepted_timestamps.append(time.time())
            logg("*** SHARE ACCEPTED ***")
        else:
            global rejected
            with lock:
                rejected += 1
            logg("[!] Share rejected")
    except Exception as e:
        logg(f"[!] Submit error: {e}")

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

            logg(f"New job: {job_id}")

            extranonce2 = "0" * (extranonce2_size * 2)
            nonce = random.randint(0, 0xffffffff)

            header_static = version + prevhash + coinb1 + extranonce1 + extranonce2 + coinb2 + ntime + nbits
            header_bytes = binascii.unhexlify(header_static)

            network_target = (nbits[2:] + '00' * (int(nbits[:2],16) - 3)).zfill(64)

            share_target = target if target else diff_to_target(max(pool_diff, 1))

        for _ in range(4000000):
            nonce = (nonce - 1) & 0xffffffff
            h = hashlib.sha256(hashlib.sha256(header_bytes + nonce.to_bytes(4,'little')).digest()).digest()
            h_hex = binascii.hexlify(h[::-1]).decode()

            hashes_done += 1

            if h_hex < share_target:
                diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
                share_diff = diff1 / int(h_hex, 16)
                logg(f"Found share: diff {share_diff:.2f}")
                submit_share(nonce)

            if hashes_done % 100000 == 0:
                now = time.time()
                elapsed = now - last_report
                if elapsed > 0:
                    hr = int(100000 / elapsed)
                    hashrates[thread_id] = hr
                last_report = now

        extranonce2_int = int(extranonce2, 16)
        extranonce2_int += 1
        extranonce2 = f"{extranonce2_int:0{extranonce2_size * 2}x}"

# ======================  STRATUM ======================
def stratum_worker():
    global sock, job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce1, extranonce2_size, pool_diff, connected

    while not fShutdown:
        try:
            s = socket.socket()
            s.settimeout(30)
            s.connect((BRAIINS_HOST, BRAIINS_PORT))
            sock = s
            connected = True
            logg(f"Connected to {BRAIINS_HOST}:{BRAIINS_PORT}")

            s.sendall(b'{"id":1,"method":"mining.subscribe","params":["alfa5.py/1.0"]}\n')

            auth = {"id":2,"method":"mining.authorize","params":[user, password]}
            s.sendall((json.dumps(auth) + "\n").encode())

            buf = b""
            while not fShutdown:
                data = s.recv(4096)
                if not data:
                    connected = False
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
                        logg(f"New Work: {job_id}")
                    elif msg.get("method") == "mining.set_difficulty":
                        pool_diff = int(msg["params"][0])
                        target = diff_to_target(pool_diff)
                        logg(f"Pool difficulty: {pool_diff}")
        except Exception as e:
            connected = False
            logg(f"[!] Connection error: {e} – retrying...")
            time.sleep(10)

# ======================  GRADUAL THREAD RAMP-UP ======================
def ramp_up_threads():
    global current_threads
    step = max(1, num_cores)  # at least 1 core per step
    for target in range(step, max_threads + 1, step):
        if fShutdown:
            break
        new_threads = min(target, max_threads)
        while current_threads < new_threads:
            current_threads += 1
            threading.Thread(target=bitcoin_miner, args=(current_threads - 1,), daemon=True).start()
            logg(f"Thread {current_threads}/{max_threads} started")
            time.sleep(2)  # gradual ramp
    logg(f"All {max_threads} threads active – full power!")

# ======================  DISPLAY (Pymmdrza style - clean & colorful) ======================
def display_worker():
    stdscr = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)

    try:
        while not fShutdown:
            stdscr.clear()
            h, w = stdscr.getmaxyx()

            title = " alfa5.py - Braiins Pool CPU Miner "
            stdscr.addstr(0, 0, title.center(w), curses.color_pair(5) | curses.A_BOLD)

            status = "ONLINE" if connected else "OFFLINE"
            color = 1 if connected else 2
            stdscr.addstr(2, 2, f"Status    : {status}", curses.color_pair(color) | curses.A_BOLD)

            try:
                block_height = requests.get('https://mempool.space/api/blocks/tip/height', timeout=3).text
            except:
                block_height = "???"
            stdscr.addstr(3, 2, f"Block     : {block_height}", curses.color_pair(3))

            total_hr = sum(hashrates)
            stdscr.addstr(4, 2, f"Hashrate  : {total_hr:,} H/s", curses.color_pair(1) | curses.A_BOLD)

            stdscr.addstr(5, 2, f"Threads   : {current_threads}/{max_threads}", curses.color_pair(4))

            cpu_temp = get_cpu_temp()
            stdscr.addstr(6, 2, f"Temp      : {cpu_temp}", curses.color_pair(3))

            a_min = sum(1 for t in accepted_timestamps if time.time() - t < 60)
            r_min = sum(1 for t in rejected_timestamps if time.time() - t < 60)
            stdscr.addstr(7, 2, f"Shares/min: {a_min} accepted / {r_min} rejected", curses.color_pair(1 if a_min > 0 else 3))

            stdscr.addstr(8, 2, f"Total     : {accepted} accepted / {rejected} rejected", curses.color_pair(3))

            stdscr.addstr(9, 2, f"Pool Diff : {pool_diff if pool_diff > 0 else 'Waiting...'}", curses.color_pair(4))

            # Horizontal line
            stdscr.addstr(11, 0, "─" * w, curses.color_pair(3))

            # Logs
            start_y = 12
            for i, line in enumerate(log_lines[-max_log:]):
                if start_y + i >= h:
                    break
                color = 1 if "accepted" in line.lower() else (2 if "rejected" in line.lower() or "error" in line.lower() else 3)
                stdscr.addstr(start_y + i, 2, line[:w-4], curses.color_pair(color))

            stdscr.refresh()
            time.sleep(1)
    finally:
        curses.endwin()

# ======================  MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="alfa5.py - Braiins Pool CPU Miner")
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--worker", type=str, default="cpu002")
    args = parser.parse_args()

    user = f"{args.username}.{args.worker}"
    password = "x"

    # Startup boot messages
    boot_msgs = [
        "alfa5.py - Advanced Braiins Pool CPU Miner",
        "Initializing system...",
        f"CPU cores detected: {num_cores}",
        f"Target threads: {max_threads} (cores ×8)",
        "Connecting to Braiins Pool...",
        "Starting stratum worker...",
        "Ramping up threads gradually..."
    ]
    for msg in boot_msgs:
        print(msg)
        time.sleep(0.7)

    # Start stratum
    threading.Thread(target=stratum_worker, daemon=True).start()
    time.sleep(4)

    # Start display
    threading.Thread(target=display_worker, daemon=True).start()

    # Gradual thread ramp-up
    threading.Thread(target=ramp_up_threads, daemon=True).start()

    logg("[*] alfa5.py fully active – mining started!")

    try:
        while not fShutdown:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    fShutdown = True
    time.sleep(2)
    logg("[*] Shutdown complete")
