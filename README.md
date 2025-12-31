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
import subprocess
import struct

# ======================  diff_to_target ======================
def diff_to_target(diff):
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    target_int = diff1 // int(diff)
    return format(target_int, '064x')

# ======================  CPU TEMPERATURE ======================
def get_cpu_temp():
    try:
        result = subprocess.check_output(["sensors"], text=True)
        for line in result.splitlines():
            if 'Tctl' in line or 'Tdie' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    temp = parts[1].strip().split(' ')[0]
                    return f"{temp}"
    except:
        pass

    try:
        for hwmon in os.listdir('/sys/class/hwmon'):
            name_path = f"/sys/class/hwmon/{hwmon}/name"
            if os.path.exists(name_path):
                with open(name_path) as f:
                    name = f.read().strip()
                if name in ['k10temp', 'zenpower']:
                    temp_path = f"/sys/class/hwmon/{hwmon}/temp1_input"
                    if os.path.exists(temp_path):
                        with open(temp_path) as f:
                            temp = int(f.read().strip()) / 1000
                        return f"{temp:.1f}°C"
    except:
        pass

    return "N/A"

# ======================  FETCH LIVE DATA ======================
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
hashrates = [0] * 128  # reduced for better accuracy
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
pool_diff = 429496729600000

log_lines = []
max_log = 40

connected = False

user = ""
password = "x"

# ======================  LOGGER ======================
def logg(msg):
    timestamp = time.strftime("%H:%M:%S")
    prefixed_msg = f"[{timestamp}] {msg}"
    log_lines.append(prefixed_msg)

# ======================  CONFIG ======================
BRAIINS_HOST = 'stratum.braiins.com'
BRAIINS_PORT = 3333

num_cores = os.cpu_count() or 24
max_threads = num_cores * 2  # 2 threads per core for optimal CPU load (Threadripper)
current_threads = 0

# ======================  SIGNAL ======================
def signal_handler(sig, frame):
    global fShutdown
    fShutdown = True
    logg("Shutting down...")

signal.signal(signal.SIGINT, signal_handler)

# ======================  MERKLE ROOT ======================
def calculate_merkle_root(extranonce2_local):
    coinbase = coinb1 + extranonce1 + extranonce2_local + coinb2
    coinbase_hash = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()
    merkle = coinbase_hash
    for branch in merkle_branch:
        merkle = hashlib.sha256(hashlib.sha256(merkle + binascii.unhexlify(branch)).digest()).digest()
    return binascii.hexlify(merkle[::-1]).decode()

# ======================  SUBMIT SHARE ======================
def submit_share(nonce):
    payload = {
        "id": None,
        "method": "mining.submit",
        "params": [user, job_id, extranonce2, ntime, f"{nonce:08x}"]
    }
    try:
        msg = json.dumps(payload) + "\n"
        sock.sendall(msg.encode())
        logg(f"Submitted share: nonce={nonce:08x}")
        resp = sock.recv(4096).decode(errors='ignore').strip()
        logg(f"Pool response: {resp}")
        if '"result":true' in resp or '"result": true' in resp:
            global accepted
            with lock:
                accepted += 1
            accepted_timestamps.append(time.time())
            logg("*** SHARE ACCEPTED ***")
        else:
            global rejected
            with lock:
                rejected += 1
            rejected_timestamps.append(time.time())
            logg("[!] Share rejected")
    except Exception as e:
        logg(f"[!] Submit error: {e}")

# ======================  MINING LOOP (real hashrate, max load) ======================
def bitcoin_miner(thread_id):
    global job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce2, pool_diff

    last_job_id = None
    hashes_done = 0
    last_report = time.time()

    while not fShutdown:
        if job_id is None:
            time.sleep(0.5)
            continue

        if job_id != last_job_id:
            last_job_id = job_id
            logg(f"Thread {thread_id}: New job {job_id}")

            extranonce2_int = (thread_id << 24) | (int(time.time() * 1000) & 0xFFFFFF)
            extranonce2 = f"{extranonce2_int:0{extranonce2_size * 2}x}"

        header_prefix = (
            binascii.unhexlify(version)[::-1] +
            binascii.unhexlify(prevhash)[::-1] +
            binascii.unhexlify(calculate_merkle_root(extranonce2))[::-1] +
            binascii.unhexlify(ntime)[::-1] +
            binascii.unhexlify(nbits)[::-1]
        )

        share_target_int = int(target if target else diff_to_target(pool_diff), 16)

        nonce = random.randint(0, 0xFFFFFFFF)
        for _ in range(16000000):  # max batch for high load
            if fShutdown or job_id != last_job_id:
                break

            nonce_bytes = struct.pack("<I", nonce)
            full_header = header_prefix + nonce_bytes

            hash_result = hashlib.sha256(hashlib.sha256(full_header).digest()).digest()
            hash_int = int.from_bytes(hash_result[::-1], 'big')

            hashes_done += 1

            if hash_int < share_target_int:
                nonce_hex = f"{nonce:08x}"
                logg(f"*** FOUND SHARE! Thread {thread_id} nonce {nonce_hex} ***")
                submit_share(nonce)

            nonce = (nonce + 1) & 0xFFFFFFFF

            if hashes_done % 50000 == 0:
                now = time.time()
                elapsed = now - last_report
                if elapsed > 0:
                    real_hr = int(50000 / elapsed)  # real H/s per thread
                    hashrates[thread_id] = real_hr
                last_report = now

# ======================  STRATUM ======================
def stratum_worker():
    global sock, job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce1, extranonce2_size, pool_diff, connected

    while not fShutdown:
        try:
            s = socket.socket()
            s.settimeout(120)
            s.connect((BRAIINS_HOST, BRAIINS_PORT))
            sock = s
            connected = True
            logg("Connected to Braiins Pool")

            s.sendall(b'{"id":1,"method":"mining.subscribe","params":["alfa5.py/1.0"]}\n')

            auth = {"id":2,"method":"mining.authorize","params":[user,password]}
            s.sendall((json.dumps(auth)+"\n").encode())

            last_keepalive = time.time()

            buf = b""
            while not fShutdown:
                current_time = time.time()
                if current_time - last_keepalive > 25:
                    try:
                        s.sendall(b'\n')
                        last_keepalive = current_time
                    except:
                        pass

                try:
                    data = s.recv(4096)
                except socket.timeout:
                    continue

                if not data:
                    connected = False
                    logg("[!] Connection lost – reconnecting...")
                    break

                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    logg(f"RX: {json.dumps(msg)}")
                    if "result" in msg and msg["id"] == 1:
                        extranonce1 = msg["result"][1]
                        extranonce2_size = msg["result"][2]
                        logg(f"Subscribed – extranonce1: {extranonce1}, size: {extranonce2_size}")
                    elif msg.get("method") == "mining.set_difficulty":
                        pool_diff = int(msg["params"][0])
                        target = diff_to_target(pool_diff)
                        logg(f"Difficulty set to {pool_diff}")
                    elif msg.get("method") == "mining.notify":
                        params = msg["params"]
                        if len(params) >= 9:
                            job_id = params[0]
                            prevhash = params[1]
                            coinb1 = params[2]
                            coinb2 = params[3]
                            merkle_branch = params[4]
                            version = params[5]
                            nbits = params[6]
                            ntime = params[7]
                            clean = params[8]
                            logg(f"New job {job_id} (clean: {clean})")
        except Exception as e:
            connected = False
            logg(f"[!] Connection error: {e} – retrying...")
            time.sleep(5)

# ======================  GRADUAL THREAD RAMP-UP ======================
def ramp_up_threads():
    global current_threads
    step = 4
    for target in range(step, max_threads + 1, step):
        if fShutdown:
            break
        while current_threads < target:
            current_threads += 1
            threading.Thread(target=bitcoin_miner, args=(current_threads - 1,), daemon=True).start()
            logg(f"Thread {current_threads}/{max_threads} started")
            time.sleep(0.5)
    logg(f"All {max_threads} threads running!")

# ======================  DISPLAY (real hashrate in MH/s) ======================
def display_worker():
    global log_lines
    stdscr = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)
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
            mh_s = total_hr / 1_000_000
            stdscr.addstr(4, 2, f"Hashrate  : {mh_s:.2f} MH/s ({total_hr:,} H/s)", curses.color_pair(1) | curses.A_BOLD)

            stdscr.addstr(5, 2, f"Threads   : {current_threads}/{max_threads}", curses.color_pair(4))

            cpu_temp = get_cpu_temp()
            stdscr.addstr(6, 2, f"Temp      : {cpu_temp}", curses.color_pair(3))

            a_min = sum(1 for t in accepted_timestamps if time.time() - t < 60)
            r_min = sum(1 for t in rejected_timestamps if time.time() - t < 60)
            stdscr.addstr(7, 2, f"Shares/min: {a_min} accepted / {r_min} rejected", curses.color_pair(1 if a_min > 0 else 3))

            stdscr.addstr(8, 2, f"Total     : {accepted} accepted / {rejected} rejected", curses.color_pair(3))

            stdscr.addstr(9, 2, f"Pool Diff : {pool_diff if pool_diff else 'Waiting...'}", curses.color_pair(4))

            stdscr.addstr(11, 0, "─" * w, curses.color_pair(3))

            start_y = 12
            for i, line in enumerate(log_lines[-max_log:]):
                if start_y + i >= h:
                    break
                color = 1 if "accepted" in line.lower() else (2 if "rejected" in line.lower() or "error" in line.lower() else 3)
                stdscr.addstr(start_y + i, 2, line[:w-4], curses.color_pair(6))

            stdscr.refresh()
            time.sleep(0.4)

    finally:
        curses.endwin()

# ======================  MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="alfa5.py - Braiins Pool CPU Miner")
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--worker", type=str, default="cpu002")
    args = parser.parse_args()

    user = f"{args.username}.{args.worker}"

    hashrates = [0] * max_threads

    boot_msgs = [
        " alfa5.py - Advanced Braiins Pool CPU Miner",
        "Initializing system...",
        f"CPU cores detected: {num_cores}",
        f"Target threads: {max_threads} (2 per core)",
        "Connecting to Braiins Pool...",
        "Starting stratum worker...",
        "Ramping up threads gradually..."
    ]
    for msg in boot_msgs:
        print(msg)
        time.sleep(0.7)

    threading.Thread(target=stratum_worker, daemon=True).start()
    time.sleep(4)

    threading.Thread(target=display_worker, daemon=True).start()

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
