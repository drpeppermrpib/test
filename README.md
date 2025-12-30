#!/usr/bin/env python3

# alfa5.py - Braiins Pool CPU Miner (Fixed for HiveOS/Threadripper)
# Fixes: proper Stratum V1 protocol, correct share submission, CPU temp for AMD, hashrate calculation

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

# ======================  CPU TEMPERATURE (AMD Threadripper) ======================
def get_cpu_temp():
    try:
        # AMD k10temp driver (Threadripper standard)
        result = subprocess.check_output(["sensors", "-u"], text=True, stderr=subprocess.DEVNULL)
        for line in result.splitlines():
            if 'temp1_input' in line or 'Tctl_input' in line:
                temp = float(line.split(':').strip())
                return f"{temp:.1f}°C"
    except:
        pass

    # Fallback: hwmon
    try:
        for hwmon in os.listdir('/sys/class/hwmon'):
            name_path = f"/sys/class/hwmon/{hwmon}/name"
            if os.path.exists(name_path):
                with open(name_path) as f:
                    name = f.read().strip()
                if 'k10temp' in name or 'zenpower' in name:
                    temp_path = f"/sys/class/hwmon/{hwmon}/temp1_input"
                    if os.path.exists(temp_path):
                        with open(temp_path) as f:
                            temp = int(f.read().strip()) / 1000
                            return f"{temp:.1f}°C"
    except:
        pass
    return "N/A"

# ======================  GLOBALS ======================
fShutdown = False
hashrates =  * 512
accepted = rejected = 0
accepted_timestamps = []
rejected_timestamps = []
lock = threading.Lock()

# job data
job_id = prevhash = coinb1 = coinb2 = None
merkle_branch = []
version = nbits = ntime = None
extranonce1 = "00000000"
extranonce2_size = 4
sock = None
target = None
pool_diff = 1  # default

# log lines
log_lines = []
max_log = 40

# connection status
connected = False

# ======================  LOGGER ======================
def logg(msg):
    timestamp = time.strftime("%H:%M:%S")
    prefixed_msg = f"[{timestamp}] {msg}"
    with lock:
        log_lines.append(prefixed_msg)
        if len(log_lines) > 200:
            log_lines.pop(0)

logg("alfa5.py starting...")

# ======================  CONFIG ======================
BRAIINS_HOST = 'stratum.braiins.com'
BRAIINS_PORT = 3333

num_cores = os.cpu_count() or 24
max_threads = num_cores * 2  # Threadripper: 2 threads per core is optimal
current_threads = 0

# ======================  SIGNAL ======================
def signal_handler(sig, frame):
    global fShutdown
    fShutdown = True
    logg("Shutting down...")

signal.signal(signal.SIGINT, signal_handler)

# ======================  MERKLE ROOT ======================
def calc_merkle_root(coinbase_hash_bin):
    merkle = coinbase_hash_bin
    for branch in merkle_branch:
        merkle = hashlib.sha256(hashlib.sha256(merkle + binascii.unhexlify(branch)).digest()).digest()
    return merkle

# ======================  SUBMIT SHARE ======================
def submit_share(nonce_hex, extranonce2_used, ntime_used):
    global accepted, rejected
    
    payload = {
        "id": int(time.time() * 1000),
        "method": "mining.submit",
        "params": [user, job_id, extranonce2_used, ntime_used, nonce_hex]
    }
    
    try:
        msg = json.dumps(payload) + "\n"
        sock.sendall(msg.encode())
        logg(f"Submitted: nonce={nonce_hex}")
        
        # Read response
        resp = sock.recv(4096).decode().strip()
        logg(f"Response: {resp}")
        
        if '"result":true' in resp or '"result": true' in resp:
            with lock:
                accepted += 1
                accepted_timestamps.append(time.time())
            logg("*** SHARE ACCEPTED ***")
        else:
            with lock:
                rejected += 1
                rejected_timestamps.append(time.time())
            logg(f"[!] Share rejected: {resp}")
    except Exception as e:
        logg(f"[!] Submit error: {e}")

# ======================  MINING LOOP (FIXED) ======================
def bitcoin_miner(thread_id):
    global job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce1, pool_diff

    last_job_id = None
    hashes_done = 0
    last_report = time.time()

    while not fShutdown:
        # Wait for valid job
        if None in (job_id, version, prevhash, nbits, ntime, coinb1, coinb2):
            time.sleep(0.5)
            continue

        # New job detected
        if job_id != last_job_id:
            last_job_id = job_id
            logg(f"Thread {thread_id}: New job {job_id}")

        # Generate unique extranonce2 for this thread
        extranonce2_int = (thread_id << 24) | (int(time.time()) & 0xFFFFFF)
        extranonce2 = f"{extranonce2_int:0{extranonce2_size * 2}x}"

        # Build coinbase transaction
        coinbase_tx = coinb1 + extranonce1 + extranonce2 + coinb2
        coinbase_hash = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase_tx)).digest()).digest()

        # Calculate merkle root
        merkle_root = calc_merkle_root(coinbase_hash)
        merkle_root_hex = binascii.hexlify(merkle_root[::-1]).decode()

        # Build block header (80 bytes)
        header = (
            binascii.unhexlify(version)[::-1] +
            binascii.unhexlify(prevhash)[::-1] +
            merkle_root +
            binascii.unhexlify(ntime)[::-1] +
            binascii.unhexlify(nbits)[::-1]
        )

        # Get target
        share_target = target if target else diff_to_target(pool_diff)
        target_int = int(share_target, 16)

        # Mining loop
        nonce = random.randint(0, 0xFFFFFFFF)
        batch_size = 1000000
        
        for _ in range(batch_size):
            if fShutdown or job_id != last_job_id:
                break

            # Try nonce
            nonce_bytes = struct.pack('<I', nonce)
            block_header = header + nonce_bytes
            
            hash_result = hashlib.sha256(hashlib.sha256(block_header).digest()).digest()
            hash_int = int.from_bytes(hash_result[::-1], 'big')

            hashes_done += 1

            # Check if valid share
            if hash_int < target_int:
                nonce_hex = f"{nonce:08x}"
                logg(f"*** FOUND SHARE! Thread {thread_id}, nonce: {nonce_hex} ***")
                submit_share(nonce_hex, extranonce2, ntime)

            nonce = (nonce + 1) & 0xFFFFFFFF

            # Update hashrate every 100k hashes
            if hashes_done % 100000 == 0:
                now = time.time()
                elapsed = now - last_report
                if elapsed > 0:
                    hr = int(100000 / elapsed)
                    hashrates[thread_id] = hr
                    last_report = now

# ======================  STRATUM (FIXED) ======================
def stratum_worker():
    global sock, job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce1, extranonce2_size, pool_diff, connected

    while not fShutdown:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(30)
            s.connect((BRAIINS_HOST, BRAIINS_PORT))
            sock = s
            connected = True
            logg(f"Connected to {BRAIINS_HOST}:{BRAIINS_PORT}")

            # Subscribe
            subscribe = {"id": 1, "method": "mining.subscribe", "params": ["alfa5.py/1.0"]}
            s.sendall((json.dumps(subscribe) + "\n").encode())

            # Authorize
            time.sleep(0.5)
            auth = {"id": 2, "method": "mining.authorize", "params": [user, password]}
            s.sendall((json.dumps(auth) + "\n").encode())

            buf = b""
            while not fShutdown:
                data = s.recv(8192)
                if not data:
                    connected = False
                    logg("[!] Connection lost")
                    break
                
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    if not line.strip():
                        continue
                    
                    try:
                        msg = json.loads(line)
                        
                        # Subscribe response
                        if "result" in msg and msg.get("id") == 1:
                            if msg["result"]:
                                extranonce1 = msg["result"]
                                extranonce2_size = msg["result"]
                                logg(f"Subscribed: extranonce1={extranonce1}, size={extranonce2_size}")
                        
                        # Set difficulty
                        elif msg.get("method") == "mining.set_difficulty":
                            pool_diff = float(msg["params"])
                            target = diff_to_target(pool_diff)
                            logg(f"Difficulty set: {pool_diff}")
                        
                        # New job
                        elif msg.get("method") == "mining.notify":
                            params = msg["params"]
                            job_id = params
                            prevhash = params
                            coinb1 = params
                            coinb2 = params
                            merkle_branch = params
                            version = params
                            nbits = params
                            ntime = params
                            logg(f"New job: {job_id}")
                        
                        # Submit response
                        elif "result" in msg and msg.get("id") != 1 and msg.get("id") != 2:
                            pass  # Already handled in submit_share
                            
                    except json.JSONDecodeError as e:
                        logg(f"JSON error: {e}")
                    except Exception as e:
                        logg(f"Parse error: {e}")

        except Exception as e:
            connected = False
            logg(f"[!] Connection error: {e}")
            time.sleep(5)
        finally:
            if sock:
                try:
                    sock.close()
                except:
                    pass
                sock = None

# ======================  GRADUAL THREAD RAMP-UP ======================
def ramp_up_threads():
    global current_threads
    
    # Start with 1/4 of threads
    initial = max(1, max_threads // 4)
    
    for i in range(initial):
        if fShutdown:
            break
        current_threads += 1
        threading.Thread(target=bitcoin_miner, args=(current_threads - 1,), daemon=True).start()
        logg(f"Thread {current_threads}/{max_threads} started")
        time.sleep(0.5)
    
    # Ramp to full over 30 seconds
    remaining = max_threads - initial
    if remaining > 0:
        delay = 30.0 / remaining
        for i in range(remaining):
            if fShutdown:
                break
            time.sleep(delay)
            current_threads += 1
            threading.Thread(target=bitcoin_miner, args=(current_threads - 1,), daemon=True).start()
            logg(f"Thread {current_threads}/{max_threads} started")
    
    logg(f"All {max_threads} threads active!")

# ======================  DISPLAY ======================
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

            title = " alfa5.py - Braiins Pool CPU Miner (Threadripper Optimized) "
            stdscr.addstr(0, 0, title.center(w), curses.color_pair(5) | curses.A_BOLD)

            status = "ONLINE" if connected else "OFFLINE"
            color = 1 if connected else 2
            stdscr.addstr(2, 2, f"Status    : {status}", curses.color_pair(color) | curses.A_BOLD)

            try:
                block_height = requests.get('https://mempool.space/api/blocks/tip/height', timeout=2).text
            except:
                block_height = "???"
            stdscr.addstr(3, 2, f"Block     : {block_height}", curses.color_pair(3))

            total_hr = sum(hashrates)
            mh_s = total_hr / 1_000_000
            stdscr.addstr(4, 2, f"Hashrate  : {mh_s:.2f} MH/s ({total_hr:,} H/s)", curses.color_pair(1) | curses.A_BOLD)

            stdscr.addstr(5, 2, f"Threads   : {current_threads}/{max_threads}", curses.color_pair(4))

            cpu_temp = get_cpu_temp()
            stdscr.addstr(6, 2, f"CPU Temp  : {cpu_temp}", curses.color_pair(3))

            a_min =
