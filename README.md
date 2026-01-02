#!/usr/bin/env python3
import socket
import ssl
import json
import time
import threading
import multiprocessing as mp
import curses
import argparse
import binascii
import struct
import hashlib
import subprocess
from datetime import datetime

# ================= USER CONFIGURATION =================
# PRESET: SOLO MINING (Targeting Blocks for your Wallet)
# To switch to FPPS, uncomment the FPPS lines and comment out the SOLO lines.

# --- OPTION 1: SOLO (Lottery Mode - You keep the block) ---
POOL_URL = "solo.braiins.com"
POOL_PORT = 3333 # Use 443 for SSL, 3333 for TCP
# Your specific BTC wallet for solo rewards:
POOL_USER = "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e" 
POOL_PASS = "x"

# --- OPTION 2: FPPS (Steady Pay - Pool keeps the block) ---
# POOL_URL = "stratum.braiins.com"
# POOL_PORT = 3333
# POOL_USER = "drpeppermrpib.rlm"
# POOL_PASS = "x"

# ================= CORE MINER =================
def get_cpu_temp():
    """Reads CPU temperature from lm-sensors"""
    try:
        res = subprocess.check_output("sensors", shell=True).decode()
        for line in res.split("\n"):
            if "Tdie" in line or "Tctl" in line or "Package id 0" in line:
                return float(line.split("+")[1].split("°")[0].strip())
    except:
        return 0.0

def miner_worker(id, job_queue, result_queue, stop_event, stats, current_diff):
    """The hashing engine"""
    nonce_start = id * 100000000
    
    while not stop_event.is_set():
        try:
            if job_queue.empty():
                time.sleep(0.1)
                continue

            # Get latest job
            job = job_queue.get()
            job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs = job

            # Calculate Target
            # Difficulty 1 = 0x00000000FFFF0000...
            diff = current_diff.value
            if diff == 0: diff = 1
            target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000 // int(diff)

            # Extranonce and Coinbase
            extranonce2 = struct.pack('<I', nonce_start & 0xFFFFFFFF).hex() 
            coinbase_bin = binascii.unhexlify(coinb1 + extranonce2 + coinb2)
            coinbase_hash = hashlib.sha256(hashlib.sha256(coinbase_bin).digest()).digest()

            merkle_root = coinbase_hash
            for branch in merkle_branch:
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + binascii.unhexlify(branch)).digest()).digest()

            # Block Header Construction
            # version (4) + prevhash (32) + merkle (32) + time (4) + nbits (4) + nonce (4)
            header_prefix = (
                binascii.unhexlify(version)[::-1] +
                binascii.unhexlify(prevhash)[::-1] +
                merkle_root +
                binascii.unhexlify(ntime)[::-1] +
                binascii.unhexlify(nbits)[::-1]
            )

            # Mining Loop
            nonce = nonce_start
            batch_size = 50000
            
            while not stop_event.is_set():
                if not job_queue.empty() and clean_jobs:
                    break # New job arrived, restart

                # Hash Batch
                for n in range(nonce, nonce + batch_size):
                    header = header_prefix + struct.pack('<I', n)
                    block_hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                    
                    # Check Target (Reverse hash for comparison)
                    hash_int = int.from_bytes(block_hash[::-1], 'big')
                    
                    if hash_int <= target:
                        result_queue.put({
                            "job_id": job_id,
                            "extranonce2": extranonce2,
                            "ntime": ntime,
                            "nonce": f"{n:08x}",
                            "result": block_hash[::-1].hex()
                        })
                
                nonce += batch_size
                stats[id] += batch_size # Update hashrate counter

        except Exception as e:
            pass

# ================= NETWORK & UI =================
def run_miner(stdscr):
    # Curses Setup
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK) # Good
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)# Warn
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)   # Bad
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Info
    curses.curs_set(0)
    stdscr.nodelay(True)

    # Shared Data
    manager = mp.Manager()
    job_queue = manager.Queue()
    result_queue = manager.Queue()
    stop_event = mp.Event()
    current_diff = mp.Value('d', 1.0)
    stats = mp.Array('i', [0] * mp.cpu_count())

    # Start Workers
    workers = []
    for i in range(mp.cpu_count()):
        p = mp.Process(target=miner_worker, args=(i, job_queue, result_queue, stop_event, stats, current_diff))
        p.start()
        workers.append(p)

    # State Variables
    sock = None
    connected = False
    logs = []
    accepted_shares = 0
    rejected_shares = 0
    start_time = time.time()
    last_response = time.time()

    def log(msg, type="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        logs.append(f"[{timestamp}] {msg}")
        if len(logs) > 50: logs.pop(0)

    log(f"Starting RLM v2.0 - Target: {POOL_URL}:{POOL_PORT}")
    log(f"User: {POOL_USER[:10]}...")

    while True:
        try:
            # 1. Connection Handler
            if not connected:
                try:
                    log("Connecting...", "WARN")
                    raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    raw_sock.settimeout(5)
                    
                    # AUTO SSL DETECT
                    if POOL_PORT == 443:
                        context = ssl.create_default_context()
                        context.check_hostname = False
                        context.verify_mode = ssl.CERT_NONE
                        sock = context.wrap_socket(raw_sock, server_hostname=POOL_URL)
                        log("SSL Handshake Complete", "INFO")
                    else:
                        sock = raw_sock
                        
                    sock.connect((POOL_URL, POOL_PORT))
                    
                    # Stratum Handshake
                    payload = json.dumps({"id": 1, "method": "mining.subscribe", "params": ["RLM/2.0"]}) + "\n"
                    sock.sendall(payload.encode())
                    
                    # Authorize
                    payload = json.dumps({"id": 2, "method": "mining.authorize", "params": [POOL_USER, POOL_PASS]}) + "\n"
                    sock.sendall(payload.encode())
                    
                    connected = True
                    log("Connected & Authorized!", "GOOD")
                except Exception as e:
                    log(f"Connection Failed: {e}", "BAD")
                    time.sleep(5)
                    continue

            # 2. Network IO (Non-blocking)
            try:
                sock.settimeout(0.1)
                data = sock.recv(4096).decode()
                if not data:
                    connected = False
                    log("Disconnected (EOF)", "BAD")
                    continue
                    
                for line in data.split('\n'):
                    if not line: continue
                    msg = json.loads(line)
                    
                    # Method: Notify (New Job)
                    if msg.get('method') == 'mining.notify':
                        params = msg['params']
                        job_id = params[0]
                        clean_jobs = params[8]
                        
                        # Pack for workers
                        # job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean
                        job_data = (params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], clean_jobs)
                        
                        # If clean_jobs=True, empty the queue to force workers to switch
                        if clean_jobs:
                            while not job_queue.empty(): job_queue.get()
                            
                        for _ in range(mp.cpu_count()):
                            job_queue.put(job_data)
                            
                        log(f"New Job: {job_id[:8]}", "INFO")

                    # Method: Set Difficulty
                    elif msg.get('method') == 'mining.set_difficulty':
                        new_diff = msg['params'][0]
                        current_diff.value = new_diff
                        log(f"Difficulty set to: {new_diff}", "WARN")

                    # ID 4: Our Share Submission Response
                    elif msg.get('id') == 4:
                        last_response = time.time()
                        if msg.get('result') == True:
                            accepted_shares += 1
                            log(">>> SHARE ACCEPTED <<<", "GOOD")
                        else:
                            rejected_shares += 1
                            err = msg.get('error')
                            log(f"Share Rejected: {err}", "BAD")

            except socket.timeout:
                pass
            except Exception as e:
                log(f"Socket Error: {e}", "BAD")
                connected = False

            # 3. Submit Shares
            while not result_queue.empty():
                res = result_queue.get()
                # Stratum submit format: user, job_id, extranonce2, ntime, nonce
                params = [POOL_USER, res['job_id'], res['extranonce2'], res['ntime'], res['nonce']]
                payload = json.dumps({"id": 4, "method": "mining.submit", "params": params}) + "\n"
                try:
                    sock.sendall(payload.encode())
                    log(f"Submitting Nonce: {res['nonce']}", "INFO")
                except:
                    connected = False

            # 4. DRAW UI
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            
            # Header
            title = f" RLM MINER v2.0 | {POOL_URL} "
            stdscr.addstr(0, 0, title.center(w), curses.A_REVERSE | curses.color_pair(4))
            
            # Status Box
            status_color = curses.color_pair(1) if connected else curses.color_pair(3)
            stdscr.addstr(2, 2, f"Status:      {'ONLINE' if connected else 'OFFLINE'}", status_color)
            
            cpu_temp = get_cpu_temp()
            temp_color = curses.color_pair(3) if cpu_temp > 80 else curses.color_pair(1)
            stdscr.addstr(3, 2, f"Temp:        {cpu_temp:.1f}°C", temp_color)
            
            # Hashrate Calc
            total_hashes = sum(stats)
            elapsed = time.time() - start_time
            hr = total_hashes / elapsed if elapsed > 0 else 0
            
            stdscr.addstr(2, 40, f"Hashrate:    {hr/1000:.2f} kH/s", curses.color_pair(4))
            stdscr.addstr(3, 40, f"Difficulty:  {current_diff.value}", curses.color_pair(2))
            
            # Shares
            stdscr.addstr(5, 2, f"Accepted:    {accepted_shares}", curses.color_pair(1))
            stdscr.addstr(6, 2, f"Rejected:    {rejected_shares}", curses.color_pair(3))
            
            # Logs Window
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(8, 2, " SYSTEM LOGS ", curses.A_REVERSE)
            
            max_logs = h - 10
            display_logs = logs[-max_logs:]
            for i, line in enumerate(display_logs):
                color = curses.color_pair(4)
                if "ACCEPTED" in line: color = curses.color_pair(1)
                elif "Rejected" in line or "Failed" in line or "OFFLINE" in line: color = curses.color_pair(3)
                
                try:
                    stdscr.addstr(10 + i, 2, line[:w-4], color)
                except: pass

            stdscr.refresh()
            time.sleep(0.1)

        except KeyboardInterrupt:
            break
            
    stop_event.set()
    for p in workers: p.terminate()

if __name__ == "__main__":
    curses.wrapper(run_miner)
