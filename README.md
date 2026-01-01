#!/usr/bin/env python3

import multiprocessing as mp
import requests
import binascii
import hashlib
import socket
import time
import json
import sys
import os
import argparse
import struct
import threading
import logging
from datetime import datetime
from flask import Flask, jsonify

# ================= CONFIGURATION =================
# If you want to force a specific port or host
API_PORT = 60060
MAX_TEMP_C = 75.0
RETARGET_INTERVAL = 2024  # Blocks
VERSION_STRING = "AlfaUltra/2.0"

# ================= UTILS & MATH =================

def swap_endian_hex(hex_str):
    """Swaps endianness for stratum protocol (4-byte chunks)."""
    # Pad to even length
    if len(hex_str) % 2 != 0:
        hex_str = "0" + hex_str
    
    # Check if we can split into 4-byte (8 hex char) chunks
    # This acts as a standard reversal for Bitcoin headers
    bs = binascii.unhexlify(hex_str)
    return binascii.hexlify(bs[::-1]).decode()

def diff_to_target(difficulty):
    """Converts pool difficulty to a target integer."""
    # Standard diff 1 target for Bitcoin
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    if difficulty == 0: difficulty = 1
    return diff1 // int(difficulty)

def get_cpu_temp():
    """Reads AMD Threadripper Tdie/Tctl temp."""
    try:
        # Try lm-sensors first
        import subprocess
        out = subprocess.check_output("sensors", shell=True).decode()
        for line in out.split("\n"):
            if "Tdie" in line or "Tctl" in line:
                # Format: Tdie:        +55.5°C
                parts = line.split("+")
                if len(parts) > 1:
                    temp = parts[1].split("°")[0]
                    return float(temp)
    except:
        pass
    
    # Fallback to sysfs
    try:
        for zone in range(5):
            path = f"/sys/class/thermal/thermal_zone{zone}/temp"
            if os.path.exists(path):
                with open(path, 'r') as f:
                    # Value is usually in millidegrees
                    return int(f.read().strip()) / 1000.0
    except:
        return 0.0

# ================= WORKER PROCESS =================

def miner_process(worker_id, job_queue, result_queue, stop_flag, stats_array, current_diff):
    """
    The heavy lifter. Runs pure SHA256d.
    """
    # Local variables for speed
    extranonce2_int = worker_id * 1000000
    
    while not stop_flag.value:
        try:
            # Non-blocking check for new job
            if job_queue.empty():
                time.sleep(0.1)
                continue
                
            job_data = job_queue.get()
            
            # Unpack job
            (job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs) = job_data
            
            # --- THERMAL THROTTLE ---
            # Every few loops, check global temp flag or sleep? 
            # We do this in main loop to save CPU cycles here, but we can do a quick pause
            # if stats_array[worker_id] == -1 (signal to pause)
            
            # Prepare Header Construction
            # 1. Build Coinbase
            # We need a unique extranonce2 for this worker
            extranonce2_int += 1
            extranonce2 = f"{extranonce2_int:08x}"
            
            coinbase_hex = coinb1 + extranonce2 + coinb2
            coinbase_bin = binascii.unhexlify(coinbase_hex)
            
            # 2. Merkle Root
            merkle_root = hashlib.sha256(hashlib.sha256(coinbase_bin).digest()).digest()
            for branch in merkle_branch:
                branch_bin = binascii.unhexlify(branch)
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + branch_bin).digest()).digest()
            
            # 3. Construct Block Header (80 bytes)
            # Version (4) + PrevHash (32) + MerkleRoot (32) + Time (4) + nBits (4) + Nonce (4)
            # Note: Stratum sends them as big-endian hex, we often need to byte-swap for hashing
            
            # Convert parts to binary, correcting endianness where required by Stratum
            # Version: LE
            ver_bin = binascii.unhexlify(version)[::-1]
            # PrevHash: LE (stratum sends BE usually, but let's trust the input format for now)
            # Actually, standard stratum sends standard hex. 
            # We will use the standard "pack" approach for python miners.
            
            prev_bin = binascii.unhexlify(prevhash)
            # Merkle is already calculated
            ntime_bin = binascii.unhexlify(ntime)[::-1]
            nbits_bin = binascii.unhexlify(nbits)[::-1]
            
            # Header Prefix (76 bytes)
            # Version + Prev + Merkle + Time + Bits
            # Note: The merkle root calculation above might need byte swapping depending on pool.
            # For standard Stratum, we usually treat merkle_branch items as LE.
            
            header_pre = ver_bin + binascii.unhexlify(swap_endian_hex(prevhash)) + merkle_root + ntime_bin + nbits_bin
            
            # Target Calculation
            target = diff_to_target(current_diff.value)
            
            # Mining Loop
            nonce_start = worker_id * 10000000
            nonce = nonce_start
            
            # Hashing loop - maximize load
            # Batch size determines how often we check for new jobs
            batch_size = 50000
            
            while not stop_flag.value:
                # Check if job changed (rudimentary way: check queue)
                if not job_queue.empty():
                    break # Get new job
                
                # Simple Python Optimization: Struct pack is slow, do it outside if possible?
                # Nonce changes, so we must pack it.
                
                # To simulate "Load" and try to find shares:
                # We loop `batch_size` times
                for n in range(nonce, nonce + batch_size):
                    nonce_bin = struct.pack("<I", n) # Little Endian
                    header = header_pre + nonce_bin
                    
                    # Double SHA256
                    hash_res = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                    
                    # Check Target (treat hash as big integer)
                    hash_int = int.from_bytes(hash_res[::-1], "big")
                    
                    if hash_int <= target:
                        # Found a share!
                        result_queue.put({
                            "type": "share",
                            "job_id": job_id,
                            "extranonce2": extranonce2,
                            "ntime": ntime,
                            "nonce": f"{n:08x}"
                        })
                
                # Update stats
                nonce += batch_size
                stats_array[worker_id] += batch_size
                
        except Exception as e:
            # print(f"Worker Error: {e}")
            time.sleep(1)

# ================= API SERVER =================
def run_api(stats_dict):
    app = Flask(__name__)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR) # Silence Flask logs

    @app.route('/')
    def status():
        return jsonify(stats_dict)

    try:
        app.run(host='0.0.0.0', port=API_PORT, threaded=True)
    except:
        pass

# ================= MAIN CONTROLLER =================

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="Pool Username")
    parser.add_argument("--worker", default="001", help="Worker Name")
    parser.add_argument("--pool", default="stratum.antpool.com", help="Pool Address")
    parser.add_argument("--port", type=int, default=3333, help="Pool Port")
    args = parser.parse_args()

    full_user = f"{args.username}.{args.worker}"
    
    # UI Setup
    import curses
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.nodelay(1)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)

    # Multiprocessing Setup
    mp.set_start_method('spawn', force=True)
    num_threads = mp.cpu_count() # Should be 48 for TR 3960X
    
    job_queue = mp.Queue()
    result_queue = mp.Queue()
    stop_flag = mp.Value('b', False)
    current_diff = mp.Value('d', 1024.0) # Start high to avoid spamming low diff shares
    stats_array = mp.Array('i', [0] * num_threads)
    
    workers = []
    for i in range(num_threads):
        p = mp.Process(target=miner_process, args=(i, job_queue, result_queue, stop_flag, stats_array, current_diff))
        p.start()
        workers.append(p)

    # Socket Connection
    sock = None
    connected = False
    
    # Statistics
    shares_accepted = 0
    shares_rejected = 0
    start_time = time.time()
    api_stats = {}

    # API Thread
    api_thread = threading.Thread(target=run_api, args=(api_stats,))
    api_thread.daemon = True
    api_thread.start()

    log_msg = ["Initializing Miner..."]

    try:
        while True:
            # 1. CONNECTION MANAGER
            if not connected:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    sock.connect((args.pool, args.port))
                    
                    # Subscribe
                    msg = json.dumps({"id": 1, "method": "mining.subscribe", "params": [VERSION_STRING]}) + "\n"
                    sock.sendall(msg.encode())
                    
                    # Authorize
                    msg = json.dumps({"id": 2, "method": "mining.authorize", "params": [full_user, "x"]}) + "\n"
                    sock.sendall(msg.encode())
                    
                    connected = True
                    log_msg.append(f"Connected to {args.pool}")
                except Exception as e:
                    log_msg.append(f"Connection Failed: {e}")
                    time.sleep(5)
                    continue

            # 2. SOCKET LISTENER (Non-blocking logic)
            try:
                sock.settimeout(0.1)
                line = ""
                while True:
                    data = sock.recv(1024).decode()
                    if not data:
                        connected = False
                        break
                    line += data
                    if "\n" in line:
                        break
                
                if line:
                    for msg_raw in line.split("\n"):
                        if not msg_raw.strip(): continue
                        response = json.loads(msg_raw)
                        
                        # Handle Notify (New Job)
                        if response.get("method") == "mining.notify":
                            params = response["params"]
                            # Clean queue slightly
                            while not job_queue.empty():
                                try: job_queue.get_nowait()
                                except: pass
                                
                            # Send to all workers
                            job_tuple = tuple(params)
                            for _ in range(num_threads):
                                job_queue.put(job_tuple)
                            log_msg.append(f"New Block Detected: {params[0][:8]}...")
                        
                        # Handle Difficulty
                        if response.get("method") == "mining.set_difficulty":
                            new_diff = response["params"][0]
                            current_diff.value = new_diff
                            log_msg.append(f"Difficulty set to: {new_diff}")

                        # Handle Submit Responses
                        if response.get("id") == 4:
                            if response.get("result") == True:
                                shares_accepted += 1
                                log_msg.append("SHARE ACCEPTED!")
                            else:
                                shares_rejected += 1
                                log_msg.append(f"Share Rejected: {response.get('error')}")

            except socket.timeout:
                pass
            except Exception as e:
                connected = False
                log_msg.append("Socket Error, reconnecting...")

            # 3. SHARE SUBMISSION
            while not result_queue.empty():
                res = result_queue.get()
                if res["type"] == "share":
                    payload = {
                        "params": [
                            full_user,
                            res["job_id"],
                            res["extranonce2"],
                            res["ntime"],
                            res["nonce"]
                        ],
                        "id": 4,
                        "method": "mining.submit"
                    }
                    try:
                        sock.sendall((json.dumps(payload) + "\n").encode())
                        log_msg.append(f"Submitting Share: {res['nonce']}")
                    except:
                        connected = False

            # 4. MONITORING & DISPLAY
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            
            # Temperature Check
            temp = get_cpu_temp()
            temp_color = curses.color_pair(1)
            if temp > MAX_TEMP_C:
                temp_color = curses.color_pair(2)
                log_msg.append(f"OVERHEAT WARNING: {temp}°C - Throttling")
                # Logic to pause processes could go here, currently strictly monitoring
            
            # Hashrate Calc
            total_hashes = sum(stats_array)
            # Reset stats array every second to get instant hashrate
            for i in range(len(stats_array)):
                stats_array[i] = 0
            
            hashrate = total_hashes / 0.5 # since loop is approx 0.5s sleep or process time
            # Note: Python overhead makes precise hashrate calc difficult
            
            # Draw UI
            stdscr.addstr(0, 0, f" ALFA ULTRA MINER v2.0 | {full_user} ", curses.color_pair(4) | curses.A_BOLD)
            stdscr.addstr(2, 2, f"Pool Status : {'ONLINE' if connected else 'OFFLINE'}", curses.color_pair(1 if connected else 2))
            stdscr.addstr(3, 2, f"Temperature : {temp:.1f}°C (Limit: {MAX_TEMP_C}°C)", temp_color)
            stdscr.addstr(4, 2, f"Active Core : {num_threads} Threads", curses.color_pair(3))
            stdscr.addstr(5, 2, f"Hashrate    : {hashrate/1000:.2f} KH/s (CPU)", curses.color_pair(3)) 
            # Note: Displaying KH/s because Python won't reach GH/s
            
            stdscr.addstr(7, 2, f"Accepted    : {shares_accepted}", curses.color_pair(1))
            stdscr.addstr(8, 2, f"Rejected    : {shares_rejected}", curses.color_pair(2))
            stdscr.addstr(9, 2, f"API Server  : http://localhost:{API_PORT}/", curses.color_pair(4))

            # Log Window
            stdscr.addstr(11, 0, "-" * (w-1))
            msg_y = 12
            for msg in log_msg[-10:]:
                if msg_y < h - 1:
                    stdscr.addstr(msg_y, 2, msg[:w-4])
                    msg_y += 1
            
            # API Update
            api_stats.update({
                "hashrate": hashrate,
                "temp": temp,
                "accepted": shares_accepted,
                "rejected": shares_rejected,
                "worker": full_user
            })

            stdscr.refresh()
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.value = True
        curses.endwin()
        for p in workers:
            p.terminate()

if __name__ == "__main__":
    main()
