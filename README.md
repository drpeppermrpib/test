#!/usr/bin/env python3

import multiprocessing as mp
import binascii
import hashlib
import socket
import time
import json
import os
import argparse
import struct
import subprocess
import threading
import curses
from flask import Flask, jsonify

# ================= CONFIGURATION =================
API_PORT = 60060
MAX_TEMP_C = 82.0      # Safe ceiling for 24/7 operation
VERSION_STRING = "AlfaUltra/4.0-TR-Optimized"

# ================= UTILS & MATH =================
def set_low_priority():
    """Sets the process to 'Idle' priority so the OS stays smooth."""
    try:
        # Linux/Mac
        os.nice(19) 
    except AttributeError:
        # Windows
        import psutil
        p = psutil.Process(os.getpid())
        p.nice(psutil.IDLE_PRIORITY_CLASS)

def swap_endian_hex(hex_str):
    if len(hex_str) % 2 != 0: hex_str = "0" + hex_str
    bs = binascii.unhexlify(hex_str)
    return binascii.hexlify(bs[::-1]).decode()

def diff_to_target(difficulty):
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    if difficulty <= 0: difficulty = 1
    return diff1 // int(difficulty)

def get_cpu_temp():
    try:
        # Optimized for Threadripper/AMD Tctl
        out = subprocess.check_output("sensors", shell=True).decode()
        for line in out.split("\n"):
            if "Tctl" in line or "Tdie" in line:
                return float(line.split("+")[1].split("°")[0].strip())
    except: pass
    return 0.0

# ================= WORKER PROCESS =================
def miner_process(worker_id, job_queue, result_queue, stop_flag, stats_array, current_diff, throttle_factor):
    # Apply low priority to each worker thread
    set_low_priority()
    
    extranonce2_int = worker_id * 1000000
    while not stop_flag.value:
        try:
            if job_queue.empty():
                time.sleep(0.1)
                continue
            
            job_data = job_queue.get()
            (job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, _) = job_data
            
            extranonce2_int += 1
            extranonce2 = f"{extranonce2_int:08x}"
            coinbase_bin = binascii.unhexlify(coinb1 + extranonce2 + coinb2)
            
            # Merkle Root
            merkle_root = hashlib.sha256(hashlib.sha256(coinbase_bin).digest()).digest()
            for branch in merkle_branch:
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + binascii.unhexlify(branch)).digest()).digest()
            
            # Pre-calc header to save cycles
            header_pre = (binascii.unhexlify(version)[::-1] + 
                          binascii.unhexlify(swap_endian_hex(prevhash)) + 
                          merkle_root + 
                          binascii.unhexlify(ntime)[::-1] + 
                          binascii.unhexlify(nbits)[::-1])
            
            target = diff_to_target(current_diff.value)
            nonce = worker_id * 20000000 # Wider nonce space
            
            # Optimized hashing loop
            while not stop_flag.value:
                if not job_queue.empty(): break
                
                # Dynamic Thermal Throttle
                if throttle_factor.value > 0:
                    time.sleep(throttle_factor.value)

                # Batch of 50k hashes
                for n in range(nonce, nonce + 50000):
                    header = header_pre + struct.pack("<I", n)
                    # Double SHA256 (Pure Python)
                    hash_res = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                    
                    if int.from_bytes(hash_res[::-1], "big") <= target:
                        result_queue.put({
                            "type": "share", 
                            "job_id": job_id, 
                            "extranonce2": extranonce2, 
                            "ntime": ntime, 
                            "nonce": f"{n:08x}"
                        })
                
                nonce += 50000
                stats_array[worker_id] += 50000
        except: time.sleep(1)

# ================= MAIN CONTROLLER =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--worker", default="001")
    parser.add_argument("--pool", default="ss.antpool.com")
    parser.add_argument("--port", type=int, default=3333)
    args = parser.parse_args()

    clean_pool = args.pool.replace("stratum+tcp://", "").split(":")[0]
    full_user = f"{args.username}.{args.worker}"
    
    # Initialize UI
    stdscr = curses.initscr()
    curses.noecho(); curses.cbreak(); stdscr.nodelay(1)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK) # Good
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)   # Danger
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)# Throttle
    
    num_threads = mp.cpu_count()
    job_queue = mp.Queue(); result_queue = mp.Queue()
    stop_flag = mp.Value('b', False); current_diff = mp.Value('d', 1024.0)
    throttle_factor = mp.Value('d', 0.0)
    stats_array = mp.Array('i', [0] * num_threads)
    
    workers = [mp.Process(target=miner_process, args=(i, job_queue, result_queue, stop_flag, stats_array, current_diff, throttle_factor)) for i in range(num_threads)]
    for w in workers: w.start()

    sock = None
    connected = False
    shares_accepted = 0; shares_rejected = 0
    log_msg = ["System: Priority set to LOW (Idle)"]
    last_keepalive = time.time()

    try:
        while True:
            # 1. Connection Management
            if not connected:
                try:
                    if sock: sock.close()
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    sock.connect((clean_pool, args.port))
                    sock.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": [VERSION_STRING]}) + "\n").encode())
                    sock.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [full_user, "x"]}) + "\n").encode())
                    connected = True
                except Exception as e:
                    log_msg.append(f"Network: Offline ({e})")
                    time.sleep(5); continue

            # 2. Server Keep-Alive
            if time.time() - last_keepalive > 60:
                try:
                    sock.sendall(b'{"id":0,"method":"mining.noop","params":[]}\n')
                    last_keepalive = time.time()
                except: connected = False

            # 3. Message Handling
            try:
                sock.settimeout(0.1)
                data = sock.recv(2048).decode()
                if data:
                    for line in data.split("\n"):
                        if not line.strip(): continue
                        resp = json.loads(line)
                        if resp.get("method") == "mining.notify":
                            while not job_queue.empty(): job_queue.get_nowait()
                            for _ in range(num_threads): job_queue.put(tuple(resp["params"]))
                        elif resp.get("method") == "mining.set_difficulty":
                            current_diff.value = float(resp["params"][0])
                        elif resp.get("id") == 4:
                            if resp.get("result"): shares_accepted += 1
                            else: shares_rejected += 1
                elif not data: connected = False
            except: pass

            # 4. Thermal Governor Logic
            temp = get_cpu_temp()
            if temp > MAX_TEMP_C:
                # Soft throttle: sleep worker threads 0.1s to 0.4s
                throttle_factor.value = min(0.4, (temp - MAX_TEMP_C) / 10.0)
            else:
                throttle_factor.value = 0.0

            # 5. Share Submission
            while not result_queue.empty():
                res = result_queue.get()
                payload = {"params": [full_user, res["job_id"], res["extranonce2"], res["ntime"], res["nonce"]], "id": 4, "method": "mining.submit"}
                try:
                    sock.sendall((json.dumps(payload) + "\n").encode())
                    log_msg.append(f"Share: {res['nonce']} submitted.")
                except: connected = False

            # UI Update
            stdscr.clear()
            h = sum(stats_array)
            for i in range(len(stats_array)): stats_array[i] = 0
            
            stdscr.addstr(0, 2, f"ALFA ULTRA v4.0 | TR {num_threads}-CORE | {full_user}", curses.A_BOLD)
            stdscr.addstr(2, 2, f"Status  : {'ONLINE' if connected else 'OFFLINE'}", curses.color_pair(1 if connected else 2))
            
            t_col = 2 if temp > MAX_TEMP_C else (3 if temp > MAX_TEMP_C - 5 else 1)
            stdscr.addstr(3, 2, f"Core Temp: {temp:.1f}°C", curses.color_pair(t_col))
            
            if throttle_factor.value > 0:
                stdscr.addstr(3, 30, f"[ THERMAL THROTTLE ACTIVE: {throttle_factor.value*100:.0f}% ]", curses.color_pair(3))

            stdscr.addstr(4, 2, f"Hashrate : {h/500:.2f} KH/s")
            stdscr.addstr(5, 2, f"Shares   : {shares_accepted} Accepted / {shares_rejected} Rejected")

            msg_y = 7
            for m in log_msg[-10:]:
                if msg_y < stdscr.getmaxyx()[0]-1:
                    stdscr.addstr(msg_y, 2, f"> {m}")
                    msg_y += 1
            
            stdscr.refresh()
            time.sleep(0.5)

    except KeyboardInterrupt: pass
    finally:
        stop_flag.value = True
        curses.endwin()
        for p in workers: p.terminate()

if __name__ == "__main__":
    main()
