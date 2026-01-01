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
import subprocess
from datetime import datetime
from flask import Flask, jsonify
import threading

# ================= CONFIGURATION =================
API_PORT = 60060
MAX_TEMP_C = 85.0
VERSION_STRING = "AlfaUltra/2.0"

# ================= UTILS & MATH =================
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
        out = subprocess.check_output("sensors", shell=True).decode()
        for line in out.split("\n"):
            if "Tdie" in line or "Tctl" in line:
                parts = line.split("+")
                if len(parts) > 1:
                    temp_str = parts[1].split("°")[0].strip()
                    return float(temp_str)
    except: pass
    return 0.0

# ================= WORKER PROCESS =================
def miner_process(worker_id, job_queue, result_queue, stop_flag, stats_array, current_diff):
    extranonce2_int = worker_id * 1000000
    while not stop_flag.value:
        try:
            if job_queue.empty():
                time.sleep(0.1)
                continue
            
            job_data = job_queue.get()
            (job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs) = job_data
            
            extranonce2_int += 1
            extranonce2 = f"{extranonce2_int:08x}"
            coinbase_hex = coinb1 + extranonce2 + coinb2
            coinbase_bin = binascii.unhexlify(coinbase_hex)
            
            merkle_root = hashlib.sha256(hashlib.sha256(coinbase_bin).digest()).digest()
            for branch in merkle_branch:
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + binascii.unhexlify(branch)).digest()).digest()
            
            ver_bin = binascii.unhexlify(version)[::-1]
            prev_bin = binascii.unhexlify(swap_endian_hex(prevhash))
            ntime_bin = binascii.unhexlify(ntime)[::-1]
            nbits_bin = binascii.unhexlify(nbits)[::-1]
            header_pre = ver_bin + prev_bin + merkle_root + ntime_bin + nbits_bin
            
            target = diff_to_target(current_diff.value)
            nonce = worker_id * 10000000
            batch_size = 100000
            
            while not stop_flag.value:
                if not job_queue.empty(): break
                
                for n in range(nonce, nonce + batch_size):
                    header = header_pre + struct.pack("<I", n)
                    hash_res = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                    if int.from_bytes(hash_res[::-1], "big") <= target:
                        result_queue.put({"type": "share", "job_id": job_id, "extranonce2": extranonce2, "ntime": ntime, "nonce": f"{n:08x}"})
                
                nonce += batch_size
                stats_array[worker_id] += batch_size
        except: time.sleep(1)

# ================= API SERVER =================
def run_api(stats_dict):
    app = Flask(__name__)
    @app.route('/')
    def status(): return jsonify(stats_dict)
    app.run(host='0.0.0.0', port=API_PORT, threaded=True)

# ================= MAIN CONTROLLER =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--worker", default="001")
    parser.add_argument("--pool", default="stratum.antpool.com")
    parser.add_argument("--port", type=int, default=3333)
    args = parser.parse_args()

    clean_pool = args.pool.replace("stratum+tcp://", "").split(":")[0]
    full_user = f"{args.username}.{args.worker}"
    
    import curses
    stdscr = curses.initscr()
    curses.noecho(); curses.cbreak(); stdscr.nodelay(1)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)
    
    num_threads = mp.cpu_count()
    job_queue = mp.Queue(); result_queue = mp.Queue()
    stop_flag = mp.Value('b', False); current_diff = mp.Value('d', 1024.0)
    stats_array = mp.Array('i', [0] * num_threads)
    
    workers = [mp.Process(target=miner_process, args=(i, job_queue, result_queue, stop_flag, stats_array, current_diff)) for i in range(num_threads)]
    for w in workers: w.start()

    api_stats = {}
    threading.Thread(target=run_api, args=(api_stats,), daemon=True).start()

    sock = None
    connected = False
    shares_accepted = 0; shares_rejected = 0
    log_msg = ["Initializing AlfaUltra..."]

    try:
        while True:
            if not connected:
                try:
                    if sock: sock.close()
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(120)
                    sock.connect((clean_pool, args.port))
                    sock.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": [VERSION_STRING]}) + "\n").encode())
                    sock.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [full_user, "x"]}) + "\n").encode())
                    connected = True
                    log_msg.append(f"Connected to {clean_pool}:{args.port}")
                except Exception as e:
                    log_msg.append(f"Connect Fail: {e}")
                    time.sleep(5); continue

            try:
                sock.settimeout(0.2)
                data = sock.recv(4096).decode()
                if not data: connected = False; continue
                
                for line in data.split("\n"):
                    if not line.strip(): continue
                    resp = json.loads(line)
                    if resp.get("method") == "mining.notify":
                        while not job_queue.empty(): job_queue.get_nowait()
                        for _ in range(num_threads): job_queue.put(tuple(resp["params"]))
                        log_msg.append(f"New Job: {resp['params'][0][:8]}")
                    elif resp.get("method") == "mining.set_difficulty":
                        current_diff.value = float(resp["params"][0])
                        log_msg.append(f"Difficulty set to {current_diff.value}")
                    elif resp.get("id") == 4:
                        if resp.get("result"): 
                            shares_accepted += 1
                            log_msg.append("*** SHARE ACCEPTED ***")
                        else: 
                            shares_rejected += 1
                            log_msg.append("[!] Share rejected")
            except socket.timeout: pass
            except Exception: connected = False

            while not result_queue.empty():
                res = result_queue.get()
                payload = {"params": [full_user, res["job_id"], res["extranonce2"], res["ntime"], res["nonce"]], "id": 4, "method": "mining.submit"}
                try: sock.sendall((json.dumps(payload) + "\n").encode())
                except: connected = False

            # UI Refresh
            stdscr.clear()
            temp = get_cpu_temp()
            h = sum(stats_array)
            for i in range(len(stats_array)): stats_array[i] = 0
            
            # Pretty print title
            title = f" ALFA ULTRA | {full_user} | {clean_pool}:{args.port} "
            stdscr.addstr(0, 0, title.center(stdscr.getmaxyx()[1]), curses.color_pair(5) | curses.A_BOLD)

            # Pretty status
            status_text = "ONLINE" if connected else "OFFLINE"
            status_color = 1 if connected else 2
            stdscr.addstr(2, 2, f"Status    : {status_text}", curses.color_pair(status_color) | curses.A_BOLD)

            # Temp with pretty format
            stdscr.addstr(3, 2, f"Temp      : {temp:.1f}°C | Limit: {MAX_TEMP_C}°C", curses.color_pair(3))

            # Hashrate pretty
            kh_s = h / 500
            stdscr.addstr(4, 2, f"Speed     : {kh_s:.2f} KH/s", curses.color_pair(1) | curses.A_BOLD)

            # Shares pretty
            stdscr.addstr(5, 2, f"Shares    : Acc {shares_accepted} / Rej {shares_rejected}", curses.color_pair(1 if shares_accepted > shares_rejected else 2))

            # Log section with pretty header
            stdscr.addstr(7, 2, "Log:", curses.color_pair(4) | curses.A_UNDERLINE)
            msg_y = 8
            for m in log_msg[-8:]:
                color = 1 if "ACCEPTED" in m else (2 if "rejected" in m.lower() or "error" in m.lower() else 3)
                stdscr.addstr(msg_y, 2, f"> {m}", curses.color_pair(color))
                msg_y += 1
            
            api_stats.update({"hashrate": h, "temp": temp, "accepted": shares_accepted, "rejected": shares_rejected})
            stdscr.refresh()
            time.sleep(0.5)

    except KeyboardInterrupt: pass
    finally:
        stop_flag.value = True
        curses.endwin()
        for p in workers: p.terminate()

if __name__ == "__main__":
    main()
