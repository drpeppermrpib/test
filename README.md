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
MAX_TEMP_C = 79.0  # Throttle begins at 79°C
VERSION_STRING = "AlfaUltra/5.0-Dashboard-Edition"

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
                    return float(parts[1].split("°")[0].strip())
    except: pass
    return 0.0

# ================= WORKER PROCESS =================
def miner_process(worker_id, job_queue, result_queue, stop_flag, stats_array, current_diff, throttle_factor):
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
            
            merkle_root = hashlib.sha256(hashlib.sha256(coinbase_bin).digest()).digest()
            for branch in merkle_branch:
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + binascii.unhexlify(branch)).digest()).digest()
            
            header_pre = (binascii.unhexlify(version)[::-1] + 
                          binascii.unhexlify(swap_endian_hex(prevhash)) + 
                          merkle_root + 
                          binascii.unhexlify(ntime)[::-1] + 
                          binascii.unhexlify(nbits)[::-1])
            
            target = diff_to_target(current_diff.value)
            nonce = worker_id * 20000000
            batch_size = 40000 # Smaller batches for faster thermal response
            
            while not stop_flag.value:
                if not job_queue.empty(): break
                
                # Thermal Throttle Sleep
                if throttle_factor.value > 0:
                    time.sleep(throttle_factor.value)

                for n in range(nonce, nonce + batch_size):
                    header = header_pre + struct.pack("<I", n)
                    hash_res = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                    
                    if int.from_bytes(hash_res[::-1], "big") <= target:
                        result_queue.put({
                            "type": "share", 
                            "job_id": job_id, 
                            "extranonce2": extranonce2, 
                            "ntime": ntime, 
                            "nonce": f"{n:08x}"
                        })
                
                nonce += batch_size
                stats_array[worker_id] += batch_size
        except: time.sleep(1)

# ================= DASHBOARD API SERVER =================
def run_api(stats_dict):
    app = Flask(__name__)
    # Disable flask logging for cleaner terminal
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route('/')
    def status():
        return jsonify({
            "status": "Online",
            "miner": VERSION_STRING,
            "hashrate_khs": stats_dict.get("hashrate", 0) / 500,
            "temp_c": stats_dict.get("temp", 0),
            "shares_accepted": stats_dict.get("accepted", 0),
            "shares_rejected": stats_dict.get("rejected", 0),
            "current_job": stats_dict.get("job_id", "None"),
            "difficulty": stats_dict.get("diff", 0),
            "throttling": stats_dict.get("is_throttling", False)
        })

    app.run(host='0.0.0.0', port=API_PORT, threaded=True)

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
    
    stdscr = curses.initscr()
    curses.noecho(); curses.cbreak(); stdscr.nodelay(1)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)
    
    num_threads = mp.cpu_count()
    job_queue = mp.Queue(); result_queue = mp.Queue()
    stop_flag = mp.Value('b', False); current_diff = mp.Value('d', 1024.0)
    throttle_factor = mp.Value('d', 0.0)
    stats_array = mp.Array('i', [0] * num_threads)
    
    workers = [mp.Process(target=miner_process, args=(i, job_queue, result_queue, stop_flag, stats_array, current_diff, throttle_factor)) for i in range(num_threads)]
    for w in workers: w.start()

    # Shared stats for Dashboard WebUI
    api_stats = {"accepted": 0, "rejected": 0, "hashrate": 0, "temp": 0, "job_id": "None", "diff": 1024, "is_throttling": False}
    threading.Thread(target=run_api, args=(api_stats,), daemon=True).start()

    sock = None
    connected = False
    shares_accepted = 0; shares_rejected = 0
    log_msg = ["Initializing AlfaUltra v5.0..."]
    last_keepalive = time.time()

    try:
        while True:
            # 1. Connection
            if not connected:
                try:
                    if sock: sock.close()
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(10)
                    sock.connect((clean_pool, args.port))
                    sock.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": [VERSION_STRING]}) + "\n").encode())
                    sock.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [full_user, "x"]}) + "\n").encode())
                    connected = True
                    log_msg.append(f"Connected to {clean_pool}")
                except Exception as e:
                    log_msg.append(f"Connect Fail: {e}")
                    time.sleep(5); continue

            # 2. Keep-Alive
            if time.time() - last_keepalive > 60:
                try:
                    sock.sendall(b'{"id":0,"method":"mining.noop","params":[]}\n')
                    last_keepalive = time.time()
                except: connected = False

            # 3. Stratum Traffic
            try:
                sock.settimeout(0.2)
                data = sock.recv(4096).decode()
                if not data: connected = False; continue
                
                for line in data.split("\n"):
                    if not line.strip(): continue
                    resp = json.loads(line)
                    if resp.get("method") == "mining.notify":
                        params = resp["params"]
                        api_stats["job_id"] = params[0] # Update Dashboard
                        while not job_queue.empty(): job_queue.get_nowait()
                        for _ in range(num_threads): job_queue.put(tuple(params))
                        log_msg.append(f"Job Rcvd: {params[0][:8]}")
                    elif resp.get("method") == "mining.set_difficulty":
                        current_diff.value = float(resp["params"][0])
                        api_stats["diff"] = current_diff.value
                    elif resp.get("id") == 4:
                        if resp.get("result"): 
                            shares_accepted += 1
                            api_stats["accepted"] = shares_accepted
                            log_msg.append("*** SHARE ACCEPTED ***")
                        else: 
                            shares_rejected += 1
                            api_stats["rejected"] = shares_rejected
                            log_msg.append("[!] Rejected Share")
            except: pass

            # 4. Thermal Governor (79°C Limit)
            temp = get_cpu_temp()
            api_stats["temp"] = temp
            if temp > MAX_TEMP_C:
                throttle_factor.value = min(0.6, (temp - MAX_TEMP_C) / 8.0)
                api_stats["is_throttling"] = True
            else:
                throttle_factor.value = 0.0
                api_stats["is_throttling"] = False

            # 5. Share Submit
            while not result_queue.empty():
                res = result_queue.get()
                payload = {"params": [full_user, res["job_id"], res["extranonce2"], res["ntime"], res["nonce"]], "id": 4, "method": "mining.submit"}
                try: sock.sendall((json.dumps(payload) + "\n").encode())
                except: connected = False

            # UI Update
            stdscr.clear()
            h = sum(stats_array)
            api_stats["hashrate"] = h
            for i in range(len(stats_array)): stats_array[i] = 0
            
            title = f" ALFA ULTRA DASHBOARD | {full_user} | {clean_pool} "
            stdscr.addstr(0, 0, title.center(stdscr.getmaxyx()[1]), curses.color_pair(5) | curses.A_BOLD)

            stdscr.addstr(2, 2, f"Status    : {'ONLINE' if connected else 'OFFLINE'}", curses.color_pair(1 if connected else 2))
            
            t_col = 2 if temp > MAX_TEMP_C else (3 if temp > MAX_TEMP_C - 4 else 1)
            stdscr.addstr(3, 2, f"Temp      : {temp:.1f}°C | Throttle: {MAX_TEMP_C}°C", curses.color_pair(t_col))
            
            stdscr.addstr(4, 2, f"Hashrate  : {h/500:.2f} KH/s", curses.color_pair(1))
            stdscr.addstr(5, 2, f"Web UI    : http://localhost:{API_PORT}/", curses.color_pair(4))

            stdscr.addstr(7, 2, "Mining Logs:", curses.color_pair(4) | curses.A_UNDERLINE)
            msg_y = 8
            for m in log_msg[-35:]:
                if msg_y < stdscr.getmaxyx()[0] - 1:
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
