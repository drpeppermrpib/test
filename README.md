#!/usr/bin/env python3

import multiprocessing as mp
import binascii
import hashlib
import socket
import time
import json
import argparse
import struct
import subprocess
import threading
import curses
from flask import Flask, jsonify

# ================= CONFIGURATION =================
API_PORT = 60060
MAX_TEMP_C = 79.0  # Throttling limit
VERSION_STRING = "AlfaUltra/5.3-Fixed"

# ================= UTILS & MATH =================
def swap_endian_hex(hex_str):
    if len(hex_str) % 2 != 0: hex_str = "0" + hex_str
    return binascii.hexlify(binascii.unhexlify(hex_str)[::-1]).decode()

def diff_to_target(difficulty):
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    return diff1 // int(max(1, difficulty))

def get_cpu_temp():
    try:
        out = subprocess.check_output("sensors", shell=True).decode()
        for line in out.split("\n"):
            if "Tdie" in line or "Tctl" in line:
                parts = line.split("+")
                if len(parts) > 1: return float(parts[1].split("°")[0].strip())
    except: pass
    return 0.0

# ================= MINER WORKER =================
def miner_process(worker_id, job_queue, result_queue, stop_flag, stats_array, current_diff, throttle_factor):
    extranonce2_int = worker_id * 1000000
    while not stop_flag.value:
        try:
            if job_queue.empty():
                time.sleep(0.1); continue
            
            job_data = job_queue.get(timeout=1)
            (job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, _) = job_data
            
            extranonce2 = f"{extranonce2_int:08x}"
            extranonce2_int += 1
            
            coinbase = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinb1 + extranonce2 + coinb2)).digest()).digest()
            merkle_root = coinbase
            for branch in merkle_branch:
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + binascii.unhexlify(branch)).digest()).digest()
            
            header_pre = (binascii.unhexlify(version)[::-1] + binascii.unhexlify(swap_endian_hex(prevhash)) + 
                          merkle_root + binascii.unhexlify(ntime)[::-1] + binascii.unhexlify(nbits)[::-1])
            
            target = diff_to_target(current_diff.value)
            nonce = worker_id * 10000000
            
            while not stop_flag.value:
                if not job_queue.empty(): break
                if throttle_factor.value > 0: time.sleep(throttle_factor.value)

                for n in range(nonce, nonce + 20000):
                    header = header_pre + struct.pack("<I", n)
                    hash_res = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                    if int.from_bytes(hash_res[::-1], "big") <= target:
                        result_queue.put({"job_id": job_id, "extranonce2": extranonce2, "ntime": ntime, "nonce": f"{n:08x}"})
                
                nonce += 20000
                stats_array[worker_id] += 20000
        except: time.sleep(1)

# ================= WEB DASHBOARD =================
def run_api(api_data):
    app = Flask(__name__)
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    @app.route('/')
    def dashboard():
        return jsonify(api_data)
    app.run(host='0.0.0.0', port=API_PORT, threaded=True)

# ================= MAIN CONTROLLER =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--worker", default="001")
    parser.add_argument("--pool", default="ss.antpool.com")
    parser.add_argument("--port", type=int, default=3333)
    args = parser.parse_args()

    # Curses Setup
    stdscr = curses.initscr(); curses.noecho(); curses.cbreak(); stdscr.nodelay(1)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLUE)

    num_threads = mp.cpu_count()
    job_queue = mp.Queue(); result_queue = mp.Queue()
    stop_flag = mp.Value('b', False); current_diff = mp.Value('d', 1.0)
    throttle_factor = mp.Value('d', 0.0); stats_array = mp.Array('i', [0] * num_threads)

    workers = [mp.Process(target=miner_process, args=(i, job_queue, result_queue, stop_flag, stats_array, current_diff, throttle_factor)) for i in range(num_threads)]
    for w in workers: w.start()

    # Shared WebUI Data
    api_data = {"status": "Disconnected", "hashrate": 0, "temp": 0, "accepted": 0, "job": "None", "diff": 1}
    threading.Thread(target=run_api, args=(api_data,), daemon=True).start()

    logs = ["AlfaUltra Started. Waiting for Pool..."]
    sock = None
    connected = False
    
    try:
        while True:
            if not connected:
                try:
                    if sock: sock.close()
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((args.pool, args.port))
                    sock.sendall(json.dumps({"id":1,"method":"mining.subscribe","params":[VERSION_STRING]}) + "\n")
                    sock.sendall(json.dumps({"id":2,"method":"mining.authorize","params":[f"{args.username}.{args.worker}","x"]}) + "\n")
                    sock.sendall(json.dumps({"id":3,"method":"mining.suggest_difficulty","params":[1.0]}) + "\n")
                    connected = True
                    api_data["status"] = "Online"
                    logs.append(f"Connected to {args.pool}:{args.port}")
                except Exception as e:
                    api_data["status"] = "Offline"
                    logs.append(f"Connection Error: {e}"); time.sleep(5); continue

            # Handle Incoming Data
            try:
                sock.settimeout(0.1)
                data = sock.recv(4096).decode()
                for line in data.split("\n"):
                    if not line.strip(): continue
                    msg = json.loads(line)
                    if msg.get("method") == "mining.notify":
                        params = msg["params"]
                        api_data["job"] = params[0][:8]
                        while not job_queue.empty(): job_queue.get()
                        for _ in range(num_threads): job_queue.put(tuple(params))
                        logs.append(f"New Job: {params[0][:8]}")
                    elif msg.get("method") == "mining.set_difficulty":
                        current_diff.value = float(msg["params"][0])
                        api_data["diff"] = current_diff.value
                    elif msg.get("id") == 4 and msg.get("result") == True:
                        api_data["accepted"] += 1
                        logs.append(">>> SHARE ACCEPTED! <<<")
            except socket.timeout: pass
            except: connected = False

            # Thermal Control
            temp = get_cpu_temp()
            api_data["temp"] = temp
            throttle_factor.value = max(0.0, (temp - MAX_TEMP_C) / 5.0) if temp > MAX_TEMP_C else 0.0

            # Submit Shares
            while not result_queue.empty():
                res = result_queue.get()
                payload = {"params": [f"{args.username}.{args.worker}", res["job_id"], res["extranonce2"], res["ntime"], res["nonce"]], "id": 4, "method": "mining.submit"}
                sock.sendall(json.dumps(payload) + "\n")
                logs.append(f"Submitting Share: {res['nonce']}")

            # UI Update
            stdscr.clear()
            h = sum(stats_array); api_data["hashrate"] = h
            for i in range(len(stats_array)): stats_array[i] = 0
            
            stdscr.addstr(0, 0, f" ALFA ULTRA | {args.username}.{args.worker} ".center(stdscr.getmaxyx()[1]), curses.color_pair(4))
            stdscr.addstr(2, 2, f"Status: {api_data['status']}", curses.color_pair(1 if connected else 2))
            stdscr.addstr(3, 2, f"Hashrate: {h/500:.2f} KH/s | Temp: {temp:.1f}°C (Limit 79°C)", curses.color_pair(3))
            stdscr.addstr(4, 2, f"Accepted Shares: {api_data['accepted']} | Current Job: {api_data['job']}", curses.color_pair(1))
            stdscr.addstr(5, 2, f"Web UI: http://localhost:{API_PORT}/", curses.color_pair(3))
            
            for i, log in enumerate(logs[-15:]):
                stdscr.addstr(7 + i, 2, f"> {log}")
            
            stdscr.refresh(); time.sleep(0.5)

    except KeyboardInterrupt: pass
    finally:
        stop_flag.value = True
        curses.endwin()
        for p in workers: p.terminate()

if __name__ == "__main__":
    main()
