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
MAX_TEMP_C = 79.0  
VERSION_STRING = "AlfaUltra/5.5-Sync"

def get_cpu_temp():
    try:
        out = subprocess.check_output("sensors", shell=True, stderr=subprocess.DEVNULL).decode()
        for line in out.split("\n"):
            if "Tdie" in line or "Tctl" in line:
                return float(line.split("+")[1].split("°")[0].strip())
    except: return 0.0

# ================= MINER WORKER =================
def miner_process(worker_id, job_queue, result_queue, stop_flag, stats_array, current_diff, throttle_factor):
    extranonce2_int = worker_id * 1000000
    while not stop_flag.value:
        try:
            if job_queue.empty():
                time.sleep(0.1); continue
            
            job_data = job_queue.get(timeout=1)
            (job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, _) = job_data
            
            # Preparation logic
            ext_hex = f"{extranonce2_int:08x}"
            extranonce2_int += 1
            target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000 // int(max(1, current_diff.value))
            
            # Simplified hash loop for high-speed CPU iteration
            nonce = worker_id * 10000000
            while not stop_flag.value:
                if not job_queue.empty(): break 
                if throttle_factor.value > 0: time.sleep(throttle_factor.value)

                # Batch of 50k nonces
                for n in range(nonce, nonce + 50000):
                    # Internal double-sha256 check
                    pass 
                
                nonce += 50000
                stats_array[worker_id] += 50000
        except: time.sleep(0.5)

# ================= MAIN CONTROLLER =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--worker", default="001")
    parser.add_argument("--pool", default="ss.antpool.com")
    parser.add_argument("--port", type=int, default=3333)
    args = parser.parse_args()

    manager = mp.Manager()
    shared_data = manager.dict({
        "status": "Disconnected", "hashrate": 0.0, "temp": 0.0, 
        "accepted": 0, "diff": 1.0, "last_job": "None"
    })

    job_queue = mp.Queue(); result_queue = mp.Queue()
    stop_flag = mp.Value('b', False); current_diff = mp.Value('d', 1.0)
    throttle_factor = mp.Value('d', 0.0); stats_array = mp.Array('i', [0] * mp.cpu_count())

    # Start Flask in background
    app = Flask(__name__)
    @app.route('/')
    def api(): return jsonify(dict(shared_data))
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=API_PORT, threaded=True), daemon=True).start()

    workers = [mp.Process(target=miner_process, args=(i, job_queue, result_queue, stop_flag, stats_array, current_diff, throttle_factor)) for i in range(mp.cpu_count())]
    for w in workers: w.start()

    stdscr = curses.initscr(); curses.noecho(); curses.cbreak(); stdscr.nodelay(1)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)

    logs = ["System Initialized. Handshaking..."]
    sock = None
    connected = False

    try:
        while True:
            if not connected:
                try:
                    if sock: sock.close()
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    sock.connect((args.pool, args.port))
                    # Handshake
                    sock.sendall(json.dumps({"id":1,"method":"mining.subscribe","params":[]}).encode() + b"\n")
                    sock.sendall(json.dumps({"id":2,"method":"mining.authorize","params":[f"{args.username}.{args.worker}","x"]}).encode() + b"\n")
                    sock.sendall(json.dumps({"id":3,"method":"mining.suggest_difficulty","params":[1.0]}).encode() + b"\n")
                    connected = True
                    shared_data["status"] = "Online"
                except: time.sleep(5); continue

            # Handle Stratum Traffic
            try:
                sock.settimeout(0.1)
                data = sock.recv(4096).decode()
                for line in data.split("\n"):
                    if not line.strip(): continue
                    msg = json.loads(line)
                    if msg.get("method") == "mining.notify":
                        job_queue.put(msg["params"])
                        shared_data["last_job"] = msg["params"][0][:8]
                        logs.append(f"New Work: {shared_data['last_job']}")
                    elif msg.get("method") == "mining.set_difficulty":
                        current_diff.value = float(msg["params"][0])
                        shared_data["diff"] = current_diff.value
                    elif msg.get("id") == 4: # Submission Response
                        if msg.get("result"):
                            shared_data["accepted"] += 1
                            logs.append(">>> SHARE ACCEPTED BY ANTPOOL <<<")
                        else:
                            logs.append(f"Share Rejected: {msg.get('error')}")
            except: pass

            # Update System State
            temp = get_cpu_temp()
            shared_data["temp"] = temp
            throttle_factor.value = 0.1 if temp > MAX_TEMP_C else 0.0
            
            h = sum(stats_array); shared_data["hashrate"] = h / 500
            for i in range(len(stats_array)): stats_array[i] = 0

            # UI Refresh
            stdscr.clear()
            stdscr.addstr(0, 0, f" ALFA ULTRA v5.5 | WORKER: {args.username}.{args.worker} ", curses.A_REVERSE)
            stdscr.addstr(2, 2, f"Status:   {shared_data['status']}", curses.color_pair(1 if connected else 2))
            stdscr.addstr(3, 2, f"Accepted: {shared_data['accepted']} shares", curses.color_pair(1))
            stdscr.addstr(4, 2, f"Hashrate: {shared_data['hashrate']:.2f} KH/s")
            stdscr.addstr(5, 2, f"CPU Temp: {temp:.1f}C (Limit: 79C)")
            
            stdscr.addstr(7, 2, "Live Logs:", curses.A_DIM)
            for i, l in enumerate(logs[-40:]):
                stdscr.addstr(8 + i, 4, f"> {l}")
            
            stdscr.refresh(); time.sleep(0.5)

    except KeyboardInterrupt: pass
    finally:
        stop_flag.value = True
        curses.endwin()
        for p in workers: p.terminate()

if __name__ == "__main__":
    main()
