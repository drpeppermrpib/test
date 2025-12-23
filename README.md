#!/usr/bin/env python3

import argparse
import json
import socket
import time
import sys
import os
import curses
import subprocess
import binascii
import hashlib
from numba import cuda, uint32, void
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np

# ======================  CPU TEMPERATURE ======================
def get_cpu_temp():
    try:
        result = subprocess.check_output(["sensors"], text=True)
        temps = []
        for line in result.splitlines():
            if 'Tctl' in line or 'Tdie' in line or 'edge' in line or 'junction' in line:
                match = re.search(r'[\+\-]?[\d\.]+°C', line)
                if match:
                    temps.append(float(match.group(0).replace('°C', '').strip()))
        if temps:
            avg = sum(temps) / len(temps)
            max_temp = max(temps)
            return f"{avg:.1f}°C (avg) / {max_temp:.1f}°C (max)"
    except:
        pass
    return "N/A"

# ======================  GLOBALS ======================
fShutdown = False
hashrate = 0
accepted = rejected = 0
accepted_timestamps = []
rejected_timestamps = []
log_lines = []
max_log = 40

sock = None
job_id = prevhash = coinb1 = coinb2 = None
merkle_branch = version = nbits = ntime = None
extranonce1 = extranonce2 = extranonce2_size = None
target = None
host = port = user = password = None

# ======================  LOGGER ======================
def logg(msg):
    timestamp = int(time.time() * 100000)
    prefixed_msg = f"₿ ({timestamp}) {msg}"
    log_lines.append(prefixed_msg)

logg("GPU Miner starting...")

# ======================  CONFIG ======================
BRAIINS_HOST = 'stratum.braiins.com'
BRAIINS_PORT = 3333

# ======================  GPU KERNEL ======================
@cuda.jit
def sha256_kernel(header, nonce_start, nonces, results):
    idx = cuda.grid(1)
    if idx < nonces.shape[0]:
        nonce = nonce_start + uint32(idx)
        state = cuda.const.array_like(header)
        # Simple SHA256 (educational, not optimized for speed)
        # Full SHA256 on GPU is complex; this is a placeholder for real implementation
        # In practice, use optimized kernels like from ccminer or custom CUDA
        h = uint32(0)
        for i in range(len(header)):
            h = h ^ header[i]
        h = h ^ nonce
        results[idx] = h

# ======================  SUBMIT SHARE ======================
def submit_share(nonce):
    payload = {
        "id": 1,
        "method": "mining.submit",
        "params": [user, job_id, extranonce2, ntime, f"{nonce:08x}"]
    }
    try:
        logg(f"stratum_api: tx: {json.dumps(payload)}")
        sock.sendall((json.dumps(payload) + "\n").encode())
        resp = sock.recv(1024).decode().strip()
        logg(f"stratum_task: rx: {resp}")
        logg("stratum_task: message result accepted" if "true" in resp.lower() else "[!] Share rejected")

        if "true" in resp.lower():
            global accepted
            accepted += 1
            accepted_timestamps.append(time.time())
            log_lines.append("*** SHARE ACCEPTED ***")
            curses.beep()
        else:
            global rejected
            rejected += 1
            rejected_timestamps.append(time.time())
    except Exception as e:
        logg(f"[!] Submit failed: {e}")

# ======================  GPU MINING LOOP ======================
def gpu_miner():
    global job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, target, extranonce2

    last_job_id = None
    hashes_done = 0
    last_report = time.time()

    while not fShutdown:
        if job_id != last_job_id:
            last_job_id = job_id
            if None in (nbits, version, prevhash, ntime):
                time.sleep(0.5)
                continue

            logg(f"New Work Dequeued {job_id}")

            # Prepare header (simplified)
            header_static = version + prevhash + coinb1 + extranonce1 + extranonce2 + coinb2 + ntime + nbits
            header_bytes = binascii.unhexlify(header_static)

            share_target = target if target else "00000000ffff0000000000000000000000000000000000000000000000000000"

            nonce_start = random.randint(0, 0xffffffff)

        # GPU grid/block config (adjust for your GPU)
        threads_per_block = 256
        blocks = 1024
        total_threads = threads_per_block * blocks

        # Allocate GPU memory
        d_header = cuda.to_device(np.frombuffer(header_bytes, dtype=np.uint8))
        d_nonces = cuda.device_array(total_threads, dtype=np.uint32)
        d_results = cuda.device_array(total_threads, dtype=np.uint32)

        # Launch kernel (placeholder - real SHA256 kernel needed for high speed)
        sha256_kernel[blocks, threads_per_block](d_header, nonce_start, d_nonces, d_results)

        # Copy results back
        results = d_results.copy_to_host()

        hashes_done += total_threads

        # Check for share (simplified - real check needed)
        for i in range(total_threads):
            if results[i] < int(share_target, 16):
                nonce = nonce_start + i
                submit_share(nonce)

        # Update hashrate
        now = time.time()
        elapsed = now - last_report
        if elapsed > 0:
            global hashrate
            hashrate = int(total_threads / elapsed)
        last_report = now

# ======================  STRATUM ======================
def stratum_worker():
    global sock, job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce1, extranonce2_size

    while not fShutdown:
        try:
            s = socket.socket()
            s.connect((host, port))
            sock = s

            s.sendall(b'{"id":1,"method":"mining.subscribe","params":[]}\n')

            auth = {"id":2,"method":"mining.authorize","params":[user,password]}
            s.sendall((json.dumps(auth)+"\n").encode())

            buf = b""
            while not fShutdown:
                data = s.recv(4096)
                if not data:
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
                        logg(f"New Work Dequeued {job_id}")
                    elif msg.get("method") == "mining.set_difficulty":
                        target = diff_to_target(msg["params"][0])
                        logg(f"Difficulty set to {msg['params'][0]}")
        except Exception as e:
            logg(f"[!] Stratum error: {e} – reconnecting in 10s...")
            time.sleep(10)

# ======================  DISPLAY ======================
def display_worker():
    global log_lines, hashrate
    stdscr = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.noecho(); curses.cbreak(); stdscr.keypad(True)

    try:
        while not fShutdown:
            stdscr.clear()
            screen_height, screen_width = stdscr.getmaxyx()

            now = time.time()
            a_min = sum(1 for t in accepted_timestamps if now-t<60)
            r_min = sum(1 for t in rejected_timestamps if now-t<60)

            cpu_temp = get_cpu_temp()

            stdscr.addstr(0, max(0, screen_width - 20), "Ctrl+C to quit", curses.color_pair(3))
            title = "Bitcoin Miner (GPU) - Braiins Pool"
            stdscr.addstr(2, 0, title, curses.color_pair(4)|curses.A_BOLD)

            stdscr.addstr(4, 0, f"Hashrate     : {hashrate:,} H/s", curses.color_pair(1))
            stdscr.addstr(5, 0, f"GPU Temp     : {cpu_temp}", curses.color_pair(3))
            stdscr.addstr(6, 0, f"Shares       : {accepted} accepted / {rejected} rejected")
            stdscr.addstr(7, 0, f"Last minute  : {a_min} acc / {r_min} rej")

            stdscr.addstr(9, 0, "─" * (screen_width - 1), curses.color_pair(3))

            start_y = 10
            for i, line in enumerate(log_lines[-max_log:]):
                if start_y + i >= screen_height:
                    break
                stdscr.addstr(start_y + i, 0, line[:screen_width-1], curses.color_pair(5))

            stdscr.refresh()
            time.sleep(1)
    finally:
        curses.endwin()

# ======================  MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Braiins Pool SHA-256 GPU Miner")
    parser.add_argument("--username", type=str, required=True, help="Braiins username or payout address")
    parser.add_argument("--worker", type=str, default="gpu002", help="Worker name")
    args = parser.parse_args()

    host = BRAIINS_HOST
    port = BRAIINS_PORT
    user = f"{args.username}.{args.worker}"
    password = "x"

    # Start stratum
    threading.Thread(target=stratum_worker, daemon=True).start()
    time.sleep(5)

    # Start GPU mining
    threading.Thread(target=gpu_miner, daemon=True).start()

    # Display
    threading.Thread(target=display_worker, daemon=True).start()

    logg("[*] Miner running – press Ctrl+C to stop")
    try:
        while not fShutdown:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    fShutdown = True
    time.sleep(2)
    logg("[*] Shutdown complete")
