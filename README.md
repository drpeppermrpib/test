#!/usr/bin/env python3

import threading
import requests
import binascii
import hashlib
import logging
import random
import socket
import time
import json
import sys
import os
import curses
import argparse
import signal

# ======================  diff_to_target ======================
def diff_to_target(diff):
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    target_int = diff1 // int(diff)
    return format(target_int, '064x')

# ======================  GLOBALS ======================
fShutdown = False
hashrates = []
accepted = rejected = 0
accepted_timestamps = []
rejected_timestamps = []
lock = threading.Lock()

# job data
job_id = prevhash = coinb1 = coinb2 = None
merkle_branch = version = nbits = ntime = None
extranonce1 = extranonce2 = extranonce2_size = None
sock = None
target = None
mode = "pool"
host = port = user = password = None

# ======================  LOGGER ======================
def logg(msg):
    print(msg)
    try:
        logging.basicConfig(level=logging.INFO, filename="miner.log",
                            format='%(asctime)s %(message)s', force=True)
        logging.info(msg)
    except:
        pass

logg("[*] Miner starting...")

# ======================  OPTIONAL GPU (safe) ======================
gpu_enabled = False
try:
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    gpu_enabled = True
    logg("[*] PyCUDA loaded – GPU support active")
except Exception as e:
    logg(f"[!] PyCUDA not found ({e}) – CPU-only mode")

# CUDA kernel for SHA256 (optimized from open-source examples)
if gpu_enabled:
    cuda_mod = SourceModule("""
    #include <stdint.h>

    __device__ uint32_t ROTR32(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }

    __device__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (~x & z);
    }

    __device__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    __device__ uint32_t Sigma0(uint32_t x) {
        return ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22);
    }

    __device__ uint32_t Sigma1(uint32_t x) {
        return ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25);
    }

    __device__ uint32_t sigma0(uint32_t x) {
        return ROTR32(x, 7) ^ ROTR32(x, 18) ^ (x >> 3);
    }

    __device__ uint32_t sigma1(uint32_t x) {
        return ROTR32(x, 17) ^ ROTR32(x, 19) ^ (x >> 10);
    }

    __constant__ uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e0bf33, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    __device__ void sha256_init(uint32_t *state) {
        state[0] = 0x6a09e667;
        state[1] = 0xbb67ae85;
        state[2] = 0x3c6ef372;
        state[3] = 0xa54ff53a;
        state[4] = 0x510e527f;
        state[5] = 0x9b05688c;
        state[6] = 0x1f83d9ab;
        state[7] = 0x5be0cd19;
    }

    __device__ void sha256_transform(uint32_t *state, const uint8_t *block) {
        uint32_t W[64];
        uint32_t a, b, c, d, e, f, g, h;
        int i;

        for (i = 0; i < 16; i++) {
            W[i] = (block[i * 4] << 24) | (block[i * 4 + 1] << 16) | (block[i * 4 + 2] << 8) | (block[i * 4 + 3]);
        }

        for (i = 16; i < 64; i++) {
            W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16];
        }

        a = state[0];
        b = state[1];
        c = state[2];
        d = state[3];
        e = state[4];
        f = state[5];
        g = state[6];
        h = state[7];

        for (i = 0; i < 64; i++) {
            uint32_t t1 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i];
            uint32_t t2 = Sigma0(a) + Maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    __global__ void mine_kernel(uint8_t *header_base, uint32_t header_len, uint32_t start_nonce, uint32_t num_nonces, uint8_t *target, uint32_t *found_nonce) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_nonces) return;

        uint32_t nonce = start_nonce + idx;
        uint8_t header[80];
        memcpy(header, header_base, header_len);
        // Insert nonce into header (bytes 76-79, little-endian)
        header[76] = (nonce >> 0) & 0xFF;
        header[77] = (nonce >> 8) & 0xFF;
        header[78] = (nonce >> 16) & 0xFF;
        header[79] = (nonce >> 24) & 0xFF;

        uint32_t state[8];
        sha256_init(state);
        sha256_transform(state, header);
        sha256_init(state2);
        sha256_transform(state2, (uint8_t*)state);  # double hash

        // Check if hash < target (big-endian compare)
        bool valid = true;
        for (int i = 0; i < 32; i++) {
            if (((uint8_t*)state2)[31 - i] > target[i]) { valid = false; break; }
            if (((uint8_t*)state2)[31 - i] < target[i]) break;
        }
        if (valid) {
            atomicMin(found_nonce, nonce);
        }
    }
    """)

# ======================  CONFIG ======================
SOLO_HOST = 'solo.ckpool.org'
SOLO_PORT = 3333
SOLO_ADDRESS = 'bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.001'

POOL_HOST = 'ss.antpool.com'
POOL_PORT = 3333
POOL_WORKER = 'Xk2000.001'
POOL_PASSWORD = 'x'

# Start with low threads and slowly increase to reach high load
initial_threads = 4
max_threads = max(1, os.cpu_count() * 2)  # e.g. 48 on your rig
num_threads = initial_threads

# ======================  SIGNAL ======================
def signal_handler(sig, frame):
    global fShutdown
    fShutdown = True
    logg("\n[!] Shutting down...")

signal.signal(signal.SIGINT, signal_handler)

# ======================  MERKLE ROOT ======================
def calculate_merkle_root():
    if None in (coinb1, extranonce1, extranonce2, coinb2, merkle_branch):
        return "0" * 64
    coinbase = coinb1 + extranonce1 + extranonce2 + coinb2
    h = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()
    for b in merkle_branch:
        h = hashlib.sha256(hashlib.sha256(h + binascii.unhexlify(b)).digest()).digest()
    return binascii.hexlify(h).decode()[::-1]

# ======================  SUBMIT SHARE ======================
def submit_share(nonce):
    payload = {
        "id": 1,
        "method": "mining.submit",
        "params": [user, job_id, extranonce2, ntime, f"{nonce:08x}"]
    }
    try:
        sock.sendall((json.dumps(payload) + "\n").encode())
        resp = sock.recv(1024).decode().strip()
        logg(f"[+] Share submitted → {resp}")

        with lock:
            if "true" in resp.lower():
                global accepted, accepted_timestamps
                accepted += 1
                accepted_timestamps.append(time.time())
                print("\n" + "="*60)
                print("*** SHARE ACCEPTED ***")
                print(f"Nonce: {nonce:08x}")
                print(f"Time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*60 + "\n")
                print("\a", end="", flush=True)  # beep
            else:
                global rejected, rejected_timestamps
                rejected += 1
                rejected_timestamps.append(time.time())
                logg("[!] Share rejected")
    except Exception as e:
        logg(f"[!] Submit failed: {e}")

# ======================  BENCHMARK ======================
def run_benchmark(seconds=10):
    logg(f"[*] Running {seconds}-second benchmark...")
    start_time = time.time()
    hashes = 0
    nonce = 0
    dummy_header = b"dummy" * 20  # dummy data for benchmarking

    while time.time() - start_time < seconds:
        for _ in range(10000):
            h = hashlib.sha256(hashlib.sha256(dummy_header + nonce.to_bytes(4,'little')).digest()).digest()
            hashes += 1
            nonce += 1

    elapsed = time.time() - start_time
    hr = int(hashes / elapsed) if elapsed > 0 else 0
    logg(f"[*] Benchmark complete: {hr:,} H/s ({hashes:,} hashes in {elapsed:.2f}s)")
    return hr

# ======================  MINING LOOP ======================
def bitcoin_miner(thread_id):
    global nbits, version, prevhash, ntime, target

    while None in (nbits, version, prevhash, ntime):
        time.sleep(0.5)

    header_static = version + prevhash + calculate_merkle_root() + ntime + nbits
    header_bytes = binascii.unhexlify(header_static)

    current_target = target or (nbits[2:] + '00' * (int(nbits[:2],16) - 3)).zfill(64)

    nonce = 0xffffffff
    hashes_done = 0
    last_report = time.time()

    while not fShutdown:
        for _ in range(100000):
            nonce = (nonce - 1) & 0xffffffff
            h = hashlib.sha256(hashlib.sha256(header_bytes + nonce.to_bytes(4,'little')).digest()).digest()
            h_hex = binascii.hexlify(h[::-1]).decode()

            hashes_done += 1
            if h_hex < current_target:
                submit_share(nonce)
                return

            if hashes_done % 1_000_000 == 0:
                now = time.time()
                elapsed = now - last_report
                if elapsed > 0:
                    hr = int(1_000_000 / elapsed)
                    with lock:
                        hashrates[thread_id] = hr
                last_report = now

        time.sleep(0.001)

# ======================  STRATUM ======================
def stratum_worker():
    global sock, job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce1, extranonce2_size

    s = socket.socket()
    s.connect((host, port))
    sock = s

    s.sendall(b'{"id":1,"method":"mining.subscribe","params":[]}\n')
    s.recv(4096)

    auth = {"id":2,"method":"mining.authorize","params":[user,password]}
    s.sendall((json.dumps(auth)+"\n").encode())

    buf = b""
    while not fShutdown:
        try:
            data = s.recv(4096)
            if not data: break
            buf += data
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                if not line.strip(): continue
                msg = json.loads(line)
                if msg.get("method") == "mining.notify":
                    (job_id, prevhash, coinb1, coinb2,
                     merkle_branch, version, nbits, ntime, _) = msg["params"]
                    logg(f"[*] New job #{job_id} | Prevhash: {prevhash} | Target: {target}")
                elif msg.get("method") == "mining.set_difficulty":
                    target = diff_to_target(msg["params"][0])
                    logg(f"[*] Difficulty set to {msg['params'][0]}")
        except Exception as e:
            logg(f"[!] Stratum error: {e}")
            break

# ======================  DYNAMIC THREAD SCALING ======================
def thread_scaler():
    global num_threads
    while not fShutdown:
        time.sleep(30)  # check every 30 seconds
        if fShutdown: break

        load1, _, _ = os.getloadavg()
        current_load = load1

        if current_load < 46:  # slowly increase until ~46
            num_threads = min(max_threads, num_threads + 4)
            logg(f"[*] Load {current_load:.2f} – increasing to {num_threads} threads")
            # Add new threads if needed
            for i in range(num_threads - len(hashrates)):
                threading.Thread(target=bitcoin_miner, args=(len(hashrates),), daemon=True).start()
                hashrates.append(0)
        elif current_load > 48:
            num_threads = max(initial_threads, num_threads - 4)
            logg(f"[*] Load {current_load:.2f} – decreasing to {num_threads} threads")

# ======================  DISPLAY ======================
def display_worker():
    stdscr = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN,  curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED,    curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN,   curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # pink for errors
    curses.noecho(); curses.cbreak(); stdscr.keypad(True)

    error_lines = []
    max_errors = 10

    try:
        while not fShutdown:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            now = time.time()
            with lock:
                a_min = sum(1 for t in accepted_timestamps if now-t<60)
                r_min = sum(1 for t in rejected_timestamps if now-t<60)

            # Top right: Ctrl+C to quit
            stdscr.addstr(0, max(0, width - 20), "Ctrl+C to quit", curses.color_pair(3))

            # Title
            title = f"Bitcoin {mode.upper()} Miner (CPU)"
            stdscr.addstr(2, 0, title, curses.color_pair(4)|curses.A_BOLD)

            # Static stats
            try:
                height = requests.get('https://blockchain.info/q/getblockcount',timeout=3).text
            except:
                height = "???"
            stdscr.addstr(4, 0, f"Block height : ~{height}", curses.color_pair(3))
            stdscr.addstr(5, 0, f"Hashrate     : {sum(hashrates):,} H/s", curses.color_pair(1))
            stdscr.addstr(6, 0, f"Threads      : {num_threads}", curses.color_pair(3))
            stdscr.addstr(7, 0, f"Shares       : {accepted} accepted / {rejected} rejected")
            stdscr.addstr(8, 0, f"Last minute  : {a_min} acc / {r_min} rej")

            # Horizontal line
            stdscr.addstr(10, 0, "─" * (width - 1), curses.color_pair(3))

            # Dynamic log / errors on the right (pink)
            start_y = 11
            for i, line in enumerate(error_lines[-max_errors:]):
                if start_y + i >= height:
                    break
                stdscr.addstr(start_y + i, 0, line[:width-1], curses.color_pair(5))

            stdscr.refresh()
            time.sleep(1)
    finally:
        curses.endwin()

# Override print to capture logs for display
original_print = print
def custom_print(*args, **kwargs):
    original_print(*args, **kwargs)
    msg = " ".join(str(a) for a in args)
    if "[*]" in msg or "error" in msg.lower():
        with lock:
            error_lines.append(msg)

print = custom_print

# ======================  MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["solo","pool"], default="pool")
    args = parser.parse_args()

    mode = args.mode
    if mode == "solo":
        host, port, user, password = SOLO_HOST, SOLO_PORT, SOLO_ADDRESS, "x"
    else:
        host, port, user, password = POOL_HOST, POOL_PORT, POOL_WORKER, POOL_PASSWORD

    hashrates = [0] * num_threads

    threading.Thread(target=stratum_worker, daemon=True).start()
    time.sleep(3)

    # Start initial miners
    for i in range(initial_threads):
        threading.Thread(target=bitcoin_miner, args=(i,), daemon=True).start()

    # Dynamic thread scaler
    threading.Thread(target=thread_scaler, daemon=True).start()

    # Display
    threading.Thread(target=display_worker, daemon=True).start()

    logg("[*] Miner running – press Ctrl+C to stop")
    try:
        while not fShutdown:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    logg("[*] Shutdown complete")
