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

# ======================  GLOBALS  ======================
fShutdown = False
listfThreadRunning = []
nHeightDiff = {}
updatedPrevHash = None
job_id = prevhash = coinb1 = coinb2 = None
merkle_branch = version = nbits = ntime = None
extranonce1 = extranonce2 = extranonce2_size = None
sock = None
target = None
mode = "pool"
host = port = user = password = None

# Stats
hashrates = []
accepted = rejected = 0
accepted_timestamps = []
rejected_timestamps = []
lock = threading.Lock()

# ======================  LOGGER  ======================
def logg(msg: str):
    print(msg)
    try:
        logging.basicConfig(level=logging.INFO,
                            filename="miner.log",
                            format='%(asctime)s %(message)s',
                            force=True)
        logging.info(msg)
    except:
        pass

logg("[*] Miner starting...")

# ======================  OPTIONAL GPU  ======================
gpu_enabled = False
try:
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    gpu_enabled = True
    logg("[*] PyCUDA loaded – GPU support active")
except Exception as e:
    logg(f"[!] PyCUDA not available ({e}) – CPU-only mode")

# ======================  CONFIG  ======================
SOLO_HOST = 'solo.ckpool.org'
SOLO_PORT = 3333
SOLO_ADDRESS = 'bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.001'

POOL_HOST = 'ss.antpool.com'
POOL_PORT = 3333
POOL_WORKER = 'Xk2000.001'
POOL_PASSWORD = 'x'

num_threads = max(1, os.cpu_count() * 2)

# ======================  HELPERS  ======================
def diff_to_target(diff: float) -> str:
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    target_int = diff1 // int(diff)
    return format(target_int, '064x')

def signal_handler(sig, frame):
    global fShutdown
    fShutdown = True
    logg("\n[!] Shutting down...")

signal.signal(signal.SIGINT, signal_handler)

# ======================  MINING LOOP  ======================
def bitcoin_miner(thread_id: int):
    global nbits, version, prevhash, ntime, target

    # Wait for first job
    while None in (nbits, version, prevhash, ntime):
        time.sleep(0.5)

    # Build static header part (nonce will be added later)
    merkle_root = calculate_merkle_root()
    header_static = version + prevhash + merkle_root + ntime + nbits
    header_bytes = binascii.unhexlify(header_static)

    current_target = target or (nbits[2:] + '00' * (int(nbits[:2], 16) - 3)).zfill(64)

    nonce = 0xffffffff
    hashes_done = 0
    last_report = time.time()

    while not fShutdown:
        # GPU batch (if available)
        if gpu_enabled:
            try:
                batch = 10_000_000
                grid = (batch + 1023) // 1024
                found = np.array([0xffffffff], dtype=np.uint32)
                found_gpu = cuda.to_device(found)

                cuda_mod.get_function("mine_kernel")(
                    cuda.In(header_bytes),
                     np.uint32(len(header_bytes)),
                     np.uint32(nonce - batch),
                     np.uint32(batch),
                     cuda.In(binascii.unhexlify(current_target)),
                     found_gpu,
                     block=(1024,1,1), grid=(grid,1))

                cuda.memcpy_dtoh(found, found_gpu)
                if found[0] != 0xffffffff:
                    submit_share(int(found[0]))
                    return
                hashes_done += batch
                nonce -= batch
            except Exception as e:
                logg(f"[!] GPU error (thread {thread_id}): {e}")

        # CPU batch
        for _ in range(50000):
            nonce = (nonce - 1) & 0xffffffff
            h = hashlib.sha256(hashlib.sha256(header_bytes + nonce.to_bytes(4, 'little')).digest()).digest()
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

def calculate_merkle_root():
    coinbase = coinb1 + extranonce1 + extranonce2 + coinb2
    h = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()
    for b in merkle_branch:
        h = hashlib.sha256(hashlib.sha256(h + binascii.unhexlify(b)).digest()).digest()
    return binascii.hexlify(h).decode()[::-1]  # little-endian

def submit_share(nonce: int):
    payload = {
        "id": 1,
        "method": "mining.submit",
        "params": [user, job_id, extranonce2, ntime, f"{nonce:08x}"]
    }
    msg = json.dumps(payload) + "\n"
    try:
        sock.sendall(msg.encode())
        resp = sock.recv(1024).decode().strip()
        logg(f"[*] Submitted nonce {nonce:08x} → {resp}")
        with lock:
            if "true" in resp.lower():
                global accepted, accepted_timestamps
                accepted += 1
                accepted_timestamps.append(time.time())
            else:
                global rejected, rejected_timestamps
                rejected += 1
                rejected_timestamps.append(time.time())
    except Exception as e:
        logg(f"[!] Submit failed: {e}")

# ======================  STRATUM LISTENER  ======================
def stratum_worker():
    global sock, job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce1, extranonce2_size

    s = socket.socket()
    s.connect((host, port))
    sock = s

    s.sendall(b'{"id":1,"method":"mining.subscribe","params":[]}\n')
    s.recv(1024)

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
                    logg(f"[*] New job #{job_id}")
                elif msg.get("method") == "mining.set_difficulty":
                    target = diff_to_target(msg["params"][0])
        except Exception as e:
            logg(f"[!] Stratum error: {e}")
            break

# ======================  DISPLAY  ======================
def display_worker():
    stdscr = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED,   , curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN  , curses.COLOR_BLACK)
    curses.noecho(); curses.cbreak(); stdscr.keypad(True)

    try:
        while not fShutdown:
            stdscr.clear()
            now = time.time()
            with lock:
                a_min = sum(1 for t in accepted_timestamps if now-t < 60)
                r_min = sum(1 for t in rejected_timestamps if now-t < 60)

            stdscr.addstr(0, 0, f"Bitcoin {mode.upper()} Miner (CPU + {'GPU' if gpu_enabled else 'NO GPU'})", curses.color_pair(4)|curses.A_BOLD)
            stdscr.addstr(2, 0, f"Block height : ~{requests.get('https://blockchain.info/q/getblockcount', timeout=5).text}", curses.color_pair(3))
            stdscr.addstr(3, 0, f"Hashrate     : {sum(hashrates):,} H/s", curses.color_pair(1))
            stdscr.addstr(4, 0, f"Threads      : {num_threads}", curses.color_pair(3))
            stdscr.addstr(5, 0, f"Shares       : {accepted} accepted / {rejected} rejected", curses.color_pair(1))
            stdscr.addstr(6, 0, f"Per minute   : {a_min} acc / {r_min} rej", curses.color_pair(1))
            stdscr.addstr(8, 0, "Press q to quit", curses.color_pair(3))
            stdscr.refresh()
            time.sleep(1)
            if stdscr.getch() == ord('q'):
                global fShutdown
                fShutdown = True
    finally:
        curses.endwin()

# ======================  MAIN  ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["solo", "pool"], default="pool", help="solo or pool")
    args = parser.parse_args()

    mode = args.mode
    if mode == "solo":
        host, port, user, password = SOLO_HOST, SOLO_PORT, SOLO_ADDRESS, "x"
    else:
        host, port, user, password = POOL_HOST, POOL_PORT, POOL_WORKER, POOL_PASSWORD

    hashrates = [0] * num_threads

    threading.Thread(target=stratum_worker, daemon=True).start()
    time.sleep(3)

    for i in range(num_threads):
        threading.Thread(target=bitcoin_miner, args=(i,), daemon=True).start()

    threading.Thread(target=display_worker, daemon=True).start()

    logg("[*] Miner running – press Ctrl+C or 'q' to stop")
    try:
        while not fShutdown:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    logg("[*] Shutdown complete")
