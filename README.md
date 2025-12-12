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

# -------------------------------------------------
# Simple logger (safe to call before anything else)
# -------------------------------------------------
def logg(msg: str):
    print(msg)  # console output
    try:
        logging.basicConfig(level=logging.INFO,
                            filename="miner.log",
                            format='%(asctime)s %(message)s')
        logging.info(msg)
    except:
        pass  # in case logging fails very early

logg("[*] Miner starting...")

# -------------------------------------------------
# Optional GPU support – will NOT crash if PyCUDA missing
# -------------------------------------------------
gpu_enabled = False
try:
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    gpu_enabled = True
    logg("[*] PyCUDA loaded – GPU mining will use GPU when possible")
except Exception as e:
    logg(f"[!] PyCUDA not available ({e}) – running CPU-only mode")

# -------------------------------------------------
# Inlined context.py – all globals used by the miner
# -------------------------------------------------
fShutdown = False
listfThreadRunning = []
nHeightDiff = {}
updatedPrevHash = None
job_id = prevhash = coinb1 = coinb2 = None
merkle_branch = version = nbits = ntime = None
extranonce1 = extranonce2 = extranonce2_size = None
sock = None
target = None          # will be set by pool or block target
mode = "pool"          # default
host = port = user = password = None

# Stats (shared between threads)
hashrates = []
accepted = rejected = 0
accepted_timestamps = []
rejected_timestamps = []
lock = threading.Lock()

# -------------------------------------------------
# Config
# -------------------------------------------------
SOLO_HOST = 'solo.ckpool.org'
SOLO_PORT = 3333
SOLO_ADDRESS = 'bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.001'

POOL_HOST = 'ss.antpool.com'
POOL_PORT = 3333
POOL_WORKER = 'Xk2000.001'
POOL_PASSWORD = 'x'

num_threads = max(1, os.cpu_count() * 2)   # hyper-threaded cores

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def diff_to_target(diff: float) -> str:
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    target_int = diff1 // int(diff)
    return format(target_int, '064x')

def signal_handler(sig, frame):
    global fShutdown
    fShutdown = True
    logg("\n[!] Ctrl+C received – shutting down...")

# -------------------------------------------------
# CPU + optional GPU mining loop
# -------------------------------------------------
def bitcoin_miner(thread_id: int):
    global nbits, version, prevhash, ntime, target

    # Wait for first job
    while None in (nbits, version, prevhash, ntime):
        time.sleep(0.5)

    # Build static part of header (without nonce)
    header_static = version + prevhash + merkle_branch_root() + ntime + nbits
    header_bytes = binascii.unhexlify(header_static)

    # Block target when solo mining
    block_target = nbits[2:] + '00' * (int(nbits[:2], 16) - 3)
    block_target = block_target.zfill(64)

    current_target = target or block_target

    nonce = 0xffffffff                     # start from max (your "backwards" style)
    hashes_done = 0
    last_report = time.time()

    while not fShutdown:
        # --- GPU part (if available) ---
        if gpu_enabled and hasattr(cuda_mod, 'get_function'):
            try:
                batch = 10_000_000
                grid = (batch + 1023) // 1024
                func = cuda_mod.get_function("mine_kernel")
                found = np.array([0xffffffff], dtype=np.uint32)
                found_gpu = cuda.to_device(found)

                func(cuda.In(header_bytes),
                     np.uint32(len(header_bytes)),
                     np.uint32(nonce - batch),
                     np.uint32(batch),
                     cuda.In(binascii.unhexlify(current_target)),
                     found_gpu,
                     block=(1024,1,1), grid=(grid,1))

                cuda.memcpy_dtoh(found, found_gpu)
                if found[0] != 0xffffffff:
                    nonce = int(found[0])
                    submit_share(nonce)
                    return
                hashes_done += batch
                nonce -= batch
            except Exception as e:
                logg(f"[!] GPU error (thread {thread_id}): {e}")

        # --- CPU fallback / additional hashes ---
        for _ in range(50000):                     # big enough to keep CPU busy
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
                    hr = 1_000_000 / elapsed
                    with lock:
                        hashrates[thread_id] = int(hr)
                last_report = now

        time.sleep(0.001)  # tiny yield

def merkle_branch_root():
    # Re-calculate merkle root for current extranonce2 (simplified – works for most pools)
    global extranonce1, extranonce2, coinb1, coinb2, merkle_branch
    coinbase = coinb1 + extranonce1 + extranonce2 + coinb2
    coinbase_hash = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()
    merkle = coinbase_hash
    for branch in merkle_branch:
        merkle = hashlib.sha256(hashlib.sha256(merkle + binascii.unhexlify(branch)).digest()).digest()
    return binascii.hexlify(merkle).decode()[::-1]  # little-endian

def submit_share(nonce: int):
    global job_id, ntime, extranonce2, user
    payload = {
        "id": 1,
        "method": "mining.submit",
        "params": [user, job_id, extranonce2, ntime, f"{nonce:08x}"]
    }
    msg = json.dumps(payload) + "\n"
    try:
        sock.sendall(msg.encode())
        resp = sock.recv(1024).decode()
        logg(f"[*] Submitted nonce {nonce:08x} → {resp.strip()}")
        with lock:
            if "true" in resp.lower():
                accepted += 1
                accepted_timestamps.append(time.time())
            else:
                rejected += 1
                rejected_timestamps.append(time.time())
    except Exception as e:
        logg(f"[!] Submit failed: {e}")

# -------------------------------------------------
# Stratum listener (gets work from pool)
# -------------------------------------------------
def stratum_worker():
    global sock, job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce1, extranonce2_size

    s = socket.socket()
    s.connect((host, port))
    sock = s

    # subscribe
    s.sendall(b'{"id":1,"method":"mining.subscribe","params":[]}\n')
    s.recv(1024)

    # authorize
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
                     merkle_branch, version, nbits, ntime, clean_jobs) = msg["params"]
                    updatedPrevHash = prevhash
                    logg(f"[*] New job #{job_id} – height ~{get_current_block_height()}")
                elif msg.get("method") == "mining.set_difficulty":
                    target = diff_to_target(msg["params"][0])
                    logg(f"[*] Difficulty set to {msg['params'][0]}")
        except Exception as e:
            logg(f"[!] Stratum error: {e}")
            break

# -------------------------------------------------
# Display thread (curses)
# -------------------------------------------------
def display_worker():
    stdscr = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED,   curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW,curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN,  curses.COLOR_BLACK)
    curses.noecho(); curses.cbreak(); stdscr.keypad(True)

    try:
        while not fShutdown:
            stdscr.clear()
            now = time.time()
            with lock:
                a_min = sum(1 for t in accepted_timestamps if now-t<60)
                r_min = sum(1 for t in rejected_timestamps if now-t<60)

            stdscr.addstr(0, 0, f"Bitcoin {mode.upper()} Miner (CPU + {'GPU' if gpu_enabled else 'NO GPU'})", curses.color_pair(4)|curses.A_BOLD)
            stdscr.addstr(2, 0, f"Block height : ~{get_current_block_height()}", curses.color_pair(3))
            stdscr.addstr(3, 0, f"Total hashrate : {sum(hashrates):,} H/s", curses.color_pair(1))
            stdscr.addstr(4, 0, f"Threads      : {num_threads}", curses.color_pair(3))
            stdscr.addstr(5, 0, f"Shares       : {accepted} accepted  /  {rejected} rejected", curses.color_pair(1) if accepted else curses.color_pair(2))
            stdscr.addstr(6, 0, f"Per minute   : {a_min} acc / {r_min} rej", curses.color_pair(1))
            stdscr.addstr(8, 0, "Press q to quit", curses.color_pair(3))
            stdscr.refresh()
            time.sleep(1)
            if stdscr.getch() == ord('q'):
                global fShutdown
                fShutdown = True
    finally:
        curses.endwin()

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["solo","solo"], default="pool", help="solo or pool (default: pool)")
    args = parser.parse_args()

    mode = args.mode
    if mode == "solo":
        host, port, user, password = SOLO_HOST, SOLO_PORT, SOLO_ADDRESS, SOLO_PASSWORD
    else:
        host, port, user, password = POOL_HOST, POOL_PORT, POOL_WORKER, POOL_PASSWORD

    # initialise shared stats list
    hashrates = [0] * num_threads

    # start stratum listener
    threading.Thread(target=stratum_worker, daemon=True).start()

    # wait a moment for connection & first job
    time.sleep(3)

    # start mining threads
    for i in range(num_threads):
        threading.Thread(target=bitcoin_miner, args=(i,), daemon=True).start()

    # start display
    threading.Thread(target=display_worker, daemon=True).start()

    logg("[*] Miner running – press Ctrl+C or 'q' in display to stop")
    try:
        while not fShutdown:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    logg("[*] Shutdown complete")
