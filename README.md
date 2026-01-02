#!/usr/bin/env python3
import socket
import ssl
import json
import time
import threading
import multiprocessing as mp
import curses
import binascii
import struct
import hashlib
import os
import sys
import queue
from datetime import datetime

# ================= CONFIGURATION =================
# SOLO MINING TARGET
SOLO_URL = "solo.stratum.braiins.com"
SOLO_PORT = 443 # SSL Port
SOLO_USER = "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e"
SOLO_PASS = "x"

# API
API_PORT = 60060

# ================= GPU SUPPORT CHECK =================
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import numpy as np
    HAS_CUDA = True
except:
    HAS_CUDA = False

# ================= SHARED DATA & QUEUES =================
# These allow threads/processes to talk without freezing each other
class SharedData:
    def __init__(self, manager):
        self.job_queue = manager.Queue()
        self.result_queue = manager.Queue()
        self.log_queue = manager.Queue()
        self.stats = manager.dict()
        self.stats['accepted'] = 0
        self.stats['rejected'] = 0
        self.stats['hashrate'] = 0.0
        self.stats['difficulty'] = 1000.0
        self.stats['connected'] = False
        self.stats['temp_cpu'] = 0.0
        self.stats['temp_gpu'] = 0.0
        self.stop_event = manager.Event()

# ================= NETWORK CLIENT (THREADED) =================
def network_thread(shared):
    """Handles connection separately so UI never freezes"""
    while not shared.stop_event.is_set():
        sock = None
        try:
            shared.log_queue.put(("WARN", f"Connecting to {SOLO_URL}:{SOLO_PORT} (SSL)..."))
            
            # 1. Basic TCP Connection
            raw_sock = socket.create_connection((SOLO_URL, SOLO_PORT), timeout=10)
            
            # 2. SSL Wrap (Fix for SSL Errors)
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            sock = context.wrap_socket(raw_sock, server_hostname=SOLO_URL)
            
            # 3. Stratum Handshake
            # Subscribe
            sock.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["RLM/3.0"]}) + "\n").encode())
            
            # Authorize
            sock.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [SOLO_USER, SOLO_PASS]}) + "\n").encode())
            
            shared.stats['connected'] = True
            shared.log_queue.put(("GOOD", "Connected & Authorized!"))
            
            # 4. Data Loop
            sock.settimeout(0.2) # Short timeout to check for outgoing shares
            buffer = ""
            
            while not shared.stop_event.is_set():
                # A. Send Shares
                while not shared.result_queue.empty():
                    res = shared.result_queue.get()
                    payload = json.dumps({
                        "id": 4,
                        "method": "mining.submit",
                        "params": [SOLO_USER, res['job_id'], res['extranonce2'], res['ntime'], res['nonce']]
                    }) + "\n"
                    sock.sendall(payload.encode())
                    shared.log_queue.put(("INFO", f"Submitting Share: {res['nonce']}"))

                # B. Receive Data
                try:
                    data = sock.recv(4096).decode()
                    if not data:
                        raise Exception("Connection Closed by Pool")
                    
                    buffer += data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if not line: continue
                        
                        msg = json.loads(line)
                        
                        # New Job
                        if msg.get('method') == 'mining.notify':
                            p = msg['params']
                            # job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs
                            job_data = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
                            
                            if p[8]: # Clean jobs
                                while not shared.job_queue.empty(): shared.job_queue.get()
                            
                            # Flood queue for all workers
                            for _ in range(mp.cpu_count() + 2):
                                shared.job_queue.put(job_data)
                                
                            shared.log_queue.put(("INFO", f"New Block: {p[0][:8]}..."))

                        # Set Difficulty
                        elif msg.get('method') == 'mining.set_difficulty':
                            shared.stats['difficulty'] = msg['params'][0]
                            shared.log_queue.put(("WARN", f"Difficulty: {msg['params'][0]}"))

                        # Share Response
                        elif msg.get('id') == 4:
                            if msg.get('result') == True:
                                shared.stats['accepted'] += 1
                                shared.log_queue.put(("GOOD", ">>> SHARE ACCEPTED <<<"))
                            else:
                                shared.stats['rejected'] += 1
                                err = msg.get('error')
                                shared.log_queue.put(("BAD", f"Share Rejected: {err}"))

                except socket.timeout:
                    pass # Normal, just loop back to check result_queue

        except Exception as e:
            shared.stats['connected'] = False
            shared.log_queue.put(("BAD", f"Net Error: {e}"))
            time.sleep(5) # Wait before retry
        finally:
            if sock: 
                try: sock.close()
                except: pass

# ================= MINING WORKERS =================
def cpu_worker(id, shared):
    """Pure Number Crunching"""
    my_hashes = 0
    while not shared.stop_event.is_set():
        if shared.job_queue.empty():
            time.sleep(0.05)
            continue
            
        try:
            job = shared.job_queue.get()
            job_id, prevhash, coinb1, coinb2, merkle_branch, ver, nbits, ntime, clean = job
            
            # Prep Header
            extranonce2 = struct.pack('<I', id).hex().zfill(8)
            coinbase = binascii.unhexlify(coinb1 + extranonce2 + coinb2)
            coinbase_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            
            merkle_root = coinbase_hash
            for b in merkle_branch:
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + binascii.unhexlify(b)).digest()).digest()
                
            header_pre = (
                binascii.unhexlify(ver)[::-1] +
                binascii.unhexlify(prevhash)[::-1] +
                merkle_root +
                binascii.unhexlify(ntime)[::-1] +
                binascii.unhexlify(nbits)[::-1]
            )
            
            # Target
            diff = shared.stats['difficulty']
            target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000 // int(diff if diff > 0 else 1)
            
            # Mine Batch
            nonce = id * 1000000
            for n in range(nonce, nonce + 50000):
                header = header_pre + struct.pack('<I', n)
                block_hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                
                # Check
                if int.from_bytes(block_hash[::-1], 'big') <= target:
                    shared.result_queue.put({
                        "job_id": job_id,
                        "extranonce2": extranonce2,
                        "ntime": ntime,
                        "nonce": f"{n:08x}"
                    })
            
            # Simple hashrate tracking (rough)
            # In a real scenario we'd use a shared counter, but that locks too much.
            
        except: pass

def gpu_worker_stub(shared):
    """Placeholder for GPU worker to keep code simple"""
    if not HAS_CUDA: return
    # This process would load PyCUDA and crunch similar to CPU worker
    # but in parallel batches of millions.
    while not shared.stop_event.is_set():
        if shared.job_queue.empty():
            time.sleep(0.1)
            continue
        # Consume job to simulate work
        shared.job_queue.get()
        time.sleep(0.5)

# ================= UI & MAIN =================
def get_temps(shared):
    """Runs occasionally to update temp stats"""
    try:
        res = os.popen("sensors").read()
        for line in res.split("\n"):
            if "Tdie" in line or "Package id 0" in line:
                shared.stats['temp_cpu'] = float(line.split("+")[1].split("°")[0])
                break
    except: pass

def main(stdscr):
    # Setup Curses
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.curs_set(0)
    stdscr.nodelay(True) # NON-BLOCKING INPUT

    # Setup Multiprocessing
    manager = mp.Manager()
    shared = SharedData(manager)

    # Start Net Thread
    net_t = threading.Thread(target=network_thread, args=(shared,))
    net_t.daemon = True
    net_t.start()

    # Start Miners
    procs = []
    cpu_count = mp.cpu_count()
    for i in range(cpu_count):
        p = mp.Process(target=cpu_worker, args=(i, shared))
        p.start()
        procs.append(p)

    if HAS_CUDA:
        p_gpu = mp.Process(target=gpu_worker_stub, args=(shared,))
        p_gpu.start()
        procs.append(p_gpu)

    # Local Logs Buffer
    logs = []
    start_time = time.time()
    last_temp_check = 0

    try:
        while True:
            # 1. Process Logs from Queue
            while not shared.log_queue.empty():
                try:
                    lvl, msg = shared.log_queue.get_nowait()
                    ts = datetime.now().strftime("%H:%M:%S")
                    logs.append((ts, lvl, msg))
                    if len(logs) > 50: logs.pop(0)
                except: break

            # 2. Update Temps (every 5s)
            if time.time() - last_temp_check > 5:
                get_temps(shared)
                last_temp_check = time.time()

            # 3. Draw UI
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            
            # Top Bar
            stdscr.addstr(0, 0, f" RLM MINER v3.0 | SOLO TARGETING ".center(w), curses.A_REVERSE | curses.color_pair(4))
            
            # Status Grid
            status = "CONNECTED" if shared.stats['connected'] else "CONNECTING..."
            s_col = curses.color_pair(1) if shared.stats['connected'] else curses.color_pair(2)
            
            stdscr.addstr(2, 2, f"STATUS:      {status}", s_col)
            stdscr.addstr(3, 2, f"POOL:        {SOLO_URL}", curses.color_pair(4))
            stdscr.addstr(4, 2, f"USER:        {SOLO_USER[:15]}...", curses.color_pair(4))
            
            # Stats Grid
            stdscr.addstr(2, 40, f"CPU TEMP:    {shared.stats['temp_cpu']}°C", curses.color_pair(1))
            stdscr.addstr(3, 40, f"DIFFICULTY:  {shared.stats['difficulty']}", curses.color_pair(2))
            stdscr.addstr(4, 40, f"ACTIVE CPUS: {cpu_count}", curses.color_pair(4))

            # Shares
            acc = shared.stats['accepted']
            rej = shared.stats['rejected']
            stdscr.addstr(6, 2, f"ACCEPTED: {acc}", curses.color_pair(1))
            stdscr.addstr(6, 20, f"REJECTED: {rej}", curses.color_pair(3))

            # Log Window
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(8, 2, " LIVE LOGS ", curses.A_REVERSE)
            
            for i, (ts, lvl, msg) in enumerate(logs[-(h-10):]):
                c = curses.color_pair(4)
                if lvl == "GOOD": c = curses.color_pair(1)
                elif lvl == "WARN": c = curses.color_pair(2)
                elif lvl == "BAD": c = curses.color_pair(3)
                
                line_str = f"[{ts}] {msg}"
                stdscr.addstr(10+i, 1, line_str[:w-2], c)

            stdscr.refresh()
            
            # Check Input
            k = stdscr.getch()
            if k == ord('q'): break
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        shared.stop_event.set()
        for p in procs: p.terminate()

if __name__ == "__main__":
    curses.wrapper(main)
