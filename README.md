#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KXT MINER SUITE v45
===================
Base: Beta v20 (UI, Sensors, Proxy)
Engine: kxt.py v19 (Block Hashing & Submission)
Add-on: 60s Bench + 79C Limit
"""

import sys

# FIX: Large integer string conversion limit
try: sys.set_int_max_str_digits(0)
except: pass

import os
import time
import socket
import json
import threading
import multiprocessing as mp
import curses
import binascii
import struct
import hashlib
import random
import select
import subprocess
import signal
import resource
import queue
from datetime import datetime

# ==============================================================================
# SECTION 1: SYSTEM PREP (V20 BASE)
# ==============================================================================

EXIT_FLAG = mp.Event()

def signal_handler(signum, frame):
    """Silent exit handler to prevent UI corruption."""
    if not EXIT_FLAG.is_set():
        EXIT_FLAG.set()
        try:
            import curses
            curses.endwin()
        except: pass
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def boot_checks():
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))
    except: pass

boot_checks()

# ==============================================================================
# SECTION 2: CONFIGURATION
# ==============================================================================

CONFIG = {
    "POOL": "solo.stratum.braiins.com",
    "PORT": 3333,
    "USER": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.rig1",
    "PASS": "x",
    "PROXY_PORT": 60060,
    "BENCH_TIME": 60,
    "THROTTLE_TEMP": 79.0,
    "RESUME_TEMP": 75.0,
}

# ==============================================================================
# SECTION 3: CUDA (HEAVY LOAD)
# ==============================================================================

CUDA_SRC = """
extern "C" {
    #include <stdint.h>
    __global__ void kxt_burn(uint32_t *output, uint32_t seed) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        volatile uint32_t a = 0x6a09e667 + idx + seed;
        volatile uint32_t b = 0xbb67ae85;
        #pragma unroll 128
        for(int i=0; i < 25000; i++) {
            a = (a << 5) | (a >> 27);
            b ^= a;
            a += b + 0xDEADBEEF;
            if (i % 1000 == 0) output[idx % 1024] = a;
        }
    }
}
"""

# ==============================================================================
# SECTION 4: SENSORS (V20 SIMPLE LOGIC - AS REQUESTED)
# ==============================================================================

class HardwareHAL:
    @staticmethod
    def get_cpu_temp():
        # ORIGINAL V20 SIMPLE LOGIC
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                val = float(f.read().strip())
                if val > 1000: val /= 1000.0
                return val
        except: pass
        return 0.0

    @staticmethod
    def get_gpu_temp():
        try:
            o = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True)
            return float(o.decode().strip())
        except: return 0.0

    @staticmethod
    def force_fans():
        try: subprocess.run("nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: pass

# ==============================================================================
# SECTION 5: BENCHMARK (TEXT MODE - 60s)
# ==============================================================================

def cpu_task(stop, throttle, cnt):
    import math
    size = 60
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    B = [[random.random() for _ in range(size)] for _ in range(size)]
    while not stop.is_set():
        throttle.wait()
        _ = [[sum(a*b for a,b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]
        with cnt.get_lock(): cnt.value += 1

def gpu_task(stop, throttle):
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        mod = SourceModule(CUDA_SRC)
        func = mod.get_function("kxt_burn")
        out = cuda.mem_alloc(4096)
        while not stop.is_set():
            throttle.wait()
            func(out, np.uint32(time.time()), block=(512,1,1), grid=(128,1))
            cuda.Context.synchronize()
    except: time.sleep(1)

def run_benchmark():
    os.system('clear')
    print("=== KXT v45 SYSTEM AUDIT (V20 BASE) ===")
    
    threading.Thread(target=lambda: [HardwareHAL.force_fans(), time.sleep(5)], daemon=True).start()
    
    stop = mp.Event()
    throttle = mp.Event()
    throttle.set()
    cnt = mp.Value('i', 0)
    procs = []
    
    print(f"\n[PHASE 1] CPU LOAD ({CONFIG['BENCH_TIME']}s)")
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_task, args=(stop, throttle, cnt))
        p.start(); procs.append(p)
    
    start = time.time()
    throttled = False
    
    try:
        while time.time() - start < CONFIG['BENCH_TIME']:
            rem = int(CONFIG['BENCH_TIME'] - (time.time() - start))
            c = HardwareHAL.get_cpu_temp()
            
            # 79C Limit
            if not throttled and c >= CONFIG['THROTTLE_TEMP']:
                throttle.clear(); throttled = True
            elif throttled and c <= CONFIG['RESUME_TEMP']:
                throttle.set(); throttled = False
            
            status = "MAX POWER" if not throttled else "THROTTLED"
            sys.stdout.write(f"\rTime: {rem}s | CPU: {c:.1f}C | Status: {status}    ")
            sys.stdout.flush()
            time.sleep(0.5)
    except KeyboardInterrupt: pass
    
    print(f"\n\n[PHASE 2] GPU LOAD ({CONFIG['BENCH_TIME']}s)")
    gp = mp.Process(target=gpu_task, args=(stop, throttle))
    gp.start(); procs.append(gp)
    
    start = time.time()
    try:
        while time.time() - start < CONFIG['BENCH_TIME']:
            rem = int(CONFIG['BENCH_TIME'] - (time.time() - start))
            c = HardwareHAL.get_cpu_temp()
            g = HardwareHAL.get_gpu_temp()
            
            if not throttled and c >= CONFIG['THROTTLE_TEMP']:
                throttle.clear(); throttled = True
            elif throttled and c <= CONFIG['RESUME_TEMP']:
                throttle.set(); throttled = False
            
            status = "MAX POWER" if not throttled else "THROTTLED"
            sys.stdout.write(f"\rTime: {rem}s | CPU: {c:.1f}C | GPU: {g:.1f}C | Status: {status}    ")
            sys.stdout.flush()
            time.sleep(0.5)
    except KeyboardInterrupt: pass
    
    stop.set()
    for p in procs: p.terminate()
    print("\n\n[DONE] Starting Miner...")
    time.sleep(2)

# ==============================================================================
# SECTION 6: MINING CORE (V19 LOGIC)
# ==============================================================================

class StratumClient(threading.Thread):
    def __init__(self, state, job_q, res_q, log_q):
        super().__init__()
        self.s = state; self.j = job_q; self.r = res_q; self.l = log_q
        self.sock = None; self.mid = 1; self.buf = ""; self.daemon = True

    def run(self):
        while not EXIT_FLAG.is_set():
            try:
                self.sock = socket.create_connection((CONFIG['POOL'], CONFIG['PORT']), timeout=10)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                self.send("mining.subscribe", ["KXT-v19-Engine"])
                self.send("mining.authorize", [CONFIG['USER'], CONFIG['PASS']])
                self.s.connected.value = True
                self.l.put(("NET", "Connected"))
                
                while not EXIT_FLAG.is_set():
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        
                        # --- V19 SUBMISSION FORMAT ---
                        # This specific parameter order is what worked in v19
                        self.send("mining.submit", [
                            CONFIG['USER'],
                            r['jid'],
                            r['en2'],
                            r['ntime'],
                            r['nonce']
                        ])
                        # -----------------------------
                        
                        self.l.put(("TX", f"Nonce {r['nonce']}"))
                        with self.s.shares.get_lock(): self.s.shares.value += 1
                        
                    r, _, _ = select.select([self.sock], [], [], 0.1)
                    if r:
                        d = self.sock.recv(4096)
                        if not d: raise ConnectionError("Closed")
                        self.buf += d.decode()
                        while '\n' in self.buf:
                            l, self.buf = self.buf.split('\n', 1)
                            if l: self.parse(json.loads(l))
            except Exception as e:
                self.s.connected.value = False
                time.sleep(5)

    def send(self, m, p):
        try:
            msg = json.dumps({"id": self.mid, "method": m, "params": p}) + "\n"
            self.sock.sendall(msg.encode())
            self.mid += 1
        except: pass

    def parse(self, msg):
        mid = msg.get('id')
        if mid == 1 and msg.get('result'):
            self.s.en1 = msg['result'][1]
            self.s.en2sz = msg['result'][2]
            
        if mid and mid > 2:
            if msg.get('result'):
                with self.s.accepted.get_lock(): self.s.accepted.value += 1
                self.l.put(("RX", "Share ACCEPTED"))
            else:
                with self.s.rejected.get_lock(): self.s.rejected.value += 1
                self.l.put(("RX", "Share REJECTED"))

        if msg.get('method') == 'mining.notify':
            p = msg['params']
            self.l.put(("JOB", f"Block {p[0][:8]}"))
            if p[8]:
                while not self.j.empty(): 
                    try: self.j.get_nowait()
                    except: pass
            
            # Pass En1/Size to workers
            job = (p, self.s.en1, self.s.en2sz)
            for _ in range(mp.cpu_count() * 2): self.j.put(job)

def miner_worker(id, job_q, res_q, stop, throttle):
    # --- V19 HASHING ENGINE ---
    nonce = id * 5000000
    while not stop.is_set():
        throttle.wait()
        try:
            try:
                job, en1, en2sz = job_q.get(timeout=0.1)
                jid, prev, c1, c2, mb, ver, nbits, ntime, clean = job
                if clean: nonce = id * 5000000
                
                # V19: Generate Random Extranonce2
                en2_bin = os.urandom(en2sz)
                en2 = binascii.hexlify(en2_bin).decode()
                
                # V19: Calculate Coinbase
                coinbase = binascii.unhexlify(c1 + en1 + en2 + c2)
                coinbase_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
                
                # V19: Calculate Merkle Root
                merkle_root = coinbase_hash
                for branch in mb:
                    merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + binascii.unhexlify(branch)).digest()).digest()
                    
                # V19: Assemble Header (Little Endian Swaps)
                header = (
                    binascii.unhexlify(ver)[::-1] +
                    binascii.unhexlify(prev)[::-1] +
                    merkle_root +
                    binascii.unhexlify(ntime)[::-1] +
                    binascii.unhexlify(nbits)[::-1]
                )
                
                # V19: Hashing Loop
                target = b'\x00\x00' # Optimization
                for n in range(nonce, nonce+2000):
                    nonce_bin = struct.pack('<I', n)
                    h = header + nonce_bin
                    d = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                    
                    if d.endswith(target):
                        # V19: Success Packet
                        res_q.put({
                            'jid': jid, 
                            'en2': en2, 
                            'ntime': ntime, 
                            'nonce': binascii.hexlify(nonce_bin).decode()
                        })
                        break
                nonce += 2000
                
            except queue.Empty: continue
        except: continue

# ==============================================================================
# SECTION 7: PROXY (V20 BASE)
# ==============================================================================

class Proxy(threading.Thread):
    def __init__(self, log_q, state):
        super().__init__(); self.l = log_q; self.s = state; self.daemon = True
    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", CONFIG['PROXY_PORT'])); s.listen(50)
            self.l.put(("PRX", f"Proxy {CONFIG['PROXY_PORT']}"))
            while not EXIT_FLAG.is_set():
                try: c, a = s.accept(); threading.Thread(target=self.h, args=(c,), daemon=True).start()
                except: pass
        except: pass
    def h(self, c):
        try:
            with self.s.proxy_clients.get_lock(): self.s.proxy_clients.value += 1
            p = socket.create_connection((CONFIG['POOL'], 3333))
            while not EXIT_FLAG.is_set():
                r, _, _ = select.select([c, p], [], [], 1)
                if c in r:
                    d = c.recv(4096)
                    if not d: break
                    p.sendall(d)
                if p in r:
                    d = p.recv(4096)
                    if not d: break
                    c.sendall(d)
        except: pass
        finally:
            try: c.close()
            except: pass
            try: p.close()
            except: pass
            with self.s.proxy_clients.get_lock(): self.s.proxy_clients.value -= 1

# ==============================================================================
# SECTION 8: DASHBOARD (V20 LAYOUT)
# ==============================================================================

def draw_box(stdscr, y, x, h, w, title, color_pair):
    try:
        stdscr.attron(color_pair)
        stdscr.addch(y, x, curses.ACS_ULCORNER); stdscr.addch(y, x + w - 1, curses.ACS_URCORNER)
        stdscr.addch(y + h - 1, x, curses.ACS_LLCORNER); stdscr.addch(y + h - 1, x + w - 1, curses.ACS_LRCORNER)
        stdscr.hline(y, x + 1, curses.ACS_HLINE, w - 2); stdscr.hline(y + h - 1, x + 1, curses.ACS_HLINE, w - 2)
        stdscr.vline(y + 1, x, curses.ACS_VLINE, h - 2); stdscr.vline(y + 1, x + w - 1, curses.ACS_VLINE, h - 2)
        stdscr.addstr(y, x + 2, f" {title} "); stdscr.attroff(color_pair)
    except: pass

def draw_bar(val, max_val, width=10):
    pct = max(0.0, min(1.0, val / max_val))
    fill = int(pct * width)
    return f"[{'|'*fill}{' '*(width-fill)}]"

def dashboard(stdscr, state, job_q, res_q, log_q):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    stdscr.nodelay(True); curses.curs_set(0)
    
    StratumClient(state, job_q, res_q, log_q).start()
    Proxy(log_q, state).start()
    
    workers = []
    stop = mp.Event(); throttle = mp.Event(); throttle.set()
    
    for i in range(mp.cpu_count()):
        p = mp.Process(target=miner_worker, args=(i, job_q, res_q, stop, throttle))
        p.start(); workers.append(p)
        
    log_buffer = []
    throttled = False
    
    while not EXIT_FLAG.is_set():
        while not log_q.empty():
            log_buffer.append(log_q.get())
            if len(log_buffer) > 20: log_buffer.pop(0)
            
        c = HardwareHAL.get_cpu_temp()
        g = HardwareHAL.get_gpu_temp()
        
        if not throttled and c >= CONFIG['THROTTLE_TEMP']:
            throttle.clear(); throttled = True
            log_q.put(("WARN", "Temp > 79C. Throttling..."))
        elif throttled and c <= CONFIG['RESUME_TEMP']:
            throttle.set(); throttled = False
            log_q.put(("INFO", "Resuming..."))

        stdscr.erase(); h, w = stdscr.getmaxyx()
        
        stdscr.addstr(0, 0, " KXT MINER v45 ".center(w), curses.A_REVERSE)
        
        draw_box(stdscr, 2, 1, 6, 28, "LOCAL", curses.color_pair(3))
        stdscr.addstr(3, 3, f"CPU: {c:.1f}C {draw_bar(c, 90, 5)}")
        stdscr.addstr(4, 3, f"GPU: {g:.1f}C {draw_bar(g, 90, 5)}")
        status = "MINING" if not throttled else "THROTTLED"
        stdscr.addstr(5, 3, f"Status: {status}", curses.color_pair(1 if not throttled else 3))
            
        draw_box(stdscr, 2, 30, 6, 28, "NETWORK", curses.color_pair(3))
        stdscr.addstr(3, 32, f"Connected: {state.connected.value}")
        stdscr.addstr(4, 32, f"Shares: {state.shares.value}")
        
        draw_box(stdscr, 2, 59, 6, 28, "STATS", curses.color_pair(3))
        stdscr.addstr(3, 61, f"Accepted: {state.accepted.value}", curses.color_pair(1))
        stdscr.addstr(4, 61, f"Rejected: {state.rejected.value}", curses.color_pair(3))
        
        draw_box(stdscr, 2, 88, 6, 24, "PROXY", curses.color_pair(3))
        stdscr.addstr(3, 90, f"Clients: {state.proxy_clients.value}")
        
        stdscr.hline(8, 0, curses.ACS_HLINE, w)
        for i, (lvl, msg) in enumerate(log_buffer):
            if 9+i >= h-1: break
            col = curses.color_pair(4)
            if lvl == "RX": col = curses.color_pair(1)
            if lvl == "ERR": col = curses.color_pair(3)
            stdscr.addstr(9+i, 2, f"[{lvl}] {msg}", col)
            
        stdscr.refresh()
        try:
            if stdscr.getch() == ord('q'): EXIT_FLAG.set()
        except: pass
        time.sleep(0.1)
        
    stop.set()
    for p in workers: p.terminate()

if __name__ == "__main__":
    try:
        run_benchmark()
        
        manager = mp.Manager()
        state = manager.Namespace()
        state.connected = manager.Value('b', False)
        state.shares = manager.Value('i', 0)
        state.accepted = manager.Value('i', 0)
        state.rejected = manager.Value('i', 0)
        state.proxy_clients = manager.Value('i', 0)
        state.en1 = None
        state.en2sz = 4
        
        job_queue = manager.Queue()
        res_queue = manager.Queue()
        log_queue = manager.Queue()
        
        curses.wrapper(dashboard, state, job_queue, res_queue, log_queue)
    except KeyboardInterrupt: pass
