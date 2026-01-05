#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KXT MINER SUITE (v39) - HYBRID RELEASE
======================================
Base: Beta v33 (Stable Architecture)
Engine: KXT Heavy Load + Smart Governor
Fixes: Miner not opening, Math Error, Temp Readings, Throttling
"""

import sys
import os
import time
import socket
import json
import threading
import multiprocessing as mp
import binascii
import struct
import hashlib
import random
import select
import subprocess
import signal
import resource
import glob
import math  # <--- FIXED: Global Import
import re
import queue
from datetime import datetime

# ==============================================================================
# SECTION 1: SYSTEM HARDENING
# ==============================================================================

# Global Signal Handler to prevent BrokenPipeError
def signal_handler(signum, frame):
    print("\n[KXT] Shutdown Sequence Initiated...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def kxt_boot():
    # File Descriptors
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))
    except: pass

kxt_boot()

try: import curses
except: 
    print("[FATAL] Curses missing. Run in terminal.")
    sys.exit(1)

# ==============================================================================
# SECTION 2: CONFIGURATION
# ==============================================================================

CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "WORKER": "rig1",
    "PASS": "x",
    "PROXY_PORT": 60060,
    
    # Bench Settings
    "BENCH_TIME": 60,  # 60 Seconds per stage
    "THROTTLE_TEMP": 79.0, # Limit
}

# ==============================================================================
# SECTION 3: CUDA KERNEL (VOLATILE)
# ==============================================================================

CUDA_KXT_SRC = """
extern "C" {
    #include <stdint.h>
    __global__ void kxt_burn(uint32_t *output, uint32_t seed) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        volatile uint32_t a = 0x6a09e667 + idx;
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
# SECTION 4: ADVANCED SENSOR HAL (THE FIX)
# ==============================================================================

class HAL:
    @staticmethod
    def get_cpu_temp():
        readings = []
        # Method 1: Thermal Zones
        try:
            for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
                try:
                    with open(p) as f:
                        v = float(f.read().strip()) / 1000.0
                        if 20 < v < 115 and v != 100.0: readings.append(v)
                except: pass
        except: pass
        
        # Method 2: Hwmon
        try:
            for p in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
                try:
                    with open(p) as f:
                        v = float(f.read().strip()) / 1000.0
                        if 20 < v < 115 and v != 100.0: readings.append(v)
                except: pass
        except: pass
        
        # Method 3: Sensors CMD
        try:
            o = subprocess.check_output("sensors", shell=True).decode()
            for x in re.findall(r'\+([0-9]+\.[0-9]+)', o):
                v = float(x)
                if 20 < v < 115 and v != 100.0: readings.append(v)
        except: pass

        if readings: return max(readings) # Returns the highest (Real Die Temp)
        return 0.0

    @staticmethod
    def get_gpu_temp():
        try:
            o = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True)
            return float(o.decode().strip())
        except: return 0.0

    @staticmethod
    def set_fans_max():
        try:
            subprocess.run("nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: pass

# ==============================================================================
# SECTION 5: BENCHMARK (TEXT MODE LIKE v33)
# ==============================================================================

def cpu_bench_task(stop_ev, throttle_ev):
    import math
    while not stop_ev.is_set():
        throttle_ev.wait() # Pause if throttled
        
        # KXT Heavy Load
        size = 50
        A = [[random.random() for _ in range(size)] for _ in range(size)]
        B = [[random.random() for _ in range(size)] for _ in range(size)]
        _ = [[sum(a*b for a,b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

def gpu_bench_task(stop_ev, throttle_ev):
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_KXT_SRC)
        func = mod.get_function("kxt_burn")
        out = cuda.mem_alloc(4096)
        
        while not stop_ev.is_set():
            throttle_ev.wait()
            func(out, np.uint32(time.time()), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
    except: time.sleep(0.1)

def run_benchmark():
    os.system('clear')
    print("=== KXT v39 SYSTEM AUDIT ===")
    
    # Fan Force
    threading.Thread(target=lambda: [HAL.set_fans_max(), time.sleep(10)], daemon=True).start()
    
    # Setup
    stop = mp.Event()
    throttle = mp.Event()
    throttle.set() # Allow running
    procs = []
    
    # Start CPU Engines
    print(f"\n[PHASE 1] CPU THERMAL STRESS ({CONFIG['BENCH_TIME']}s)")
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_bench_task, args=(stop, throttle))
        p.start(); procs.append(p)
        
    start = time.time()
    throttled = False
    
    try:
        while time.time() - start < CONFIG['BENCH_TIME']:
            rem = int(CONFIG['BENCH_TIME'] - (time.time() - start))
            t = HAL.get_cpu_temp()
            
            # Smart Governor Logic
            if not throttled and t > CONFIG['THROTTLE_TEMP']:
                throttle.clear() # Pause workers
                throttled = True
            elif throttled and t < (CONFIG['THROTTLE_TEMP'] - 5.0):
                throttle.set() # Resume workers
                throttled = False
                
            status = "MAX LOAD" if not throttled else "THROTTLED"
            sys.stdout.write(f"\rTime: {rem}s | CPU: {t:.1f}C | Status: {status}    ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt: pass
    
    # Start GPU
    print(f"\n\n[PHASE 2] FULL SYSTEM LOAD ({CONFIG['BENCH_TIME']}s)")
    gp = mp.Process(target=gpu_bench_task, args=(stop, throttle))
    gp.start(); procs.append(gp)
    
    start = time.time()
    try:
        while time.time() - start < CONFIG['BENCH_TIME']:
            rem = int(CONFIG['BENCH_TIME'] - (time.time() - start))
            t = HAL.get_cpu_temp()
            g = HAL.get_gpu_temp()
            
            # Logic
            if not throttled and t > CONFIG['THROTTLE_TEMP']:
                throttle.clear(); throttled = True
            elif throttled and t < (CONFIG['THROTTLE_TEMP'] - 5.0):
                throttle.set(); throttled = False
                
            status = "MAX LOAD" if not throttled else "THROTTLED"
            sys.stdout.write(f"\rTime: {rem}s | CPU: {t:.1f}C | GPU: {g:.1f}C | Status: {status}    ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt: pass
    
    stop.set()
    for p in procs: p.terminate()
    print("\n\n[DONE] Benchmark Complete. Starting Miner...")
    time.sleep(2)

# ==============================================================================
# SECTION 6: STRATUM CLIENT
# ==============================================================================

class StratumClient(threading.Thread):
    def __init__(self, state, job_q, res_q, log_q):
        super().__init__()
        self.state = state; self.job_q = job_q; self.res_q = res_q; self.log_q = log_q
        self.sock = None; self.msg_id = 1; self.buffer = ""; self.daemon = True

    def run(self):
        while True:
            try:
                self.sock = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=15)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                self.send("mining.subscribe", ["KXT-v39"])
                self.send("mining.authorize", [f"{CONFIG['WALLET']}.{CONFIG['WORKER']}_local", CONFIG['PASS']])
                self.state.connected = True
                self.log_q.put(("NET", "Connected"))
                
                while True:
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        self.send("mining.submit", [
                            f"{CONFIG['WALLET']}.{CONFIG['WORKER']}_local",
                            r['jid'], r['en2'], r['ntime'], r['nonce']
                        ])
                        self.log_q.put(("TX", f"Nonce {r['nonce']}"))
                        self.state.local_tx += 1

                    r, _, _ = select.select([self.sock], [], [], 0.1)
                    if r:
                        d = self.sock.recv(4096)
                        if not d: break
                        self.buffer += d.decode()
                        while '\n' in self.buffer:
                            line, self.buffer = self.buffer.split('\n', 1)
                            if line.strip(): self.parse(json.loads(line))
            except:
                self.state.connected = False
                time.sleep(5)

    def send(self, method, params):
        try:
            msg = json.dumps({"id": self.msg_id, "method": method, "params": params}) + "\n"
            self.sock.sendall(msg.encode())
            self.msg_id += 1
        except: pass

    def parse(self, msg):
        mid = msg.get('id')
        method = msg.get('method')
        
        if mid == 1 and msg.get('result'):
            self.state.extranonce1 = msg['result'][1]
            self.state.extranonce2_size = msg['result'][2]
            
        if mid and mid > 2:
            if msg.get('result'):
                self.state.accepted += 1
                self.log_q.put(("RX", "Accepted"))
            else:
                self.state.rejected += 1
                self.log_q.put(("RX", "Rejected"))
                
        if method == 'mining.notify':
            self.log_q.put(("JOB", f"Block {msg['params'][0][:8]}"))
            if msg['params'][8]:
                while not self.job_q.empty(): 
                    try: self.job_q.get_nowait()
                    except: pass
            
            job = (msg['params'], self.state.extranonce1, self.state.extranonce2_size)
            for _ in range(mp.cpu_count() * 2): self.job_q.put(job)

# ==============================================================================
# SECTION 7: WORKERS
# ==============================================================================

def miner_worker(id, job_q, res_q, stop):
    nonce = id * 10000000
    cur_job = None
    
    while not stop.is_set():
        try:
            try:
                params, en1, en2sz = job_q.get(timeout=0.1)
                cur_job = (params, en1, en2sz)
                if params[8]: nonce = id * 10000000
            except: 
                if not cur_job: continue
            
            params, en1, en2sz = cur_job
            jid, prev, c1, c2, mb, ver, nbits, ntime, clean = params
            
            en2 = struct.pack('>I', random.randint(0, 2**32-1)).hex().zfill(en2sz*2)
            coinbase = binascii.unhexlify(c1 + en1 + en2 + c2)
            root = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            for b in mb: root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(b)).digest()).digest()
            
            header = binascii.unhexlify(ver)[::-1] + binascii.unhexlify(prev)[::-1] + root + binascii.unhexlify(ntime)[::-1] + binascii.unhexlify(nbits)[::-1]
            target = b'\x00\x00\x00\x00'
            
            for n in range(nonce, nonce + 5000):
                h = header + struct.pack('<I', n)
                d = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                if d.endswith(target):
                    res_q.put({'jid': jid, 'en2': en2, 'ntime': ntime, 'nonce': struct.pack('>I', n).hex()})
                    break
            nonce += 5000
        except: continue

class Proxy(threading.Thread):
    def __init__(self, log_q, state):
        super().__init__()
        self.log_q = log_q; self.state = state; self.daemon = True
    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", CONFIG['PROXY_PORT'])); s.listen(50)
            self.log_q.put(("PRX", f"Listen {CONFIG['PROXY_PORT']}"))
            while True:
                try: c, a = s.accept(); threading.Thread(target=self.h, args=(c,a), daemon=True).start()
                except: pass
        except: pass
    
    def h(self, c, a):
        try:
            self.state.proxy_clients += 1
            p = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']))
            inputs = [c, p]
            while True:
                r, _, _ = select.select(inputs, [], [], 1)
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
            self.state.proxy_clients -= 1

# ==============================================================================
# SECTION 8: DASHBOARD (The one you liked)
# ==============================================================================

def dashboard(stdscr, state, job_q, res_q, log_q):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
    stdscr.nodelay(True)
    
    client = StratumClient(state, job_q, res_q, log_q)
    client.start()
    Proxy(log_q, state).start()
    
    stop = mp.Event()
    workers = []
    for i in range(mp.cpu_count()):
        p = mp.Process(target=miner_worker, args=(i, job_q, res_q, stop))
        p.start(); workers.append(p)
        
    logs = []
    
    while True:
        while not log_q.empty():
            logs.append(log_q.get())
            if len(logs) > 20: logs.pop(0)
            
        stdscr.erase(); h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " KXT MINER v39 ".center(w), curses.color_pair(5))
        
        c = HAL.get_cpu_temp()
        g = HAL.get_gpu_temp()
        
        stdscr.addstr(2, 2, "LOCAL", curses.color_pair(4))
        stdscr.addstr(3, 2, f"CPU: {c:.1f}C")
        stdscr.addstr(4, 2, f"GPU: {g:.1f}C")
        
        stdscr.addstr(2, 30, "NETWORK", curses.color_pair(4))
        stdscr.addstr(3, 30, f"Link: {state.connected}")
        stdscr.addstr(4, 30, f"Shares: {state.local_tx}")
        stdscr.addstr(5, 30, f"Acc/Rej: {state.accepted}/{state.rejected}")
        
        stdscr.addstr(2, 60, "PROXY", curses.color_pair(4))
        stdscr.addstr(3, 60, f"Clients: {state.proxy_clients}")
        
        stdscr.hline(7, 0, '-', w)
        for i, (lvl, msg) in enumerate(logs):
            if 8+i >= h-1: break
            col = curses.color_pair(1)
            if lvl == "RX": col = curses.color_pair(2)
            stdscr.addstr(8+i, 2, f"[{lvl}] {msg}", col)
            
        stdscr.refresh()
        try:
            if stdscr.getch() == ord('q'): break
        except: pass
        time.sleep(0.1)
        
    stop.set()
    for p in workers: p.terminate()

if __name__ == "__main__":
    try:
        # 1. Run Bench in Text Mode (Proven to work)
        run_benchmark()
        
        # 2. Start Miner GUI
        man = mp.Manager()
        state = man.Namespace()
        state.connected = False
        state.local_tx = 0
        state.accepted = 0
        state.rejected = 0
        state.proxy_clients = 0
        state.extranonce1 = "00000000"
        state.extranonce2_size = 4
        
        job_q = man.Queue()
        res_q = man.Queue()
        log_q = man.Queue()
        
        curses.wrapper(dashboard, state, job_q, res_q, log_q)
    except KeyboardInterrupt: pass
