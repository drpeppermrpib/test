#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KXT MINER SUITE (v35) - REFORGED
================================
Architecture: Classic Sensors + 4-Column UI + Heavy Load
Target: solo.stratum.braiins.com:3333
Fixes: Temp Readings, Visual Layout, Broken Pipes
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
import math  # GLOBAL IMPORT (Fixes NameError)
from datetime import datetime

# ==============================================================================
# SECTION 1: SYSTEM CORE & SIGNAL HANDLING
# ==============================================================================

EXIT_FLAG = mp.Event()

def signal_handler(signum, frame):
    """Prevents BrokenPipeError by handling exit signals gracefully."""
    EXIT_FLAG.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def boot_check():
    # Fix File Descriptors
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))
    except: pass
    
    # Install Drivers if missing
    req = ["psutil", "requests"]
    for r in req:
        try: __import__(r)
        except: 
            try: subprocess.check_call([sys.executable, "-m", "pip", "install", r])
            except: pass

boot_check()

try: import curses
except: 
    print("[FAIL] Curses library missing. Run in standard Linux terminal.")
    sys.exit(1)

# ==============================================================================
# SECTION 2: CONFIGURATION
# ==============================================================================

CONFIG = {
    "POOL": "solo.stratum.braiins.com",
    "PORT": 3333,
    "USER": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.rig1",
    "PASS": "x",
    "PROXY_PORT": 60060,
    
    # Bench Duration (Seconds)
    "BENCH_TIME": 300,
    
    # Offset for CPU Temp (If it reads too high/low vs motherboard)
    "TEMP_OFFSET": 0.0 
}

# ==============================================================================
# SECTION 3: CUDA KERNEL (HIGH INTENSITY)
# ==============================================================================

CUDA_KXT_SRC = """
extern "C" {
    #include <stdint.h>

    __global__ void kxt_burn(uint32_t *output, uint32_t seed) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // VOLATILE forces execution
        volatile uint32_t a = 0x6a09e667 + idx;
        volatile uint32_t b = 0xbb67ae85;
        
        // 50,000 Iterations per thread = Massive Load
        #pragma unroll 128
        for(int i=0; i < 50000; i++) {
            a = (a << 5) | (a >> 27);
            b ^= a;
            a += b + 0xDEADBEEF;
        }
        
        if (a == 0) output[0] = b;
    }
}
"""

# ==============================================================================
# SECTION 4: HARDWARE SENSORS (CLASSIC METHOD)
# ==============================================================================

class HardwareHAL:
    @staticmethod
    def get_cpu_temp():
        # 1. Try `sensors` command (The "Good" way from before)
        try:
            out = subprocess.check_output("sensors", shell=True).decode()
            for line in out.splitlines():
                # Look for specific Package/Core lines
                if "Package id 0" in line or "Tdie" in line:
                    parts = line.split('+')
                    if len(parts) > 1:
                        val = float(parts[1].split('.')[0])
                        return val + CONFIG['TEMP_OFFSET']
        except: pass

        # 2. Fallback to Thermal Zones
        try:
            zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
            for z in zones:
                with open(z, 'r') as f:
                    val = float(f.read().strip())
                    if val > 1000: val /= 1000.0
                    if val > 20: return val + CONFIG['TEMP_OFFSET']
        except: pass
            
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
        try:
            for pwm in glob.glob("/sys/class/hwmon/hwmon*/pwm*"):
                if os.access(pwm, os.W_OK):
                    with open(pwm, 'w') as f: f.write("255")
        except: pass

# ==============================================================================
# SECTION 5: UTILS
# ==============================================================================

def draw_bar(val, max_val, width=10):
    pct = min(1.0, val / max_val)
    fill = int(pct * width)
    bar = "|" * fill + " " * (width - fill)
    return f"[{bar}]"

# ==============================================================================
# SECTION 6: LOAD GENERATORS
# ==============================================================================

def cpu_stress(stop_ev, counter):
    # Pure Math Loop for Heat
    while not stop_ev.is_set():
        # Matrix simulation logic
        x = 1.5
        for i in range(5000):
            x = math.sin(x) * math.sqrt(x + 1)
        
        with counter.get_lock():
            counter.value += 5000

def gpu_stress(stop_ev):
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_KXT_SRC)
        func = mod.get_function("kxt_burn")
        out = cuda.mem_alloc(4096)
        
        while not stop_ev.is_set():
            func(out, np.uint32(time.time()), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
    except: 
        time.sleep(0.1)

# ==============================================================================
# SECTION 7: STRATUM CLIENT
# ==============================================================================

class StratumClient(threading.Thread):
    def __init__(self, state, job_q, res_q, log_q):
        super().__init__()
        self.state = state
        self.job_q = job_q
        self.res_q = res_q
        self.log_q = log_q
        self.sock = None
        self.msg_id = 1
        self.buffer = ""
        self.daemon = True
        self.extranonce1 = None
        self.extranonce2_size = 4

    def connect(self):
        try:
            self.sock = socket.create_connection((CONFIG['POOL'], CONFIG['PORT']), timeout=10)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.state.connected.value = True
            return True
        except:
            self.state.connected.value = False
            return False

    def run(self):
        while not EXIT_FLAG.is_set():
            if not self.sock:
                if self.connect():
                    self.send("mining.subscribe", ["KXT-v35"])
                    self.send("mining.authorize", [CONFIG['USER'], CONFIG['PASS']])
                    self.log_q.put(("NET", "Connected"))
                else:
                    time.sleep(5)
                    continue

            try:
                # Flush
                while not self.res_q.empty():
                    r = self.res_q.get()
                    self.send("mining.submit", [
                        CONFIG['USER'], r['jid'], r['en2'], r['ntime'], r['nonce']
                    ])
                    self.log_q.put(("TX", f"Nonce {r['nonce']}"))
                    with self.state.shares.get_lock(): self.state.shares.value += 1

                # Read
                r, _, _ = select.select([self.sock], [], [], 0.1)
                if r:
                    d = self.sock.recv(4096)
                    if not d: 
                        self.sock.close(); self.sock = None
                        continue
                    self.buffer += d.decode()
                    while '\n' in self.buffer:
                        line, self.buffer = self.buffer.split('\n', 1)
                        if line: self.process(json.loads(line))
            except Exception as e:
                self.sock = None
                self.state.connected.value = False
                time.sleep(2)

    def send(self, m, p):
        if self.sock:
            try:
                msg = json.dumps({"id": self.msg_id, "method": m, "params": p}) + "\n"
                self.sock.sendall(msg.encode())
                self.msg_id += 1
            except: self.sock = None

    def process(self, msg):
        mid = msg.get('id')
        method = msg.get('method')
        
        if mid == 1 and msg.get('result'):
            self.extranonce1 = msg['result'][1]
            self.extranonce2_size = msg['result'][2]
            
        if mid and mid > 2:
            if msg.get('result'):
                with self.state.accepted.get_lock(): self.state.accepted.value += 1
                self.log_q.put(("RX", "Accepted"))
            else:
                with self.state.rejected.get_lock(): self.state.rejected.value += 1
                self.log_q.put(("RX", "Rejected"))

        if method == 'mining.notify':
            p = msg['params']
            self.log_q.put(("JOB", f"Block {p[0][:8]}"))
            if p[8]:
                while not self.job_q.empty(): 
                    try: self.job_q.get_nowait()
                    except: pass
            
            en1 = self.extranonce1 if self.extranonce1 else "00000000"
            job = (p, en1, self.extranonce2_size)
            for _ in range(mp.cpu_count() * 2): self.job_q.put(job)

def cpu_worker(id, job_q, res_q, stop, counter):
    nonce = id * 5000000
    cur_job = None
    
    while not stop.is_set():
        try:
            try:
                params, en1, en2sz = job_q.get(timeout=0.1)
                cur_job = params
                if params[8]: nonce = id * 5000000
            except: 
                if not cur_job: continue
            
            jid, prev, c1, c2, mb, ver, nbits, ntime, clean = cur_job
            
            en2 = struct.pack('>I', random.randint(0, 2**32-1)).hex().zfill(en2sz*2)
            coinbase = binascii.unhexlify(c1 + en1 + en2 + c2)
            cb_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            root = cb_hash
            for b in mb: root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(b)).digest()).digest()
            
            header = (
                binascii.unhexlify(ver)[::-1] + binascii.unhexlify(prev)[::-1] +
                root + binascii.unhexlify(ntime)[::-1] + binascii.unhexlify(nbits)[::-1]
            )
            
            for n in range(nonce, nonce + 20000):
                h = header + struct.pack('<I', n)
                d = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                if d.endswith(b'\x00\x00'): 
                    res_q.put({
                        'jid': jid, 'en2': en2, 'ntime': ntime, 
                        'nonce': struct.pack('>I', n).hex()
                    })
                    break
            nonce += 20000
            with counter.get_lock(): counter.value += 20000
        except: continue

class Proxy(threading.Thread):
    def __init__(self, log_q, state):
        super().__init__()
        self.log_q = log_q
        self.state = state
        self.daemon = True
    
    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", CONFIG['PROXY_PORT']))
            s.listen(50)
            self.log_q.put(("PRX", f"Port {CONFIG['PROXY_PORT']}"))
            while not EXIT_FLAG.is_set():
                try:
                    c, a = s.accept()
                    threading.Thread(target=self.handle, args=(c,a), daemon=True).start()
                except: pass
        except: pass
            
    def handle(self, c, a):
        try:
            with self.state.proxy_clients.get_lock(): self.state.proxy_clients.value += 1
            p = socket.create_connection((CONFIG['POOL'], CONFIG['PORT']))
            inputs = [c, p]
            while not EXIT_FLAG.is_set():
                r, _, _ = select.select(inputs, [], [], 1)
                if not r: continue
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
            c.close(); p.close()
            with self.state.proxy_clients.get_lock(): self.state.proxy_clients.value -= 1

# ==============================================================================
# SECTION 8: 4-COLUMN DASHBOARD + BENCHMARK UI
# ==============================================================================

def draw_box(stdscr, y, x, h, w, title, color):
    try:
        stdscr.attron(color)
        stdscr.addch(y, x, curses.ACS_ULCORNER)
        stdscr.addch(y, x + w - 1, curses.ACS_URCORNER)
        stdscr.addch(y + h - 1, x, curses.ACS_LLCORNER)
        stdscr.addch(y + h - 1, x + w - 1, curses.ACS_LRCORNER)
        
        for i in range(1, w - 1):
            stdscr.addch(y, x + i, curses.ACS_HLINE)
            stdscr.addch(y + h - 1, x + i, curses.ACS_HLINE)
            
        for i in range(1, h - 1):
            stdscr.addch(y + i, x, curses.ACS_VLINE)
            stdscr.addch(y + i, x + w - 1, curses.ACS_VLINE)
            
        stdscr.addstr(y, x + 2, f" {title} ")
        stdscr.attroff(color)
    except: pass

def main_gui(stdscr, state, job_q, res_q, log_q):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK) 
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)   
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)  
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    stdscr.nodelay(True)
    
    # --- BENCHMARK ---
    threading.Thread(target=lambda: [HardwareHAL.set_fans_max(), time.sleep(15)], daemon=True).start()
    
    stop_bench = mp.Event()
    cnt = mp.Value('d', 0.0)
    
    # 1. CPU
    procs = []
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_stress, args=(stop_bench, cnt))
        p.start()
        procs.append(p)
    
    # 2. GPU
    gp = mp.Process(target=gpu_stress, args=(stop_bench,))
    gp.start()
    procs.append(gp)
    
    start = time.time()
    while time.time() - start < CONFIG['BENCH_TIME']:
        rem = CONFIG['BENCH_TIME'] - (time.time() - start)
        c = HardwareHAL.get_cpu_temp()
        g = HardwareHAL.get_gpu_temp()
        
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        draw_box(stdscr, 2, 2, 8, w-4, "KXT SYSTEM AUDIT", curses.color_pair(4))
        stdscr.addstr(4, 4, "STATUS: MAX LOAD TESTING")
        stdscr.addstr(5, 4, f"TIME REMAINING: {int(rem)}s")
        stdscr.addstr(6, 4, f"CPU: {c:.1f}C {draw_bar(c, 90)}")
        stdscr.addstr(7, 4, f"GPU: {g:.1f}C {draw_bar(g, 90)}")
        stdscr.refresh()
        
        if EXIT_FLAG.is_set(): break
        time.sleep(1)
        
    stop_bench.set()
    for p in procs: p.terminate()
    
    if EXIT_FLAG.is_set(): return

    # --- MINER ---
    client = StratumClient(state, job_q, res_q, log_q)
    client.start()
    
    Proxy(log_q, state).start()
    
    workers = []
    stop_miners = mp.Event()
    hash_cnt = mp.Value('d', 0.0)
    
    for i in range(mp.cpu_count()):
        p = mp.Process(target=cpu_worker, args=(i, job_q, res_q, stop_miners, hash_cnt))
        p.start()
        workers.append(p)
        
    logs = []
    last_h = 0.0
    curr_hr = 0.0
    
    while not EXIT_FLAG.is_set():
        while not log_q.empty():
            logs.append(log_q.get())
            if len(logs) > 50: logs.pop(0)
            
        t = hash_cnt.value
        curr_hr = (curr_hr * 0.9) + ((t - last_h) * 10 * 0.1)
        last_h = t
        
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        col_w = (w - 2) // 4
        
        stdscr.addstr(0, 0, " KXT MINER v35 ".center(w), curses.A_REVERSE)
        
        c = HardwareHAL.get_cpu_temp()
        g = HardwareHAL.get_gpu_temp()
        
        # 4 COLUMNS
        # 1. LOCAL
        draw_box(stdscr, 2, 0, 8, col_w, "LOCAL", curses.color_pair(3))
        stdscr.addstr(3, 2, f"CPU: {c:.1f}C")
        stdscr.addstr(4, 2, f"GPU: {g:.1f}C")
        if curr_hr > 1e9: hrs = f"{curr_hr/1e9:.2f} GH/s"
        elif curr_hr > 1e6: hrs = f"{curr_hr/1e6:.2f} MH/s"
        else: hrs = f"{curr_hr/1000:.2f} kH/s"
        stdscr.addstr(5, 2, f"HR: {hrs}")
        
        # 2. NETWORK
        draw_box(stdscr, 2, col_w, 8, col_w, "NETWORK", curses.color_pair(3))
        stdscr.addstr(3, col_w+2, f"Link: {state.connected.value}")
        stdscr.addstr(4, col_w+2, f"Shares: {state.shares.value}")
        
        # 3. STATS
        draw_box(stdscr, 2, col_w*2, 8, col_w, "STATS", curses.color_pair(3))
        stdscr.addstr(3, col_w*2+2, f"Acc: {state.accepted.value}")
        stdscr.addstr(4, col_w*2+2, f"Rej: {state.rejected.value}")
        
        # 4. PROXY
        draw_box(stdscr, 2, col_w*3, 8, col_w, "PROXY", curses.color_pair(3))
        stdscr.addstr(3, col_w*3+2, f"Clients: {state.proxy_clients.value}")
        stdscr.addstr(4, col_w*3+2, f"Tx: {state.shares.value}")
        
        # LOGS
        stdscr.hline(10, 0, '-', w)
        for i, (lvl, msg) in enumerate(logs):
            if 11+i >= h-1: break
            col = curses.color_pair(1)
            if lvl in ["ERR", "RX"]: col = curses.color_pair(2)
            try: stdscr.addstr(11+i, 2, f"[{lvl}] {msg}", col)
            except: pass
            
        stdscr.refresh()
        
        try: 
            if stdscr.getch() == ord('q'): EXIT_FLAG.set()
        except: pass
        
        time.sleep(0.1)
        
    stop_miners.set()
    for p in workers: p.terminate()

if __name__ == "__main__":
    man = mp.Manager()
    state = man.Namespace()
    state.connected = man.Value('b', False)
    state.shares = man.Value('i', 0)
    state.accepted = man.Value('i', 0)
    state.rejected = man.Value('i', 0)
    state.proxy_clients = man.Value('i', 0)
    
    job_q = man.Queue()
    res_q = man.Queue()
    log_q = man.Queue()
    
    try:
        curses.wrapper(main_gui, state, job_q, res_q, log_q)
    except KeyboardInterrupt: pass
