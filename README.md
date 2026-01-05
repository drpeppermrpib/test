#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KXT MINER SUITE (v37) - STABLE THROTTLE
=======================================
Architecture: Thermal Governor (79C) + Heavy Load
Target: solo.stratum.braiins.com:3333
Fixes: SyntaxError on line 448, Python 3.10 Compatibility
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
import math
import re
import queue
from datetime import datetime

# ==============================================================================
# SECTION 1: SYSTEM CORE
# ==============================================================================

EXIT_FLAG = mp.Event()

def signal_handler(signum, frame):
    if not EXIT_FLAG.is_set():
        EXIT_FLAG.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def boot_prep():
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))
    except: pass
    
    try: import curses
    except ImportError:
        print("[FAIL] Curses missing.")
        sys.exit(1)

boot_prep()
import curses

# ==============================================================================
# SECTION 2: CONFIGURATION
# ==============================================================================

CONFIG = {
    "POOL": "solo.stratum.braiins.com",
    "PORT": 3333,
    "USER": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.rig1",
    "PASS": "x",
    "PROXY_PORT": 60060,
    "BENCH_TIME": 300,
    "THROTTLE_TEMP": 79.0,
    "RESUME_TEMP": 75.0,
}

# ==============================================================================
# SECTION 3: CUDA KERNEL
# ==============================================================================

CUDA_SRC = """
extern "C" {
    #include <stdint.h>

    __global__ void kxt_heavy(uint32_t *output, uint32_t seed) {
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
# SECTION 4: SENSORS
# ==============================================================================

class HardwareHAL:
    @staticmethod
    def get_cpu_temp():
        readings = []
        try:
            for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
                try:
                    with open(path, "r") as f:
                        val = float(f.read().strip()) / 1000.0
                        if 20 < val < 115: readings.append(val)
                except: pass
        except: pass

        try:
            for path in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
                try:
                    with open(path, "r") as f:
                        val = float(f.read().strip()) / 1000.0
                        if 20 < val < 115: readings.append(val)
                except: pass
        except: pass

        try:
            out = subprocess.check_output("sensors", shell=True).decode()
            found = re.findall(r'\+([0-9]+\.[0-9]+)', out)
            for f in found:
                try:
                    val = float(f)
                    if 20 < val < 115: readings.append(val)
                except: pass
        except: pass

        if readings: return max(readings)
        return 0.0

    @staticmethod
    def get_gpu_temp():
        try:
            out = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True)
            return float(out.decode().strip())
        except: return 0.0

    @staticmethod
    def force_fans():
        try:
            subprocess.run("nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: pass

# ==============================================================================
# SECTION 5: VISUAL UTILS
# ==============================================================================

def draw_box(stdscr, y, x, h, w, title, color_pair):
    try:
        stdscr.attron(color_pair)
        stdscr.addch(y, x, curses.ACS_ULCORNER)
        stdscr.addch(y, x + w - 1, curses.ACS_URCORNER)
        stdscr.addch(y + h - 1, x, curses.ACS_LLCORNER)
        stdscr.addch(y + h - 1, x + w - 1, curses.ACS_LRCORNER)
        stdscr.hline(y, x + 1, curses.ACS_HLINE, w - 2)
        stdscr.hline(y + h - 1, x + 1, curses.ACS_HLINE, w - 2)
        stdscr.vline(y + 1, x, curses.ACS_VLINE, h - 2)
        stdscr.vline(y + 1, x + w - 1, curses.ACS_VLINE, h - 2)
        stdscr.addstr(y, x + 2, f" {title} ")
        stdscr.attroff(color_pair)
    except: pass

def draw_bar(val, max_val, width=10):
    pct = max(0.0, min(1.0, val / max_val))
    fill = int(pct * width)
    bar = "|" * fill + " " * (width - fill)
    return f"[{bar}]"

# ==============================================================================
# SECTION 6: BENCHMARK
# ==============================================================================

def cpu_stress(stop_ev, load_counter, throttle_ev):
    import math
    while not stop_ev.is_set():
        throttle_ev.wait()
        size = 50
        A = [[random.random() for _ in range(size)] for _ in range(size)]
        B = [[random.random() for _ in range(size)] for _ in range(size)]
        _ = [[sum(a*b for a,b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]
        with load_counter.get_lock():
            load_counter.value += (size * size)

def gpu_stress(stop_ev, throttle_ev):
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        mod = SourceModule(CUDA_SRC)
        func = mod.get_function("kxt_heavy")
        out = cuda.mem_alloc(4096)
        while not stop_ev.is_set():
            throttle_ev.wait()
            func(out, np.uint32(time.time()), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
    except: time.sleep(0.1)

def run_benchmark(stdscr):
    threading.Thread(target=lambda: [HardwareHAL.force_fans(), time.sleep(15)], daemon=True).start()
    
    stop = mp.Event()
    throttle = mp.Event()
    throttle.set()
    
    cnt = mp.Value('d', 0.0)
    procs = []
    
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_stress, args=(stop, cnt, throttle))
        p.start()
        procs.append(p)
        
    start = time.time()
    throttled_state = False
    
    while time.time() - start < CONFIG['BENCH_TIME']:
        rem = CONFIG['BENCH_TIME'] - (time.time() - start)
        t = HardwareHAL.get_cpu_temp()
        
        if not throttled_state:
            if t >= CONFIG['THROTTLE_TEMP']:
                throttle.clear()
                throttled_state = True
        else:
            if t <= CONFIG['RESUME_TEMP']:
                throttle.set()
                throttled_state = False
                
        try: stdscr.getch()
        except: pass
        
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        stdscr.addstr(0, 0, " KXT GOVERNOR BENCHMARK ".center(w), curses.A_REVERSE)
        draw_box(stdscr, 2, 2, 8, w-4, "CPU STRESS TEST", curses.color_pair(1))
        
        stdscr.addstr(4, 4, f"Time Remaining: {int(rem)}s")
        stdscr.addstr(6, 4, f"CPU Die Temp  : {t:.1f}C  {draw_bar(t, 90, 20)}")
        
        status = "RUNNING (MAX LOAD)"
        col = curses.color_pair(1)
        if throttled_state:
            status = f"THROTTLED (TEMP > {CONFIG['THROTTLE_TEMP']}C) - COOLING"
            col = curses.color_pair(2)
            
        stdscr.addstr(8, 4, f"STATUS: {status}", col)
        
        stdscr.refresh()
        if EXIT_FLAG.is_set(): return
        time.sleep(0.5)
        
    gp = mp.Process(target=gpu_stress, args=(stop, throttle))
    gp.start()
    procs.append(gp)
    
    start = time.time()
    while time.time() - start < CONFIG['BENCH_TIME']:
        rem = CONFIG['BENCH_TIME'] - (time.time() - start)
        c = HardwareHAL.get_cpu_temp()
        g = HardwareHAL.get_gpu_temp()
        
        if not throttled_state:
            if c >= CONFIG['THROTTLE_TEMP']:
                throttle.clear()
                throttled_state = True
        else:
            if c <= CONFIG['RESUME_TEMP']:
                throttle.set()
                throttled_state = False

        try: stdscr.getch()
        except: pass

        stdscr.erase()
        stdscr.addstr(0, 0, " KXT GOVERNOR BENCHMARK ".center(w), curses.A_REVERSE)
        draw_box(stdscr, 2, 2, 12, w-4, "SYSTEM MAX LOAD", curses.color_pair(3))
        
        stdscr.addstr(4, 4, f"Time Remaining: {int(rem)}s")
        stdscr.addstr(6, 4, f"CPU Temp: {c:.1f}C  {draw_bar(c, 90, 20)}")
        stdscr.addstr(7, 4, f"GPU Temp: {g:.1f}C  {draw_bar(g, 90, 20)}")
        
        status = "RUNNING"
        col = curses.color_pair(1)
        if throttled_state:
            status = "THROTTLED - COOLING"
            col = curses.color_pair(2)
        stdscr.addstr(9, 4, f"STATUS: {status}", col)
        
        stdscr.refresh()
        if EXIT_FLAG.is_set(): return
        time.sleep(0.5)
        
    stop.set()
    for p in procs: p.terminate()

# ==============================================================================
# SECTION 7: MINING ENGINE
# ==============================================================================

class StratumClient(threading.Thread):
    def __init__(self, state, job_q, res_q, log_q):
        super().__init__()
        self.state = state; self.job_q = job_q; self.res_q = res_q; self.log_q = log_q
        self.sock = None; self.msg_id = 1; self.buffer = ""; self.daemon = True; self.en1 = None

    def connect(self):
        try:
            self.sock = socket.create_connection((CONFIG['POOL'], CONFIG['PORT']), timeout=10)
            self.state.connected.value = True
            return True
        except:
            self.state.connected.value = False
            return False

    def run(self):
        while not EXIT_FLAG.is_set():
            if not self.sock:
                if self.connect():
                    self.send("mining.subscribe", ["KXT-v37"])
                    self.send("mining.authorize", [CONFIG['USER'], CONFIG['PASS']])
                    self.log_q.put(("NET", "Connected"))
                else:
                    time.sleep(5); continue

            try:
                while not self.res_q.empty():
                    r = self.res_q.get()
                    self.send("mining.submit", [CONFIG['USER'], r['jid'], r['en2'], r['ntime'], r['nonce']])
                    self.log_q.put(("TX", f"Nonce {r['nonce']}"))
                    with self.state.shares.get_lock(): self.state.shares.value += 1

                r, _, _ = select.select([self.sock], [], [], 0.1)
                if r:
                    d = self.sock.recv(4096)
                    if not d: raise Exception("Closed")
                    self.buffer += d.decode()
                    while '\n' in self.buffer:
                        line, self.buffer = self.buffer.split('\n', 1)
                        if line: self.process(json.loads(line))
            except:
                self.sock = None; self.state.connected.value = False; time.sleep(2)

    def send(self, m, p):
        if self.sock:
            try:
                self.sock.sendall((json.dumps({"id": self.msg_id, "method": m, "params": p})+"\n").encode())
                self.msg_id += 1
            except: pass

    def process(self, msg):
        mid = msg.get('id'); method = msg.get('method')
        if mid == 1 and msg.get('result'): self.en1 = msg['result'][1]
        if mid and mid > 2:
            if msg.get('result'):
                with self.state.accepted.get_lock(): self.state.accepted.value += 1
                self.log_q.put(("RX", "Accepted"))
            else:
                with self.state.rejected.get_lock(): self.state.rejected.value += 1
                self.log_q.put(("RX", "Rejected"))
        if method == 'mining.notify':
            p = msg['params']; self.log_q.put(("JOB", f"Blk {p[0][:8]}"))
            if p[8]:
                while not self.job_q.empty(): 
                    try: self.job_q.get_nowait()
                    except: pass
            en1 = self.en1 if self.en1 else "00000000"
            for _ in range(mp.cpu_count() * 2): self.job_q.put((p, en1, 4))

def miner_worker(id, job_q, res_q, stop):
    nonce = id * 5000000
    while not stop.is_set():
        try:
            params, en1, en2sz = job_q.get(timeout=0.1)
            jid, prev, c1, c2, mb, ver, nbits, ntime, clean = params
            if clean: nonce = id * 5000000
            en2 = struct.pack('>I', random.randint(0, 2**32-1)).hex().zfill(en2sz*2)
            coinbase = binascii.unhexlify(c1 + en1 + en2 + c2)
            root = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            for b in mb: root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(b)).digest()).digest()
            header = binascii.unhexlify(ver)[::-1] + binascii.unhexlify(prev)[::-1] + root + binascii.unhexlify(ntime)[::-1] + binascii.unhexlify(nbits)[::-1]
            target = b'\x00\x00'
            for n in range(nonce, nonce + 5000):
                h = header + struct.pack('<I', n)
                if hashlib.sha256(hashlib.sha256(h).digest()).digest().endswith(target):
                    res_q.put({'jid': jid, 'en2': en2, 'ntime': ntime, 'nonce': struct.pack('>I', n).hex()})
                    break
            nonce += 5000
        except queue.Empty: continue
        except: continue

class Proxy(threading.Thread):
    def __init__(self, log_q, state):
        super().__init__(); self.log_q = log_q; self.state = state; self.daemon = True
    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", CONFIG['PROXY_PORT'])); s.listen(50)
            self.log_q.put(("PRX", f"Listen {CONFIG['PROXY_PORT']}"))
            while not EXIT_FLAG.is_set():
                try: c, a = s.accept(); threading.Thread(target=self.h, args=(c,a), daemon=True).start()
                except: pass
        except: pass
    def h(self, c, a):
        try:
            with self.state.proxy_clients.get_lock(): self.state.proxy_clients.value += 1
            p = socket.create_connection((CONFIG['POOL'], 3333)); inputs = [c, p]
            while not EXIT_FLAG.is_set():
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
            c.close()
            p.close()
            with self.state.proxy_clients.get_lock():
                self.state.proxy_clients.value -= 1

# ==============================================================================
# SECTION 8: DASHBOARD
# ==============================================================================

def dashboard(stdscr, state, job_q, res_q, log_q):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)
    stdscr.nodelay(True)
    
    run_benchmark(stdscr)
    if EXIT_FLAG.is_set(): return
    
    stdscr.clear()
    stdscr.addstr(5, 5, "BENCHMARK COMPLETE. STARTING MINER...", curses.A_BOLD)
    stdscr.refresh()
    time.sleep(2)
    
    client = StratumClient(state, job_q, res_q, log_q); client.start()
    Proxy(log_q, state).start()
    
    workers = []; stop = mp.Event()
    for i in range(mp.cpu_count()):
        p = mp.Process(target=miner_worker, args=(i, job_q, res_q, stop))
        p.start(); workers.append(p)
        
    gp = mp.Process(target=gpu_stress, args=(stop, mp.Event())); gp.start(); workers.append(gp)
    
    logs = []
    
    while not EXIT_FLAG.is_set():
        while not log_q.empty():
            logs.append(log_q.get())
            if len(logs) > 20: logs.pop(0)
            
        stdscr.erase(); h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " KXT MINER v37 ".center(w), curses.A_REVERSE)
        
        c = HardwareHAL.get_cpu_temp(); g = HardwareHAL.get_gpu_temp()
        
        draw_box(stdscr, 2, 1, 6, 28, "LOCAL", curses.color_pair(3))
        stdscr.addstr(3, 3, f"CPU: {c:.1f}C {draw_bar(c, 90, 5)}")
        stdscr.addstr(4, 3, f"GPU: {g:.1f}C {draw_bar(g, 90, 5)}")
        
        draw_box(stdscr, 2, 30, 6, 28, "NETWORK", curses.color_pair(3))
        stdscr.addstr(3, 32, f"Link: {'ON' if state.connected.value else 'OFF'}")
        stdscr.addstr(4, 32, f"Shares: {state.shares.value}")
        
        draw_box(stdscr, 2, 59, 6, 28, "STATS", curses.color_pair(3))
        stdscr.addstr(3, 61, f"Acc: {state.accepted.value}")
        stdscr.addstr(4, 61, f"Rej: {state.rejected.value}")
        
        stdscr.hline(8, 0, '-', w)
        for i, (lvl, msg) in enumerate(logs):
            if 9+i >= h-1: break
            col = curses.color_pair(4)
            if lvl in ["ERR", "RX"]: col = curses.color_pair(2)
            if lvl == "JOB": col = curses.color_pair(3)
            stdscr.addstr(9+i, 2, f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}", col)
            
        stdscr.refresh()
        try: 
            if stdscr.getch() == ord('q'): EXIT_FLAG.set()
        except: pass
        time.sleep(0.1)
        
    stop.set()
    for p in workers: p.terminate()

if __name__ == "__main__":
    man = mp.Manager()
    state = man.Namespace()
    state.connected = man.Value('b', False); state.shares = man.Value('i', 0)
    state.accepted = man.Value('i', 0); state.rejected = man.Value('i', 0)
    state.proxy_clients = man.Value('i', 0)
    job_q = man.Queue(); res_q = man.Queue(); log_q = man.Queue()
    try: curses.wrapper(dashboard, state, job_q, res_q, log_q)
    except KeyboardInterrupt: pass
