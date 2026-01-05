#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KXT MINER SUITE (v35) - REFORGED
================================
Architecture: Classic Sensors + Visual Dashboard + Volatile CUDA
Target: solo.stratum.braiins.com:3333
Fixes: CPU Temps, Hashrate Stalling, Visual Layout, Math Errors
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
import math # <--- GLOBAL IMPORT (Fixes NameError)
from datetime import datetime

# ==============================================================================
# SECTION 1: SYSTEM CORE
# ==============================================================================

# Clean Exit Flag
EXIT_FLAG = mp.Event()

def signal_handler(signum, frame):
    EXIT_FLAG.set()
    # No print here to avoid corrupting curses
    
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def boot_check():
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))
    except: pass
    
    # Check for sensors command
    try:
        subprocess.check_call(["which", "sensors"], stdout=subprocess.DEVNULL)
    except:
        print("[WARN] 'sensors' command not found. Installing lm-sensors...")
        try:
            subprocess.check_call(["apt-get", "update"], stdout=subprocess.DEVNULL)
            subprocess.check_call(["apt-get", "install", "-y", "lm-sensors"], stdout=subprocess.DEVNULL)
        except: pass

boot_check()

try: import curses
except: 
    print("[FAIL] Curses library missing. Please install it.")
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
    "BENCH_TIME": 300, # 5 Minutes
}

# ==============================================================================
# SECTION 3: CLASSIC SENSOR LOGIC (Restored from v16)
# ==============================================================================

class HardwareHAL:
    @staticmethod
    def get_cpu_temp():
        """
        Uses the 'sensors' command first, which is the most reliable method
        on Linux for Ryzen/Intel CPUs.
        """
        try:
            out = subprocess.check_output("sensors", shell=True).decode()
            for line in out.splitlines():
                # Priority: Tdie > Package > Core 0
                if "Tdie" in line:
                    return float(line.split('+')[1].split()[0].replace('°C',''))
                if "Package id 0" in line:
                    return float(line.split('+')[1].split()[0].replace('°C',''))
            # Fallback to Core 0
            for line in out.splitlines():
                if "Core 0" in line:
                    return float(line.split('+')[1].split()[0].replace('°C',''))
        except: pass
        
        # Fallback to sysfs if sensors fails
        try:
            zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
            for z in zones:
                try:
                    with open(z, 'r') as f:
                        t = float(f.read().strip()) / 1000.0
                        if t > 20 and t < 110: return t
                except: pass
        except: pass
        return 0.0

    @staticmethod
    def get_gpu_temp():
        try:
            out = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True).decode()
            return float(out.strip())
        except: return 0.0

    @staticmethod
    def force_fans():
        """Forces fans to 100% via Nvidia Settings & Sysfs."""
        try:
            subprocess.run("nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'", 
                           shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: pass
        
        try:
            for pwm in glob.glob("/sys/class/hwmon/hwmon*/pwm*"):
                if "enable" not in pwm:
                    try:
                        with open(pwm, 'w') as f: f.write("255")
                    except: pass
        except: pass

# ==============================================================================
# SECTION 4: CUDA KERNEL (VOLATILE)
# ==============================================================================

CUDA_KXT_SRC = """
extern "C" {
    #include <stdint.h>
    
    __global__ void kxt_burn(uint32_t *output, uint32_t seed) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        volatile uint32_t a = 0x6a09e667 + idx;
        volatile uint32_t b = 0xbb67ae85;
        
        // Heavy Math Loop
        #pragma unroll 128
        for(int i=0; i < 8000; i++) {
            a = (a << 5) | (a >> 27);
            b ^= a;
            a += b + 0xDEADBEEF;
        }
        
        if (a == 0) output[0] = b;
    }
}
"""

# ==============================================================================
# SECTION 5: WORKERS & BENCHMARK
# ==============================================================================

def draw_bar(val, max_val, width=8):
    pct = min(1.0, val / max_val)
    fill = int(pct * width)
    return "[" + "|" * fill + " " * (width - fill) + "]"

def cpu_load_gen(stop_ev, counter):
    # Using global math import
    try:
        while not stop_ev.is_set():
            # Heavy Matrix Math Simulation
            res = 0
            for i in range(2000):
                res += math.sqrt(i) * math.sin(i)
            with counter.get_lock(): counter.value += 2000
    except: pass

def gpu_load_gen(stop_ev):
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
    except: time.sleep(0.5)

def run_kxt_benchmark(stdscr):
    threading.Thread(target=lambda: [HardwareHAL.force_fans(), time.sleep(15)], daemon=True).start()
    
    stop = mp.Event()
    cnt = mp.Value('d', 0.0) # Double precision for large numbers
    procs = []
    
    # Start CPU
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_load_gen, args=(stop, cnt))
        p.start()
        procs.append(p)
        
    start_t = time.time()
    
    # Phase 1: CPU Only
    while time.time() - start_t < CONFIG['BENCH_TIME']:
        rem = CONFIG['BENCH_TIME'] - (time.time() - start_t)
        c_temp = HardwareHAL.get_cpu_temp()
        
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        # Draw Box
        stdscr.attron(curses.color_pair(4))
        stdscr.border()
        stdscr.attroff(curses.color_pair(4))
        
        stdscr.addstr(0, 2, " KXT AUDIT: PHASE 1 (CPU) ", curses.color_pair(4) | curses.A_BOLD)
        stdscr.addstr(2, 2, f"Time Remaining: {int(rem)}s")
        stdscr.addstr(4, 2, f"CPU Temp: {c_temp:.1f}C {draw_bar(c_temp, 90)}")
        stdscr.addstr(5, 2, f"Hashrate: {cnt.value/1000:.0f} kOPs")
        
        stdscr.refresh()
        if EXIT_FLAG.is_set(): break
        time.sleep(0.5)
        
    if EXIT_FLAG.is_set(): return

    # Phase 2: Add GPU
    g = mp.Process(target=gpu_load_gen, args=(stop,))
    g.start()
    procs.append(g)
    
    start_t = time.time()
    while time.time() - start_t < CONFIG['BENCH_TIME']:
        rem = CONFIG['BENCH_TIME'] - (time.time() - start_t)
        c_temp = HardwareHAL.get_cpu_temp()
        g_temp = HardwareHAL.get_gpu_temp()
        
        stdscr.erase()
        stdscr.attron(curses.color_pair(4))
        stdscr.border()
        stdscr.attroff(curses.color_pair(4))
        
        stdscr.addstr(0, 2, " KXT AUDIT: PHASE 2 (FULL LOAD) ", curses.color_pair(4) | curses.A_BOLD)
        stdscr.addstr(2, 2, f"Time Remaining: {int(rem)}s")
        stdscr.addstr(4, 2, f"CPU Temp: {c_temp:.1f}C {draw_bar(c_temp, 90)}")
        stdscr.addstr(5, 2, f"GPU Temp: {g_temp:.1f}C {draw_bar(g_temp, 90)}")
        
        stdscr.refresh()
        if EXIT_FLAG.is_set(): break
        time.sleep(0.5)
        
    stop.set()
    for p in procs: p.terminate()

# ==============================================================================
# SECTION 6: STRATUM & MINING
# ==============================================================================

class StratumClient(threading.Thread):
    def __init__(self, state, job_q, res_q, log_q):
        super().__init__()
        self.state = state; self.job_q = job_q; self.res_q = res_q; self.log_q = log_q
        self.sock = None; self.msg_id = 1; self.buffer = ""
        self.extranonce1 = None; self.extranonce2_size = 4
        self.daemon = True

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
                    self.send("mining.subscribe", ["KXT-v35"])
                    self.send("mining.authorize", [CONFIG['USER'], CONFIG['PASS']])
                    self.log_q.put(("NET", "Connected"))
                else: time.sleep(5); continue

            try:
                # Flush Out
                while not self.res_q.empty():
                    r = self.res_q.get()
                    self.send("mining.submit", [CONFIG['USER'], r['jid'], r['en2'], r['ntime'], r['nonce']])
                    self.log_q.put(("TX", f"Nonce {r['nonce']}"))
                    with self.state.shares.get_lock(): self.state.shares.value += 1

                # Read In
                r, _, _ = select.select([self.sock], [], [], 0.1)
                if r:
                    d = self.sock.recv(4096)
                    if not d: raise Exception()
                    self.buffer += d.decode()
                    while '\n' in self.buffer:
                        line, self.buffer = self.buffer.split('\n', 1)
                        if line: self.process(json.loads(line))
            except:
                self.sock = None
                self.state.connected.value = False
                self.log_q.put(("ERR", "Disconnected"))
                time.sleep(2)

    def send(self, m, p):
        try:
            msg = json.dumps({"id": self.msg_id, "method": m, "params": p}) + "\n"
            self.sock.sendall(msg.encode())
            self.msg_id += 1
        except: self.sock = None

    def process(self, msg):
        mid = msg.get('id')
        if mid == 1 and msg.get('result'):
            self.extranonce1 = msg['result'][1]
            self.extranonce2_size = msg['result'][2]
        
        if msg.get('method') == 'mining.notify':
            p = msg['params']
            self.log_q.put(("JOB", f"Block {p[0][:8]}"))
            if p[8]: # Clean
                while not self.job_q.empty():
                    try: self.job_q.get_nowait()
                    except: pass
            
            en1 = self.extranonce1 if self.extranonce1 else "00000000"
            for _ in range(mp.cpu_count() * 2):
                self.job_q.put((p, en1, self.extranonce2_size))

        if mid and mid > 2:
            if msg.get('result'):
                with self.state.accepted.get_lock(): self.state.accepted.value += 1
                self.log_q.put(("RX", "Share Accepted"))
            else:
                with self.state.rejected.get_lock(): self.state.rejected.value += 1
                self.log_q.put(("RX", "Share Rejected"))

def cpu_worker(id, job_q, res_q, stop):
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
            
            # SHA256d
            en2 = struct.pack('>I', random.randint(0, 2**32-1)).hex().zfill(en2sz*2)
            coinbase = binascii.unhexlify(c1 + en1 + en2 + c2)
            root = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            for b in mb: root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(b)).digest()).digest()
            
            header = (binascii.unhexlify(ver)[::-1] + binascii.unhexlify(prev)[::-1] + root + 
                      binascii.unhexlify(ntime)[::-1] + binascii.unhexlify(nbits)[::-1])
            
            for n in range(nonce, nonce + 5000):
                h = header + struct.pack('<I', n)
                d = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                if d.endswith(b'\x00\x00'):
                    res_q.put({'jid': jid, 'en2': en2, 'ntime': ntime, 'nonce': struct.pack('>I', n).hex()})
                    break
            nonce += 5000
        except: continue

class Proxy(threading.Thread):
    def __init__(self, log_q, state):
        super().__init__(); self.log_q = log_q; self.state = state; self.daemon = True
    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", CONFIG['PROXY_PORT']))
            s.listen(10)
            self.log_q.put(("PRX", f"Active Port {CONFIG['PROXY_PORT']}"))
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
# SECTION 7: CLASSIC UI (4 COLS + BARS)
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
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK) # OK
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)   # ERR
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)  # INFO
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK) # BORDER
    stdscr.nodelay(True)
    
    # 1. Run Benchmark
    run_kxt_benchmark(stdscr)
    if EXIT_FLAG.is_set(): return
    
    # 2. Start Miner
    client = StratumClient(state, job_q, res_q, log_q)
    client.start()
    Proxy(log_q, state).start()
    
    workers = []
    stop_workers = mp.Event()
    for i in range(mp.cpu_count()):
        p = mp.Process(target=cpu_worker, args=(i, job_q, res_q, stop_workers))
        p.start()
        workers.append(p)
    
    # GPU Dummy Heat
    def gpu_heat():
        import time; 
        while not stop_workers.is_set(): time.sleep(1)
    workers.append(mp.Process(target=gpu_heat)); workers[-1].start()
    
    logs = []
    
    while not EXIT_FLAG.is_set():
        while not log_q.empty():
            logs.append(log_q.get())
            if len(logs) > 30: logs.pop(0)
            
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        # Header
        stdscr.addstr(0, 0, " KXT MINER v35 REFORGED ".center(w), curses.color_pair(4) | curses.A_REVERSE)
        
        c_temp = HardwareHAL.get_cpu_temp()
        g_temp = HardwareHAL.get_gpu_temp()
        
        # COL 1: LOCAL
        draw_box(stdscr, 2, 2, 6, 28, "SYSTEM", curses.color_pair(3))
        stdscr.addstr(3, 4, f"CPU: {c_temp:.1f}C {draw_bar(c_temp, 90)}")
        stdscr.addstr(4, 4, f"GPU: {g_temp:.1f}C {draw_bar(g_temp, 90)}")
        
        # COL 2: NETWORK
        draw_box(stdscr, 2, 32, 6, 28, "NETWORK", curses.color_pair(3))
        status = "ONLINE" if state.connected.value else "OFFLINE"
        stdscr.addstr(3, 34, f"Link: {status}", curses.color_pair(1 if state.connected.value else 2))
        stdscr.addstr(4, 34, f"Shares: {state.shares.value}")
        
        # COL 3: STATS
        draw_box(stdscr, 2, 62, 6, 28, "STATS", curses.color_pair(3))
        stdscr.addstr(3, 64, f"Acc: {state.accepted.value}", curses.color_pair(1))
        stdscr.addstr(4, 64, f"Rej: {state.rejected.value}", curses.color_pair(2))
        
        # COL 4: PROXY
        draw_box(stdscr, 2, 92, 6, 25, "PROXY", curses.color_pair(3))
        stdscr.addstr(3, 94, f"Clients: {state.proxy_clients.value}")
        stdscr.addstr(4, 94, f"Port: {CONFIG['PROXY_PORT']}")
        
        # LOGS
        stdscr.hline(8, 0, '-', w)
        for i, (lvl, msg) in enumerate(logs):
            if 9+i >= h-1: break
            col = curses.color_pair(4)
            if lvl in ["ERR", "RX"]: col = curses.color_pair(2)
            if lvl == "JOB": col = curses.color_pair(3)
            if lvl == "NET": col = curses.color_pair(1)
            
            ts = datetime.now().strftime("%H:%M:%S")
            stdscr.addstr(9+i, 2, f"[{ts}] [{lvl}] {msg}", col)
            
        stdscr.refresh()
        
        try: 
            if stdscr.getch() == ord('q'): EXIT_FLAG.set()
        except: pass
        time.sleep(0.1)
        
    stop_workers.set()
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
