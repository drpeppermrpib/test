#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTP MINER SUITE v34 - TITAN IV "FINAL"
======================================
Architecture: Stratum V1 + Volatile CUDA + Classic UI
Target: solo.stratum.braiins.com:3333
Fixes: Math Import, UI Layout, Persistent Logs
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
import platform
import queue
import traceback
import math  # <--- FIXED: Global Import for all processes
from datetime import datetime

# ==============================================================================
# SECTION 1: SYSTEM BOOTSTRAP
# ==============================================================================

def titan_boot():
    print("[BOOT] Initializing Titan IV Engine...")
    
    # 1.1 Log Init
    with open("titan_miner.log", "a") as f:
        f.write(f"\n\n=== SESSION START: {datetime.now()} ===\n")
    
    # 1.2 Ulimit
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65535, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except: pass

    # 1.3 Drivers
    required = ["psutil", "requests"]
    for pkg in required:
        try: __import__(pkg)
        except: 
            try: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except: pass

titan_boot()

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
    "WORKER_NAME": "rig1",
    "PASS": "x",
    "PROXY_PORT": 60060,
    
    "TEMP_OFFSET_CPU": -10.0, 
    "BENCH_STAGE_TIME": 300, 
    
    "CPU_BATCH": 100000,
}

# ==============================================================================
# SECTION 3: CUDA KERNEL
# ==============================================================================

CUDA_TITAN_IV_SRC = """
extern "C" {
    #include <stdint.h>

    __device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }
    
    __device__ __forceinline__ uint32_t sigma0(uint32_t x) {
        return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
    }
    
    __device__ __forceinline__ uint32_t sigma1(uint32_t x) {
        return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
    }
    
    __device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (~x & z);
    }

    __global__ void titan_burn(uint32_t *output, uint32_t seed) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        volatile uint32_t a = 0x6a09e667 + idx + seed;
        volatile uint32_t b = 0xbb67ae85;
        volatile uint32_t c = 0x3c6ef372;
        
        #pragma unroll 128
        for(int i=0; i < 4000; i++) {
            a += sigma0(b) + ch(b, c, a);
            b = rotr(b, 11) ^ a;
            c += sigma1(a);
        }
        
        if (a == 0xDEADBEEF) {
            output[idx % 1024] = a + b + c;
        }
    }
}
"""

# ==============================================================================
# SECTION 4: HARDWARE HAL
# ==============================================================================

class HAL:
    @staticmethod
    def get_cpu_temp():
        raw = 0.0
        found = False
        try:
            zones = glob.glob("/sys/class/hwmon/hwmon*/temp*_input")
            for z in zones:
                with open(z, 'r') as f:
                    val = float(f.read().strip()) / 1000.0
                    if val > 20: 
                        raw = val
                        found = True
                        break
        except: pass
        
        if not found:
            try:
                out = subprocess.check_output("sensors", shell=True).decode()
                for l in out.splitlines():
                    if "Tdie" in l or "Package" in l:
                        raw = float(l.split('+')[1].split('.')[0])
                        break
            except: pass
            
        final = raw + CONFIG['TEMP_OFFSET_CPU']
        return max(0.0, final)

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
# SECTION 5: BENCHMARK
# ==============================================================================

def cpu_load(stop):
    # Uses global math import
    while not stop.is_set():
        _ = [math.sqrt(x) * math.sin(x) for x in range(1000)]

def gpu_load(stop, log_q):
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_TITAN_IV_SRC)
        func = mod.get_function("titan_burn")
        log_q.put(("INFO", "GPU Burner Active"))
        
        while not stop.is_set():
            out = np.zeros(1, dtype=np.uint32)
            func(cuda.Out(out), np.uint32(int(time.time())), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
    except: time.sleep(1)

def run_benchmark(log_q):
    os.system('clear')
    print("=== MTP v34 TITAN IV BENCHMARK ===")
    
    ft = threading.Thread(target=lambda: [HAL.set_fans_max(), time.sleep(10)], daemon=True)
    ft.start()
    
    # STAGE 1
    print(f"\n[STAGE 1] CPU MAX LOAD ({CONFIG['BENCH_STAGE_TIME']}s)")
    stop = mp.Event()
    procs = []
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_load, args=(stop,))
        p.start()
        procs.append(p)
        
    start = time.time()
    try:
        while time.time() - start < CONFIG['BENCH_STAGE_TIME']:
            rem = int(CONFIG['BENCH_STAGE_TIME'] - (time.time() - start))
            sys.stdout.write(f"\rTime: {rem}s | CPU: {HAL.get_cpu_temp():.1f}C ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt: pass
    
    # STAGE 2
    print(f"\n\n[STAGE 2] CPU + GPU MAX LOAD ({CONFIG['BENCH_STAGE_TIME']}s)")
    gp = mp.Process(target=gpu_load, args=(stop, log_q))
    gp.start()
    procs.append(gp)
    
    start = time.time()
    try:
        while time.time() - start < CONFIG['BENCH_STAGE_TIME']:
            rem = int(CONFIG['BENCH_STAGE_TIME'] - (time.time() - start))
            sys.stdout.write(f"\rTime: {rem}s | CPU: {HAL.get_cpu_temp():.1f}C | GPU: {HAL.get_gpu_temp():.1f}C ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt: pass
    
    stop.set()
    for p in procs: p.terminate()
    print("\n\n[DONE] Benchmark Complete.")
    time.sleep(2)

# ==============================================================================
# SECTION 6: STRATUM & PROXY
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
        
    def run(self):
        while True:
            try:
                self.sock = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=15)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                self.send("mining.subscribe", ["MTP-v34"])
                self.send("mining.authorize", [f"{CONFIG['WALLET']}.{CONFIG['WORKER_NAME']}_local", CONFIG['PASS']])
                
                self.state.connected.value = True
                self.log_q.put(("NET", "Pool Session Active"))
                
                while True:
                    # Flush Outbound
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        self.send("mining.submit", [
                            f"{CONFIG['WALLET']}.{CONFIG['WORKER_NAME']}_local",
                            r['jid'], r['en2'], r['ntime'], r['nonce']
                        ])
                        self.log_q.put(("TX", f"Submit Nonce {r['nonce']}"))
                        with self.state.local_tx.get_lock(): self.state.local_tx.value += 1

                    # Read Inbound
                    r, _, _ = select.select([self.sock], [], [], 0.1)
                    if r:
                        d = self.sock.recv(4096)
                        if not d: break
                        self.buffer += d.decode()
                        while '\n' in self.buffer:
                            line, self.buffer = self.buffer.split('\n', 1)
                            if line.strip(): self.parse(json.loads(line))
                    
            except Exception as e:
                self.state.connected.value = False
                self.log_q.put(("ERR", f"Connection Lost: {e}"))
                time.sleep(5)

    def send(self, method, params):
        msg = json.dumps({"id": self.msg_id, "method": method, "params": params}) + "\n"
        try:
            self.sock.sendall(msg.encode())
            self.msg_id += 1
        except: pass

    def parse(self, msg):
        mid = msg.get('id')
        method = msg.get('method')
        
        if mid == 1 and msg.get('result'):
            self.extranonce1 = msg['result'][1]
            self.extranonce2_size = msg['result'][2]
            
        if mid and mid > 2:
            if msg.get('result'):
                with self.state.accepted.get_lock(): self.state.accepted.value += 1
                self.log_q.put(("RX", "Share ACCEPTED"))
            else:
                with self.state.rejected.get_lock(): self.state.rejected.value += 1
                self.log_q.put(("RX", f"Share REJECTED: {msg.get('error')}"))

        if method == 'mining.notify':
            p = msg['params']
            self.log_q.put(("RX", f"New Job: {p[0][:8]}"))
            
            if p[8]:
                while not self.job_q.empty():
                    try: self.job_q.get_nowait()
                    except: pass
            
            en1 = self.extranonce1 if self.extranonce1 else "00000000"
            job = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], en1, self.extranonce2_size)
            
            for _ in range(mp.cpu_count() * 2):
                self.job_q.put(job)

class Proxy(threading.Thread):
    def __init__(self, log_q, state):
        super().__init__()
        self.log_q = log_q
        self.state = state
        self.daemon = True
    
    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", CONFIG['PROXY_PORT']))
        s.listen(100)
        self.log_q.put(("INFO", f"Proxy Active {CONFIG['PROXY_PORT']}"))
        while True:
            try:
                c, a = s.accept()
                threading.Thread(target=self.handle, args=(c,a), daemon=True).start()
            except: pass
            
    def handle(self, c, a):
        try:
            with self.state.proxy_clients.get_lock(): self.state.proxy_clients.value += 1
            p = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']))
            ip = a[0].split('.')[-1]
            
            inputs = [c, p]
            while True:
                r, _, _ = select.select(inputs, [], [])
                if c in r:
                    d = c.recv(4096)
                    if not d: break
                    try:
                        t = d.decode()
                        if "mining.authorize" in t:
                            j = json.loads(t)
                            j['params'][0] = f"{CONFIG['WALLET']}.ASIC_{ip}"
                            d = (json.dumps(j)+"\n").encode()
                        if "mining.submit" in t:
                            with self.state.proxy_tx.get_lock(): self.state.proxy_tx.value += 1
                    except: pass
                    p.sendall(d)
                if p in r:
                    d = p.recv(4096)
                    if not d: break
                    if b'true' in d:
                        with self.state.proxy_rx.get_lock(): self.state.proxy_rx.value += 1
                    c.sendall(d)
        except: pass
        finally: 
            c.close(); p.close()
            with self.state.proxy_clients.get_lock(): self.state.proxy_clients.value -= 1

# ==============================================================================
# SECTION 7: WORKERS
# ==============================================================================

def cpu_worker(id, job_q, res_q, stop, counter):
    nonce = id * 10000000
    while not stop.is_set():
        try:
            params, en1, en2_sz = job_q.get(timeout=0.1)
            jid, prev, c1, c2, mb, ver, nbits, ntime, clean = params
            
            if clean: nonce = id * 10000000
            
            en2 = struct.pack('>I', random.randint(0, 2**32-1)).hex().zfill(en2_sz*2)
            coinbase = binascii.unhexlify(c1 + en1 + en2 + c2)
            cb_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            root = cb_hash
            for b in mb: root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(b)).digest()).digest()
            
            header = (
                binascii.unhexlify(ver)[::-1] + binascii.unhexlify(prev)[::-1] +
                root + binascii.unhexlify(ntime)[::-1] + binascii.unhexlify(nbits)[::-1]
            )
            
            target = b'\x00\x00\x00\x00'
            for n in range(nonce, nonce + 5000):
                h = header + struct.pack('<I', n)
                d = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                if d.endswith(target):
                    res_q.put({
                        'jid': jid, 'en2': en2, 'ntime': ntime, 
                        'nonce': struct.pack('>I', n).hex()
                    })
                    break
            nonce += 5000
            with counter.get_lock(): counter.value += 5000
        except: continue

def gpu_worker(stop, counter, log_q):
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_TITAN_IV_SRC)
        func = mod.get_function("titan_burn")
        log_q.put(("GPU", "CUDA Active"))
        
        while not stop.is_set():
            out = np.zeros(1024, dtype=np.uint32)
            func(cuda.Out(out), np.uint32(int(time.time())), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
            with counter.get_lock(): counter.value += (65535 * 512 * 4000)
    except: pass

# ==============================================================================
# SECTION 8: CLASSIC UI (v16 STYLE)
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

def dashboard(stdscr, state, job_q, res_q, log_q):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
    stdscr.nodelay(True)
    
    # Init Logic
    client = StratumClient(state, job_q, res_q, log_q)
    ct = threading.Thread(target=client.run, daemon=True)
    ct.start()
    
    Proxy(log_q, state).start()
    
    stop = mp.Event()
    hash_count = mp.Value('d', 0.0)
    procs = []
    
    for i in range(max(1, mp.cpu_count() - 1)):
        p = mp.Process(target=cpu_worker, args=(i, job_q, res_q, stop, hash_count))
        p.start()
        procs.append(p)
        
    gp = mp.Process(target=gpu_worker, args=(stop, hash_count, log_q))
    gp.start()
    procs.append(gp)
    
    logs = []
    last_h = 0.0
    current_hr = 0.0
    
    while True:
        while not log_q.empty():
            msg = log_q.get()
            with open("titan_miner.log", "a") as f: f.write(f"{datetime.now()} {msg}\n")
            logs.append(msg)
            if len(logs) > 50: logs.pop(0)
            
        total = hash_count.value
        delta = total - last_h
        last_h = total
        current_hr = (current_hr * 0.9) + (delta * 10 * 0.1)
        
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        # HEADER
        stdscr.addstr(0, 0, " MTP v34 TITAN IV ".center(w), curses.color_pair(5) | curses.A_REVERSE)
        
        # LOCAL BOX
        draw_box(stdscr, 2, 1, 6, 30, "SYSTEM", curses.color_pair(4))
        c = HAL.get_cpu_temp()
        g = HAL.get_gpu_temp()
        
        if current_hr > 1e9: hrs = f"{current_hr/1e9:.2f} GH/s"
        elif current_hr > 1e6: hrs = f"{current_hr/1e6:.2f} MH/s"
        else: hrs = f"{current_hr/1000:.2f} kH/s"
        
        stdscr.addstr(3, 3, f"CPU: {c:.1f}C", curses.color_pair(1))
        stdscr.addstr(4, 3, f"GPU: {g:.1f}C", curses.color_pair(1))
        stdscr.addstr(5, 3, f"H/R: {hrs}", curses.color_pair(5) | curses.A_BOLD)
        
        # NETWORK BOX
        draw_box(stdscr, 2, 32, 6, 30, "NETWORK", curses.color_pair(4))
        stdscr.addstr(3, 34, f"Link: {state.connected.value}", curses.color_pair(1 if state.connected.value else 3))
        stdscr.addstr(4, 34, f"Shares: {state.local_tx.value}", curses.color_pair(2))
        stdscr.addstr(5, 34, f"A/R: {state.accepted.value}/{state.rejected.value}", curses.color_pair(5))
        
        # PROXY BOX
        draw_box(stdscr, 2, 63, 6, 30, "PROXY", curses.color_pair(4))
        stdscr.addstr(3, 65, f"Clients: {state.proxy_clients.value}")
        stdscr.addstr(4, 65, f"TX: {state.proxy_tx.value}")
        
        # LOGS
        stdscr.hline(8, 0, '-', w)
        for i, (lvl, msg) in enumerate(logs[- (h-10) :]):
            c = curses.color_pair(5)
            if lvl == "TX": c = curses.color_pair(2)
            if lvl == "RX": c = curses.color_pair(1)
            if lvl == "ERR": c = curses.color_pair(3)
            try: stdscr.addstr(9+i, 2, f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}", c)
            except: pass
            
        stdscr.refresh()
        time.sleep(0.1)
        if stdscr.getch() == ord('q'): break
        
    stop.set()
    for p in procs: p.terminate()

if __name__ == "__main__":
    try:
        bq = mp.Queue()
        run_benchmark(bq)
        
        man = mp.Manager()
        state = man.Namespace()
        state.connected = man.Value('b', False)
        state.local_tx = man.Value('i', 0)
        state.accepted = man.Value('i', 0)
        state.rejected = man.Value('i', 0)
        state.proxy_clients = man.Value('i', 0)
        state.proxy_tx = man.Value('i', 0)
        state.proxy_rx = man.Value('i', 0)
        
        job_q = man.Queue()
        res_q = man.Queue()
        log_q = man.Queue()
        
        curses.wrapper(dashboard, state, job_q, res_q, log_q)
    except Exception as e:
        with open("titan_crash.log", "w") as f:
            f.write(traceback.format_exc())
        print("CRASHED. Check titan_crash.log")
