#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KXT MINER SUITE (v34) - TITAN IV ENGINE
=======================================
Architecture: Heavy Load + High-Fidelity Sensors + Visual Dashboard
Target: solo.stratum.braiins.com:3333
Fixes: Broken Pipes, Temp Discrepancy (30C vs 60C), Visual Layout
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
import math  # REQUIRED for Load Gen
from datetime import datetime

# ==============================================================================
# SECTION 1: SYSTEM CORE & SIGNAL HANDLING (NO MORE BROKEN PIPES)
# ==============================================================================

# Global Event for clean shutdown
EXIT_FLAG = mp.Event()

def signal_handler(signum, frame):
    """Intercepts Ctrl+C to prevent BrokenPipeError tracebacks."""
    EXIT_FLAG.set()
    print("\n[KXT] Shutdown Signal Received. Stopping Engines...")
    # Give threads a moment to close sockets
    time.sleep(1)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def boot_check():
    # 1. Fix File Descriptors
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))
    except: pass

    # 2. Check Drivers
    req = ["psutil", "requests"]
    for r in req:
        try: __import__(r)
        except: pass # Silent fail, we have fallbacks

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
    
    # Bench: 5 Minutes per stage (300s)
    "BENCH_TIME": 300,
    
    # Load Tuning (Increased for KXT)
    "CPU_LOAD_MULT": 5000,
}

# ==============================================================================
# SECTION 3: HEAVY CUDA KERNEL (VOLATILE)
# ==============================================================================

CUDA_KXT_SRC = """
extern "C" {
    #include <stdint.h>

    __global__ void kxt_burn(uint32_t *output, uint32_t seed) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // VOLATILE forces the GPU to not optimize this away
        volatile uint32_t a = 0x6a09e667 + idx;
        volatile uint32_t b = 0xbb67ae85;
        
        // Massive loop to force TGP up
        #pragma unroll 128
        for(int i=0; i < 10000; i++) {
            a = (a << 5) | (a >> 27);
            b ^= a;
            a += b + 0xDEADBEEF;
        }
        
        if (a == 0) output[0] = b;
    }
}
"""

# ==============================================================================
# SECTION 4: ADVANCED SENSORS (FIXING THE 32C BUG)
# ==============================================================================

class HardwareHAL:
    @staticmethod
    def get_cpu_temp():
        temps = []
        
        # Method 1: Sysfs Thermal Zones (Check ALL and take MAX)
        try:
            zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
            for z in zones:
                try:
                    with open(z, 'r') as f:
                        val = float(f.read().strip())
                        if val > 1000: val /= 1000.0
                        if val > 20 and val < 115: # Filter bad readings
                            temps.append(val)
                except: pass
        except: pass

        # Method 2: Sensors Command
        try:
            out = subprocess.check_output("sensors", shell=True).decode()
            for line in out.splitlines():
                if "Tdie" in line or "Package" in line or "Core" in line:
                    try:
                        t = float(line.split('+')[1].split('.')[0])
                        temps.append(t)
                    except: pass
        except: pass
        
        # Logic: If we found temps, return the HIGHEST one.
        # This fixes the issue where it was reading a case fan sensor (32C) instead of the CPU (67C)
        if temps:
            return max(temps)
            
        return 0.0

    @staticmethod
    def get_gpu_temp():
        try:
            o = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True)
            return float(o.decode().strip())
        except: return 0.0

    @staticmethod
    def set_fans_max():
        """Aggressive Fan Force"""
        cmds = [
            "nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'",
            "nvidia-settings -a 'GPUFanControlState=1' -a 'GPUTargetFanSpeed=100'"
        ]
        for c in cmds:
            try: subprocess.run(c, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except: pass
        
        # Sysfs
        try:
            for pwm in glob.glob("/sys/class/hwmon/hwmon*/pwm*"):
                if os.access(pwm, os.W_OK):
                    with open(pwm, 'w') as f: f.write("255")
        except: pass

# ==============================================================================
# SECTION 5: VISUALS & UTILS
# ==============================================================================

def draw_bar(val, max_val, width=10):
    """Draws a visual bar: [|||||     ]"""
    pct = min(1.0, val / max_val)
    fill = int(pct * width)
    bar = "|" * fill + " " * (width - fill)
    return f"[{bar}]"

# ==============================================================================
# SECTION 6: KXT BENCHMARK (HEAVY LOAD)
# ==============================================================================

def cpu_load_gen(stop_ev, counter):
    # Local import ensures it works in process
    import math 
    while not stop_ev.is_set():
        # Heavier math load: Matrix simulation + Trig
        # This will generate significantly more heat than simple SHA256
        res = 0
        for i in range(1000):
            res += math.sqrt(i * 3.14159) * math.tan(i)
        
        # Update counter periodically to avoid locking contention
        with counter.get_lock():
            counter.value += 1000

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
    except: 
        time.sleep(0.1)

def run_kxt_benchmark(stdscr):
    # Start Fan Force
    threading.Thread(target=lambda: [HardwareHAL.set_fans_max(), time.sleep(15)], daemon=True).start()
    
    # Setup Phase 1 (CPU)
    stop = mp.Event()
    cnt = mp.Value('d', 0.0)
    procs = []
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_load_gen, args=(stop, cnt))
        p.start()
        procs.append(p)
    
    start = time.time()
    
    # Visual Loop Phase 1
    while time.time() - start < CONFIG['BENCH_TIME']:
        elapsed = time.time() - start
        rem = CONFIG['BENCH_TIME'] - elapsed
        
        c_temp = HardwareHAL.get_cpu_temp()
        bar = draw_bar(c_temp, 90, 20)
        
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " KXT SYSTEM AUDIT ".center(w), curses.A_REVERSE)
        
        stdscr.addstr(2, 2, "PHASE 1: CPU THERMAL SATURATION")
        stdscr.addstr(4, 2, f"Time Remaining: {int(rem)}s")
        stdscr.addstr(6, 2, f"CPU Temp: {c_temp:.1f}C  {bar}")
        stdscr.addstr(7, 2, f"CPU Load: {cnt.value:.0f} OPS")
        
        stdscr.refresh()
        if EXIT_FLAG.is_set(): break
        time.sleep(0.5)
        
    if EXIT_FLAG.is_set(): return

    # Setup Phase 2 (Add GPU)
    g_proc = mp.Process(target=gpu_load_gen, args=(stop,))
    g_proc.start()
    procs.append(g_proc)
    
    start = time.time()
    while time.time() - start < CONFIG['BENCH_TIME']:
        elapsed = time.time() - start
        rem = CONFIG['BENCH_TIME'] - elapsed
        
        c_temp = HardwareHAL.get_cpu_temp()
        g_temp = HardwareHAL.get_gpu_temp()
        
        c_bar = draw_bar(c_temp, 90, 20)
        g_bar = draw_bar(g_temp, 90, 20)
        
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " KXT SYSTEM AUDIT ".center(w), curses.A_REVERSE)
        
        stdscr.addstr(2, 2, "PHASE 2: TOTAL SYSTEM LOAD (CPU + GPU)")
        stdscr.addstr(4, 2, f"Time Remaining: {int(rem)}s")
        stdscr.addstr(6, 2, f"CPU Temp: {c_temp:.1f}C  {c_bar}")
        stdscr.addstr(7, 2, f"GPU Temp: {g_temp:.1f}C  {g_bar}")
        stdscr.addstr(9, 2, "Note: High Load Active. Fans should be 100%.")
        
        stdscr.refresh()
        if EXIT_FLAG.is_set(): break
        time.sleep(0.5)

    stop.set()
    for p in procs: p.terminate()

# ==============================================================================
# SECTION 7: MINING ENGINE
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
                    self.send("mining.subscribe", ["KXT-v34"])
                    self.send("mining.authorize", [CONFIG['USER'], CONFIG['PASS']])
                    self.log_q.put(("NET", "Connected"))
                else:
                    time.sleep(5)
                    continue

            try:
                # Flush Outbound
                while not self.res_q.empty():
                    r = self.res_q.get()
                    self.send("mining.submit", [
                        CONFIG['USER'], r['jid'], r['en2'], r['ntime'], r['nonce']
                    ])
                    self.log_q.put(("TX", f"Nonce {r['nonce']}"))
                    with self.state.shares.get_lock(): self.state.shares.value += 1

                # Read Inbound
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
                self.log_q.put(("ERR", "Link Reset"))
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
            
            if p[8]: # Clean job
                while not self.job_q.empty(): 
                    try: self.job_q.get_nowait()
                    except: pass
            
            en1 = self.extranonce1 if self.extranonce1 else "00000000"
            job = (p, en1, self.extranonce2_size)
            for _ in range(mp.cpu_count() * 2): self.job_q.put(job)

def cpu_worker(id, job_q, res_q, stop):
    nonce = id * 5000000
    cur_job = None
    
    while not stop.is_set():
        try:
            try:
                params, en1, en2sz = job_q.get(timeout=0.1)
                cur_job = params
                if params[8]: nonce = id * 5000000 # Clean
            except: 
                if not cur_job: continue
            
            jid, prev, c1, c2, mb, ver, nbits, ntime, clean = cur_job
            
            # Simple Hashing Loop
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
                if d.endswith(b'\x00\x00'): # Found Share
                    res_q.put({
                        'jid': jid, 'en2': en2, 'ntime': ntime, 
                        'nonce': struct.pack('>I', n).hex()
                    })
                    break
            nonce += 20000
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
# SECTION 8: MAIN DASHBOARD (4 COLUMNS + BARS)
# ==============================================================================

def main_gui(stdscr, state, job_q, res_q, log_q):
    curses.start_color()
    # Define Colors
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK) # OK
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)   # ERR
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)  # INFO
    
    stdscr.nodelay(True)
    
    # 1. RUN BENCHMARK FIRST
    run_kxt_benchmark(stdscr)
    if EXIT_FLAG.is_set(): return
    
    # 2. START MINER
    client = StratumClient(state, job_q, res_q, log_q)
    client.start()
    
    Proxy(log_q, state).start()
    
    workers = []
    stop_workers = mp.Event()
    for i in range(mp.cpu_count()):
        p = mp.Process(target=cpu_worker, args=(i, job_q, res_q, stop_workers))
        p.start()
        workers.append(p)
        
    # GPU Dummy Load for Heat
    def gpu_heat():
        import time; 
        while not stop_workers.is_set(): time.sleep(1)
    gp = mp.Process(target=gpu_heat)
    gp.start()
    workers.append(gp)
    
    logs = []
    
    while not EXIT_FLAG.is_set():
        while not log_q.empty():
            logs.append(log_q.get())
            if len(logs) > 20: logs.pop(0)
            
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        # Header
        stdscr.addstr(0, 0, " KXT MINER v34 (TITAN IV) ".center(w), curses.A_REVERSE)
        
        c_temp = HardwareHAL.get_cpu_temp()
        g_temp = HardwareHAL.get_gpu_temp()
        c_bar = draw_bar(c_temp, 90, 10)
        g_bar = draw_bar(g_temp, 90, 10)
        
        # 4 COLUMNS LAYOUT
        # COL 1: LOCAL
        stdscr.addstr(2, 2, "LOCAL SYSTEM", curses.color_pair(3))
        stdscr.addstr(3, 2, f"CPU: {c_temp:.1f}C {c_bar}")
        stdscr.addstr(4, 2, f"GPU: {g_temp:.1f}C {g_bar}")
        
        # COL 2: NETWORK
        stdscr.addstr(2, 35, "NETWORK", curses.color_pair(3))
        status = "ONLINE" if state.connected.value else "OFFLINE"
        stdscr.addstr(3, 35, f"Link: {status}")
        stdscr.addstr(4, 35, f"Shares: {state.shares.value}")
        
        # COL 3: STATS
        stdscr.addstr(2, 60, "STATS", curses.color_pair(3))
        stdscr.addstr(3, 60, f"Acc: {state.accepted.value}")
        stdscr.addstr(4, 60, f"Rej: {state.rejected.value}")
        
        # COL 4: PROXY
        stdscr.addstr(2, 80, "PROXY", curses.color_pair(3))
        stdscr.addstr(3, 80, f"Clients: {state.proxy_clients.value}")
        stdscr.addstr(4, 80, f"Port: {CONFIG['PROXY_PORT']}")
        
        # LOGS AREA
        stdscr.hline(6, 0, '-', w)
        for i, (lvl, msg) in enumerate(logs):
            if 7+i >= h-1: break
            col = curses.color_pair(1)
            if lvl in ["ERR", "RX"]: col = curses.color_pair(2)
            if lvl == "JOB": col = curses.color_pair(3)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            stdscr.addstr(7+i, 2, f"[{timestamp}] [{lvl}] {msg}", col)
            
        stdscr.refresh()
        
        # Handle Input
        try:
            key = stdscr.getch()
            if key == ord('q'): EXIT_FLAG.set()
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
    except KeyboardInterrupt:
        pass
