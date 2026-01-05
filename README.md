#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

# FIX: Large integer string conversion limit (From Screenshot)
try: sys.set_int_max_str_digits(0)
except: pass

import socket
import json
import time
import threading
import multiprocessing as mp
import curses
import binascii
import struct
import hashlib
import subprocess
import os
import queue
import select
import urllib.request
import signal
import atexit
import random
import resource
from datetime import datetime

# ==============================================================================
# SECTION 1: AUTO-DEPENDENCY & ZOMBIE KILLER
# ==============================================================================

# Auto-Dependency Check (Matches Screenshot lines 23-27)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def kill_stale_processes():
    """Aggressively cleans up previous instances/zombies."""
    current_pid = os.getpid()
    try:
        # Try using pkill to remove other instances of this script
        subprocess.run(f"pkill -f kxt.py | grep -v {current_pid}", shell=True, stderr=subprocess.DEVNULL)
    except: pass
    
    # Clean up standard zombies
    try:
        os.waitpid(-1, os.WNOHANG)
    except: pass

kill_stale_processes()

# ==============================================================================
# SECTION 2: CONFIGURATION & GLOBAL STATE
# ==============================================================================

CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.rig1",
    "PASS": "x",
    "PROXY_PORT": 60060,
    
    # BENCHMARK SETTINGS
    "BENCH_DURATION": 60,   # 60 Seconds
    
    # THERMAL GOVERNOR
    "TEMP_LIMIT": 79.0,     # Pause Load
    "TEMP_RESUME": 75.0,    # Resume Load
}

# Shared Exit Flag
EXIT_FLAG = mp.Event()

# ==============================================================================
# SECTION 3: SHUTDOWN HANDLER (THE FIX)
# ==============================================================================

def final_shutdown():
    """
    Ensures the window stays open and processes die.
    """
    # 1. Kill everything
    EXIT_FLAG.set()
    try: curses.endwin()
    except: pass
    
    print("\n" + "="*60)
    print(" KXT MINER SHUTDOWN SEQUENCE ".center(60))
    print("="*60)
    print(" [INFO] Stopping Worker Processes...")
    
    if HAS_PSUTIL:
        try:
            parent = psutil.Process(os.getpid())
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            _, alive = psutil.wait_procs(children, timeout=3)
            for p in alive:
                p.kill()
        except: pass
        
    print(" [INFO] Cleanup Complete.")
    print("-" * 60)
    # THE WAIT FIX:
    input(" >> Press ENTER to close this window << ")
    sys.exit(0)

def signal_handler(signum, frame):
    final_shutdown()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(kill_stale_processes)

# ==============================================================================
# SECTION 4: CUDA ENGINE (HEAVY LOAD)
# ==============================================================================

CUDA_SOURCE_CODE = """
extern "C" {
    #include <stdint.h>
    
    // Optimized SHA256 simulation for heat generation
    __global__ void kxt_heavy_load(uint32_t *output, uint32_t seed) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        volatile uint32_t a = 0x6a09e667 + idx + seed;
        volatile uint32_t b = 0xbb67ae85;
        
        // Unrolled loop for maximum ALU saturation
        #pragma unroll 128
        for(int i=0; i < 50000; i++) {
            a = (a << 5) | (a >> 27);
            b ^= a;
            a += b + 0xDEADBEEF;
            if (i % 2000 == 0) output[idx % 1024] = a;
        }
    }
}
"""

# ==============================================================================
# SECTION 5: HARDWARE HAL (SIMPLE SENSORS)
# ==============================================================================

class HardwareHAL:
    @staticmethod
    def get_cpu_temp():
        # Standard Sensor Logic (No complex filters as requested)
        if HAS_PSUTIL:
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current > 0: return entry.current
            except: pass
            
        # Fallback to sysfs
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read()) / 1000.0
        except: pass
        
        return 0.0

    @staticmethod
    def get_gpu_temp():
        try:
            cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader"
            return float(subprocess.check_output(cmd, shell=True).decode().strip())
        except: return 0.0

    @staticmethod
    def set_fans_100():
        try:
            subprocess.run("nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'", shell=True, stderr=subprocess.DEVNULL)
        except: pass

# ==============================================================================
# SECTION 6: BENCHMARK ENGINE (TEXT MODE)
# ==============================================================================

def cpu_load_task(stop_event, throttle_event, cnt):
    import math
    while not stop_event.is_set():
        throttle_event.wait()
        # Heavy Matrix Calculation (Heat Gen)
        size = 80
        A = [[random.random() for _ in range(size)] for _ in range(size)]
        B = [[random.random() for _ in range(size)] for _ in range(size)]
        _ = [[sum(a*b for a,b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]
        with cnt.get_lock(): cnt.value += 1

def gpu_load_task(stop_event, throttle_event):
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_SOURCE_CODE)
        func = mod.get_function("kxt_heavy_load")
        out = cuda.mem_alloc(4096)
        
        while not stop_event.is_set():
            throttle_event.wait()
            func(out, np.uint32(time.time()), block=(512,1,1), grid=(128,1))
            cuda.Context.synchronize()
    except: time.sleep(0.1)

def run_benchmark():
    os.system('clear')
    print("\n" + "="*50)
    print(" KXT v20 SUITE - HARDWARE AUDIT ".center(50))
    print("="*50 + "\n")
    
    HardwareHAL.set_fans_100()
    
    stop_ev = mp.Event()
    throttle_ev = mp.Event()
    throttle_ev.set()
    cnt = mp.Value('i', 0)
    
    procs = []
    
    # CPU STRESS
    print(f"[+] Spawning {mp.cpu_count()} CPU Load Threads...")
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_load_task, args=(stop_ev, throttle_ev, cnt))
        p.start()
        procs.append(p)
    
    # GPU STRESS
    print(f"[+] Adding GPU Load...")
    gp = mp.Process(target=gpu_load_task, args=(stop_ev, throttle_ev))
    gp.start()
    procs.append(gp)
        
    start_time = time.time()
    throttled = False
    
    print(f"\n[+] Starting Stress Test ({CONFIG['BENCH_DURATION']}s)...")
    try:
        while time.time() - start_time < CONFIG['BENCH_DURATION']:
            rem = int(CONFIG['BENCH_DURATION'] - (time.time() - start_time))
            c = HardwareHAL.get_cpu_temp()
            g = HardwareHAL.get_gpu_temp()
            
            # 79C GOVERNOR
            if not throttled and c >= CONFIG['TEMP_LIMIT']:
                throttle_ev.clear()
                throttled = True
            elif throttled and c <= CONFIG['TEMP_RESUME']:
                throttle_ev.set()
                throttled = False
                
            status = "MAX POWER" if not throttled else f"THROTTLED (> {CONFIG['TEMP_LIMIT']}C)"
            
            # Load Indicator
            ops = cnt.value
            
            print(f"\r >> Time: {rem}s | CPU: {c:.1f}C | GPU: {g:.1f}C | OPS: {ops} | {status}    ", end="")
            sys.stdout.flush()
            time.sleep(1)
            
    except KeyboardInterrupt:
        stop_ev.set()
        for p in procs: p.terminate()
        final_shutdown()
        
    stop_ev.set()
    for p in procs: p.terminate()
    print("\n\n[SUCCESS] Benchmark Complete. Initializing Miner...")
    time.sleep(2)

# ==============================================================================
# SECTION 7: MINING CORE (BLOCK FINDER FIXES)
# ==============================================================================

class StratumProtocol(threading.Thread):
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
        
    def run(self):
        while not EXIT_FLAG.is_set():
            try:
                self.sock = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=10)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                # Subscribe & Authorize
                self.send("mining.subscribe", ["KXT-v20Reforged"])
                self.send("mining.authorize", [f"{CONFIG['WALLET']}.rig1", CONFIG['PASS']])
                self.state.connected = True
                self.log_q.put(("NET", "Connected to Pool"))
                
                while not EXIT_FLAG.is_set():
                    # Submit Shares
                    while not self.res_q.empty():
                        submission = self.res_q.get()
                        self.send("mining.submit", [
                            f"{CONFIG['WALLET']}.rig1",
                            submission['jid'],
                            submission['en2'],
                            submission['ntime'],
                            submission['nonce']
                        ])
                        self.log_q.put(("TX", f"Submitting Nonce {submission['nonce']}"))
                        self.state.tx_count += 1
                        
                    # Receive Data
                    r, _, _ = select.select([self.sock], [], [], 0.1)
                    if r:
                        data = self.sock.recv(4096)
                        if not data: break
                        self.buffer += data.decode()
                        while '\n' in self.buffer:
                            line, self.buffer = self.buffer.split('\n', 1)
                            if line: self.handle_message(json.loads(line))
            except Exception as e:
                self.state.connected = False
                self.log_q.put(("ERR", f"Link Reset: {e}"))
                time.sleep(5)
                
    def send(self, method, params):
        try:
            payload = json.dumps({"id": self.msg_id, "method": method, "params": params}) + "\n"
            self.sock.sendall(payload.encode())
            self.msg_id += 1
        except: pass
        
    def handle_message(self, msg):
        if msg.get('id') == 1 and msg.get('result'):
            self.state.extranonce1 = msg['result'][1]
            self.state.extranonce2_size = msg['result'][2]
        if msg.get('id') and msg.get('id') > 2:
            if msg.get('result'):
                self.state.accepted += 1
                self.log_q.put(("RX", "Share ACCEPTED"))
            else:
                self.state.rejected += 1
                self.log_q.put(("RX", "Share REJECTED"))
        if msg.get('method') == 'mining.notify':
            params = msg['params']
            self.log_q.put(("JOB", f"New Block: {params[0][:8]}"))
            if params[8]:
                while not self.job_q.empty():
                    try: self.job_q.get_nowait()
                    except: pass
            job_package = (params, self.state.extranonce1, self.state.extranonce2_size)
            for _ in range(mp.cpu_count() * 2):
                self.job_q.put(job_package)

def mining_worker(worker_id, job_q, res_q, stop_event, throttle_event):
    """
    Miner with correct block hashing implementation and thermal governor.
    """
    nonce_start = worker_id * 10000000
    nonce = nonce_start
    current_job = None
    
    while not stop_event.is_set():
        throttle_event.wait() # 79C Governor check
        
        try:
            try:
                job_data = job_q.get(timeout=0.1)
                params, en1, en2_size = job_data
                if params[8]: # Clean job
                    nonce = nonce_start
                    current_job = job_data
                else:
                    current_job = job_data
            except queue.Empty:
                if current_job is None: continue
                
            # Unpack Job
            job_id, prev_hash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean = current_job[0]
            
            # 1. Random Extranonce2
            en2 = binascii.hexlify(os.urandom(en2_size)).decode()
            
            # 2. Coinbase & Merkle
            coinbase = binascii.unhexlify(coinb1 + en1 + en2 + coinb2)
            coinbase_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            merkle_root = coinbase_hash
            for branch in merkle_branch:
                branch_bin = binascii.unhexlify(branch)
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + branch_bin).digest()).digest()
                
            # 3. Block Header (Endian Swaps)
            header_prefix = (
                binascii.unhexlify(version)[::-1] +
                binascii.unhexlify(prev_hash)[::-1] +
                merkle_root +
                binascii.unhexlify(ntime)[::-1] +
                binascii.unhexlify(nbits)[::-1]
            )
            
            # 4. Search Loop
            target_bin = b'\x00\x00' # Diff approximation
            for n in range(nonce, nonce + 5000):
                nonce_bin = struct.pack('<I', n)
                header = header_prefix + nonce_bin
                block_hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                
                if block_hash.endswith(target_bin):
                    res_q.put({
                        'jid': job_id, 'en2': en2, 'ntime': ntime,
                        'nonce': binascii.hexlify(nonce_bin).decode()
                    })
                    break
            nonce += 5000
            
        except Exception: continue

class ProxyServer(threading.Thread):
    def __init__(self, log_q, state):
        super().__init__(); self.log_q = log_q; self.state = state; self.daemon = True
        
    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("0.0.0.0", CONFIG['PROXY_PORT'])); server.listen(50)
            self.log_q.put(("PRX", f"Proxy Active: {CONFIG['PROXY_PORT']}"))
            while True:
                c, a = server.accept()
                threading.Thread(target=self.handle, args=(c,), daemon=True).start()
        except: pass

    def handle(self, c):
        self.state.proxy_count += 1
        p = None
        try:
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
            self.state.proxy_count -= 1

# ==============================================================================
# SECTION 8: DASHBOARD UI
# ==============================================================================

def draw_window(stdscr, y, x, h, w, title, color):
    try:
        stdscr.attron(color)
        stdscr.addch(y, x, curses.ACS_ULCORNER); stdscr.addch(y, x+w-1, curses.ACS_URCORNER)
        stdscr.addch(y+h-1, x, curses.ACS_LLCORNER); stdscr.addch(y+h-1, x+w-1, curses.ACS_LRCORNER)
        stdscr.hline(y, x+1, curses.ACS_HLINE, w-2); stdscr.hline(y+h-1, x+1, curses.ACS_HLINE, w-2)
        stdscr.vline(y+1, x, curses.ACS_VLINE, h-2); stdscr.vline(y+1, x+w-1, curses.ACS_VLINE, h-2)
        stdscr.addstr(y, x+2, f" {title} "); stdscr.attroff(color)
    except: pass

def dashboard(stdscr, state, job_q, res_q, log_q):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    stdscr.nodelay(True); curses.curs_set(0)
    
    # Services
    StratumProtocol(state, job_q, res_q, log_q).start()
    ProxyServer(log_q, state).start()
    
    workers = []
    stop_workers = mp.Event()
    throttle_workers = mp.Event(); throttle_workers.set()
    
    for i in range(mp.cpu_count()):
        p = mp.Process(target=mining_worker, args=(i, job_q, res_q, stop_workers, throttle_workers))
        p.start(); workers.append(p)
        
    log_buffer = []
    throttled = False
    
    while True:
        while not log_q.empty():
            log_buffer.append(log_q.get())
            if len(log_buffer) > 20: log_buffer.pop(0)
            
        cpu_t = HardwareHAL.get_cpu_temp()
        gpu_t = HardwareHAL.get_gpu_temp()
        
        # 79C MINER GOVERNOR
        if not throttled and cpu_t >= CONFIG['TEMP_LIMIT']:
            throttle_workers.clear()
            throttled = True
            log_q.put(("WARN", "Temp > 79C. Throttling..."))
        elif throttled and cpu_t <= CONFIG['TEMP_RESUME']:
            throttle_workers.set()
            throttled = False
            log_q.put(("INFO", "Resuming..."))

        stdscr.erase(); h, w = stdscr.getmaxyx()
        
        title = " KXT MINER v20 [HEAVY] "
        stdscr.addstr(0, (w-len(title))//2, title, curses.A_REVERSE | curses.color_pair(4))
        
        # 4-Column Layout (Classic v20)
        draw_window(stdscr, 2, 1, 6, 30, "SYSTEM", curses.color_pair(4))
        stdscr.addstr(3, 3, f"CPU Temp: {cpu_t:.1f}C")
        stdscr.addstr(4, 3, f"GPU Temp: {gpu_t:.1f}C")
        status = "MINING" if not throttled else "THROTTLED"
        stdscr.addstr(5, 3, f"Status: {status}", curses.color_pair(1 if not throttled else 3))
            
        draw_window(stdscr, 2, 32, 6, 30, "NETWORK", curses.color_pair(4))
        stdscr.addstr(3, 34, f"Connected: {state.connected}")
        stdscr.addstr(4, 34, f"Shares: {state.tx_count}")
        
        draw_window(stdscr, 2, 63, 6, 30, "RESULTS", curses.color_pair(4))
        stdscr.addstr(3, 65, f"Accepted: {state.accepted}", curses.color_pair(1))
        stdscr.addstr(4, 65, f"Rejected: {state.rejected}", curses.color_pair(3))
        
        draw_window(stdscr, 2, 94, 6, 20, "PROXY", curses.color_pair(4))
        stdscr.addstr(3, 96, f"Clients: {state.proxy_count}")
        
        stdscr.hline(8, 0, curses.ACS_HLINE, w)
        for i, (lvl, msg) in enumerate(log_buffer):
            if 9+i >= h-1: break
            c = curses.color_pair(4)
            if lvl == "RX": c = curses.color_pair(1)
            if lvl == "ERR": c = curses.color_pair(3)
            if lvl == "JOB": c = curses.color_pair(2)
            stdscr.addstr(9+i, 2, f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}", c)
            
        stdscr.refresh()
        try:
            if stdscr.getch() == ord('q'): break
        except: pass
        time.sleep(0.1)
        
    stop_workers.set()
    for p in workers: p.terminate()
    final_shutdown()

# ==============================================================================
# SECTION 9: MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # 1. Run Text Benchmark (Pre-GUI)
    run_benchmark()
    
    # 2. Init Shared State
    manager = mp.Manager()
    state = manager.Namespace()
    state.connected = False
    state.tx_count = 0
    state.accepted = 0
    state.rejected = 0
    state.proxy_count = 0
    state.extranonce1 = "00000000"
    state.extranonce2_size = 4
    
    job_queue = manager.Queue()
    res_queue = manager.Queue()
    log_queue = manager.Queue()
    
    # 3. Launch Dashboard
    try:
        curses.wrapper(dashboard, state, job_queue, res_queue, log_queue)
    except KeyboardInterrupt:
        final_shutdown()
    except Exception as e:
        print(f"\n[CRASH] {e}")
        input("Press ENTER to exit...")
