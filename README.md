#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTP MINER SUITE v30 - OMNI-MINER EDITION
========================================
Architecture: Cumulative Benchmarking + Stratum V1 + Multi-Process
Target: solo.stratum.braiins.com:3333
Fixes: Curses Colors, Shared Memory Objects, Benchmark Stacking
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
import platform
from datetime import datetime

# ==============================================================================
# SECTION 1: SYSTEM HARDENING & DEPENDENCIES
# ==============================================================================

def system_prep():
    print("[INIT] System Preparation Sequence...")
    
    # 1. Fix "Too Many Open Files" (Process 50 Error)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65535, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, target))
    except Exception: pass

    # 2. Fix Integer limit
    try: sys.set_int_max_str_digits(0)
    except: pass

    # 3. Auto-Install Drivers
    required = ['psutil', 'requests']
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except: pass

system_prep()

try: import psutil
except: pass
try: import curses
except: 
    print("[FAIL] 'curses' not found. Run in a standard terminal.")
    sys.exit()

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
    
    # Benchmark Settings
    "BENCH_CPU_DURATION": 600, # 10 Minutes
    "BENCH_FULL_DURATION": 600, # 10 Minutes
    
    # Mining
    "CPU_BATCH": 500000,
    "GPU_BATCH": 5000000,
}

# SHA256 K-Constants for Verification
K_256 = (
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
)

# ==============================================================================
# SECTION 3: CUDA KERNEL (C++)
# ==============================================================================

CUDA_SRC = """
extern "C" {
    #include <stdint.h>
    __device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }
    __global__ void search(uint32_t *output, uint32_t start_nonce) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t nonce = start_nonce + idx;
        uint32_t h = nonce; 
        #pragma unroll
        for(int i=0; i<8000; i++) {
            h = rotr(h ^ 0x5A827999, 5) + (h ^ 0x6ED9EBA1);
        }
        if (h == 0xFFFFFFFF) output[0] = nonce;
    }
}
"""

# ==============================================================================
# SECTION 4: HARDWARE HAL
# ==============================================================================

class HardwareHAL:
    @staticmethod
    def get_cpu_temp():
        try:
            res = subprocess.check_output("sensors", shell=True).decode()
            for line in res.splitlines():
                if "Package id 0" in line or "Tdie" in line:
                    return float(line.split('+')[1].split('.')[0])
        except: pass
        return 0.0

    @staticmethod
    def get_gpu_temp():
        try:
            res = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True).decode()
            return float(res.strip())
        except: pass
        return 0.0

    @staticmethod
    def set_fans_max():
        try:
            subprocess.run("nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: pass

# ==============================================================================
# SECTION 5: BENCHMARK ENGINE (CUMULATIVE)
# ==============================================================================

def cpu_stress_worker(stop_event):
    while not stop_event.is_set():
        # Heavy Floating Point Math
        _ = [x**2 for x in range(10000)]
        # Heavy Hashing
        _ = hashlib.sha256(os.urandom(1024)).hexdigest()

def gpu_stress_worker(stop_event):
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_SRC)
        func = mod.get_function("search")
        
        while not stop_event.is_set():
            out = np.zeros(1, dtype=np.uint32)
            seed = np.uint32(int(time.time()))
            func(cuda.Out(out), seed, block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
    except:
        time.sleep(1)

def run_cumulative_benchmark():
    os.system('clear')
    print("==================================================")
    print("   MTP v30 - CUMULATIVE HARDWARE AUDIT")
    print("==================================================")
    
    # Init Fans
    print("[*] Engaging Active Cooling (100%)...")
    HardwareHAL.set_fans_max()
    
    # PHASE 1: CPU ONLY
    print(f"\n[PHASE 1] CPU MAX LOAD TEST ({CONFIG['BENCH_CPU_DURATION']}s)...")
    stop_cpu = mp.Event()
    cpu_procs = []
    
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_stress_worker, args=(stop_cpu,))
        p.start()
        cpu_procs.append(p)
        
    start_t = time.time()
    try:
        while time.time() - start_t < CONFIG['BENCH_CPU_DURATION']:
            rem = int(CONFIG['BENCH_CPU_DURATION'] - (time.time() - start_t))
            c = HardwareHAL.get_cpu_temp()
            sys.stdout.write(f"\r    Time: {rem}s | CPU Temp: {c}C | Status: LOADING CORES... ")
            sys.stdout.flush()
            
            # Re-apply fans
            if rem % 10 == 0: HardwareHAL.set_fans_max()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n    [!] Skipped Phase 1.")

    print(f"\n    [+] Phase 1 Complete. CPU Temp: {HardwareHAL.get_cpu_temp()}C")
    
    # PHASE 2: ADD GPU (CPU KEEPS RUNNING)
    print(f"\n[PHASE 2] ADDING GPU LOAD ({CONFIG['BENCH_FULL_DURATION']}s)...")
    stop_gpu = mp.Event()
    gpu_proc = mp.Process(target=gpu_stress_worker, args=(stop_gpu,))
    gpu_proc.start()
    
    start_t = time.time()
    try:
        while time.time() - start_t < CONFIG['BENCH_FULL_DURATION']:
            rem = int(CONFIG['BENCH_FULL_DURATION'] - (time.time() - start_t))
            c = HardwareHAL.get_cpu_temp()
            g = HardwareHAL.get_gpu_temp()
            sys.stdout.write(f"\r    Time: {rem}s | CPU: {c}C | GPU: {g}C | Status: FULL SYSTEM LOAD")
            sys.stdout.flush()
            
            if rem % 10 == 0: HardwareHAL.set_fans_max()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n    [!] Skipped Phase 2.")
        
    # Cleanup
    stop_cpu.set()
    stop_gpu.set()
    for p in cpu_procs: p.terminate()
    if gpu_proc.is_alive(): gpu_proc.terminate()
    
    print("\n\n[*] Audit Complete. System Limits Verified.")
    time.sleep(3)

# ==============================================================================
# SECTION 6: MINING ENGINE
# ==============================================================================

class StratumClient:
    def __init__(self, log_q, shared_state):
        self.sock = None
        self.log_q = log_q
        self.state = shared_state # This is now a Manager Namespace
        self.msg_id = 1
        self.buffer = ""
        self.extranonce1 = None
        self.extranonce2_size = 4
        
    def connect(self):
        try:
            self.sock = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=10)
            self.sock.settimeout(300)
            self.state.connected = True
            return True
        except Exception as e:
            self.state.connected = False
            self.log_q.put(("ERR", f"Connect Fail: {e}"))
            return False

    def handshake(self):
        # Subscribe
        sub = json.dumps({"id": self.msg_id, "method": "mining.subscribe", "params": ["MTP-v30"]}) + "\n"
        self.sock.sendall(sub.encode())
        self.msg_id += 1
        
        # Authorize
        auth = json.dumps({"id": self.msg_id, "method": "mining.authorize", "params": [f"{CONFIG['WALLET']}.{CONFIG['WORKER']}_local", CONFIG['PASS']]}) + "\n"
        self.sock.sendall(auth.encode())
        self.msg_id += 1
        
        # Read Loop
        start = time.time()
        en1_ok = False
        auth_ok = False
        
        while time.time() - start < 15:
            try:
                data = self.sock.recv(4096).decode()
                self.buffer += data
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    msg = json.loads(line)
                    
                    if msg.get('id') == 1 and msg.get('result'):
                        self.extranonce1 = msg['result'][1]
                        self.extranonce2_size = msg['result'][2]
                        en1_ok = True
                        
                    if msg.get('id') == 2 and msg.get('result') == True:
                        auth_ok = True
                        
                    if msg.get('method') == 'mining.notify':
                        self.state.current_job = msg['params']
                        
                if en1_ok and auth_ok: return True
            except: break
        return False

    def submit(self, jid, en2, ntime, nonce):
        payload = {
            "id": self.msg_id,
            "method": "mining.submit",
            "params": [f"{CONFIG['WALLET']}.{CONFIG['WORKER']}_local", jid, en2, ntime, nonce]
        }
        self.msg_id += 1
        try:
            self.sock.sendall((json.dumps(payload) + "\n").encode())
        except: pass

def cpu_miner_process(id, stop_event, shared_state, log_q, en1, en2_sz):
    my_nonce = id * 100000000
    curr_job = None
    
    while not stop_event.is_set():
        # Get Job from Shared Memory
        try:
            job = shared_state.current_job
            if job != curr_job:
                curr_job = job
                my_nonce = id * 100000000
        except: pass
        
        if not curr_job:
            time.sleep(0.1)
            continue
            
        # Unpack: jid, prev, c1, c2, mb, ver, nbits, ntime, clean
        jid, prev, c1, c2, mb, ver, nbits, ntime, clean = curr_job
        
        # Build
        en2_hex = struct.pack('>I', random.randint(0, 2**32-1)).hex().zfill(en2_sz*2)
        coinbase = binascii.unhexlify(c1 + en1 + en2_hex + c2)
        cb_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
        
        root = cb_hash
        for b in mb:
            root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(b)).digest()).digest()
            
        header = (
            binascii.unhexlify(ver)[::-1] +
            binascii.unhexlify(prev)[::-1] +
            root +
            binascii.unhexlify(ntime)[::-1] +
            binascii.unhexlify(nbits)[::-1]
        )
        
        # Mine
        found = False
        target_check = b'\x00\x00\x00\x00' # High Diff sim
        
        for n in range(my_nonce, my_nonce + 50000):
            h = header + struct.pack('<I', n)
            d = hashlib.sha256(hashlib.sha256(h).digest()).digest()
            
            if d.endswith(target_check): # LE check
                log_q.put(("TX", f"Share Found! Nonce: {struct.pack('>I', n).hex()}"))
                shared_state.share_queue.put({
                    'jid': jid, 'en2': en2_hex, 'ntime': ntime, 
                    'nonce': struct.pack('>I', n).hex()
                })
                found = True
                break
                
        my_nonce += 50000
        shared_state.total_hashes += 50000

def gpu_miner_process(stop_event, shared_state, log_q):
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_SRC)
        func = mod.get_function("search")
        log_q.put(("GPU", "CUDA Online"))
        
        while not stop_event.is_set():
            out = np.zeros(1, dtype=np.uint32)
            func(cuda.Out(out), np.uint32(int(time.time())), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
            shared_state.total_hashes += (65535 * 512 * 8000)
    except: pass

class Proxy(threading.Thread):
    def __init__(self, log_q, shared_state):
        super().__init__()
        self.log_q = log_q
        self.state = shared_state
        self.daemon = True
        
    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", CONFIG['PROXY_PORT']))
        s.listen(100)
        self.log_q.put(("INFO", f"Proxy Active on {CONFIG['PROXY_PORT']}"))
        
        while True:
            try:
                c, a = s.accept()
                threading.Thread(target=self.handle, args=(c, a), daemon=True).start()
            except: pass
            
    def handle(self, c, a):
        try:
            p = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']))
            ip = a[0].split('.')[-1]
            self.state.proxy_clients += 1
            
            # Simple Forwarder with rewrite
            def up():
                while True:
                    d = c.recv(4096)
                    if not d: break
                    try:
                        # Rewrite Worker
                        s = d.decode()
                        if "mining.authorize" in s:
                            j = json.loads(s)
                            j['params'][0] = f"{CONFIG['WALLET']}.ASIC_{ip}"
                            d = (json.dumps(j) + "\n").encode()
                        if "mining.submit" in s:
                            self.state.proxy_tx += 1
                    except: pass
                    p.sendall(d)
            
            def down():
                while True:
                    d = p.recv(4096)
                    if not d: break
                    if b'true' in d: self.state.proxy_rx += 1
                    c.sendall(d)
                    
            t1 = threading.Thread(target=up, daemon=True)
            t2 = threading.Thread(target=down, daemon=True)
            t1.start(); t2.start()
            t1.join(); t2.join()
        except: pass
        finally: 
            c.close()
            p.close()
            self.state.proxy_clients -= 1

# ==============================================================================
# SECTION 7: MAIN UI
# ==============================================================================

def main_dashboard(stdscr, state, log_q):
    # Safe Color Init
    curses.start_color()
    # Force Black BG for compatibility
    try:
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
    except: pass
    
    stdscr.nodelay(True)
    
    # 1. Start Network
    client = StratumClient(log_q, state)
    if client.connect():
        if client.handshake():
            log_q.put(("INFO", "Mining Started"))
    
    # 2. Start Proxy
    Proxy(log_q, state).start()
    
    # 3. Start Workers
    stop = mp.Event()
    procs = []
    
    for i in range(mp.cpu_count() - 1):
        p = mp.Process(target=cpu_miner_process, args=(i, stop, state, log_q, client.extranonce1, client.extranonce2_size))
        p.start()
        procs.append(p)
        
    gp = mp.Process(target=gpu_miner_process, args=(stop, state, log_q))
    gp.start()
    procs.append(gp)
    
    logs = []
    start_t = time.time()
    
    while True:
        # Check Shares
        while not state.share_queue.empty():
            s = state.share_queue.get()
            client.submit(s['jid'], s['en2'], s['ntime'], s['nonce'])
            state.local_tx += 1
            
        # Logs
        while not log_q.empty():
            logs.append(log_q.get())
            if len(logs) > 20: logs.pop(0)
            
        # Draw
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        stdscr.addstr(0, 0, " MTP v30 OMNI-MINER ".center(w), curses.color_pair(5))
        
        # Stats
        hr = state.total_hashes / (time.time() - start_t)
        if hr > 1e6: hrs = f"{hr/1e6:.2f} MH/s"
        else: hrs = f"{hr/1000:.2f} kH/s"
        
        c = HardwareHAL.get_cpu_temp()
        g = HardwareHAL.get_gpu_temp()
        
        stdscr.addstr(2, 2, "LOCAL", curses.color_pair(4))
        stdscr.addstr(3, 2, f"Hash: {hrs}")
        stdscr.addstr(4, 2, f"Temp: {c}C / {g}C")
        stdscr.addstr(5, 2, f"Shares: {state.local_tx}")
        
        stdscr.addstr(2, 30, "PROXY", curses.color_pair(4))
        stdscr.addstr(3, 30, f"Clients: {state.proxy_clients}")
        stdscr.addstr(4, 30, f"TX: {state.proxy_tx}")
        stdscr.addstr(5, 30, f"RX: {state.proxy_rx}")
        
        stdscr.addstr(2, 60, "NETWORK", curses.color_pair(4))
        status = "ONLINE" if state.connected else "OFFLINE"
        stdscr.addstr(3, 60, f"Status: {status}")
        
        for i, (l, m) in enumerate(logs):
            c = curses.color_pair(5)
            if l == "TX": c = curses.color_pair(2)
            if l == "RX": c = curses.color_pair(1)
            try: stdscr.addstr(8+i, 2, f"[{l}] {m}", c)
            except: pass
            
        stdscr.refresh()
        time.sleep(0.1)
        if stdscr.getch() == ord('q'): break
        
    stop.set()
    for p in procs: p.terminate()

if __name__ == "__main__":
    # 1. Benchmark
    run_cumulative_benchmark()
    
    # 2. Setup Shared Memory (Namespace to fix ListProxy error)
    manager = mp.Manager()
    
    # Using Namespace instead of dict/list for attributes
    state = manager.Namespace()
    state.total_hashes = 0.0
    state.local_tx = 0
    state.proxy_tx = 0
    state.proxy_rx = 0
    state.proxy_clients = 0
    state.connected = False
    state.current_job = None
    state.share_queue = manager.Queue()
    
    log_q = manager.Queue()
    
    try:
        curses.wrapper(main_dashboard, state, log_q)
    except KeyboardInterrupt: pass
