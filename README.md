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
from datetime import datetime

# ==============================================================================
# SECTION 1: AUTO-DEPENDENCY & CLEANUP SYSTEM
# ==============================================================================

def kill_stale_processes():
    """Aggressively cleans up previous instances/zombies."""
    current_pid = os.getpid()
    try:
        # Try using pkill to remove other instances of this script
        subprocess.run("pkill -f kxt.py | grep -v " + str(current_pid), shell=True, stderr=subprocess.DEVNULL)
    except: pass
    
    # Clean up standard zombies
    try:
        os.waitpid(-1, os.WNOHANG)
    except: pass

kill_stale_processes()

# Auto-Dependency Check (From Screenshot)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[INSTALL] Installing psutil for system monitoring...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
        HAS_PSUTIL = True
    except:
        print("[WARN] Could not install psutil. Falling back to basic sensors.")

# ==============================================================================
# SECTION 2: CONFIGURATION
# ==============================================================================

CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.rig1",
    "PASS": "x",
    "PROXY_PORT": 60060,
    
    # SAFETY SETTINGS
    "BENCH_DURATION": 60,
    "TEMP_LIMIT": 79.0,   # Pause at 79C
    "TEMP_RESUME": 75.0,  # Resume at 75C
}

# ==============================================================================
# SECTION 3: CUDA ENGINE (VOLATILE KERNEL)
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
# SECTION 4: HARDWARE ABSTRACTION LAYER (HAL)
# ==============================================================================

class HardwareHAL:
    @staticmethod
    def get_cpu_temp():
        # Standard Sensor Logic (No complex filters)
        if HAS_PSUTIL:
            try:
                temps = psutil.sensors_temperatures()
                if not temps: return 0.0
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
# SECTION 5: BENCHMARK ENGINE (PRE-GUI)
# ==============================================================================

def cpu_load_task(stop_event, throttle_event):
    import math
    while not stop_event.is_set():
        throttle_event.wait()
        # Heavy Matrix Calculation
        size = 80
        A = [[1.1 for _ in range(size)] for _ in range(size)]
        B = [[2.2 for _ in range(size)] for _ in range(size)]
        _ = [[sum(a*b for a,b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

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
    
    procs = []
    
    # CPU STRESS
    print(f"[+] Spawning {mp.cpu_count()} CPU Load Threads...")
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_load_task, args=(stop_ev, throttle_ev))
        p.start()
        procs.append(p)
        
    start_time = time.time()
    throttled = False
    
    print(f"[+] Starting CPU Stress Test ({CONFIG['BENCH_DURATION']}s)...")
    try:
        while time.time() - start_time < CONFIG['BENCH_DURATION']:
            rem = int(CONFIG['BENCH_DURATION'] - (time.time() - start_time))
            temp = HardwareHAL.get_cpu_temp()
            
            # 79C GOVERNOR
            if not throttled and temp >= CONFIG['TEMP_LIMIT']:
                throttle_ev.clear()
                throttled = True
            elif throttled and temp <= CONFIG['TEMP_RESUME']:
                throttle_ev.set()
                throttled = False
                
            status = "MAX POWER" if not throttled else "COOLING DOWN"
            print(f"\r >> Time: {rem}s | CPU: {temp:.1f}C | Status: {status}    ", end="")
            time.sleep(1)
            
    except KeyboardInterrupt:
        stop_ev.set()
        for p in procs: p.terminate()
        sys.exit(0)
        
    # GPU STRESS
    print(f"\n[+] Adding GPU Load for final burn-in...")
    gp = mp.Process(target=gpu_load_task, args=(stop_ev, throttle_ev))
    gp.start()
    procs.append(gp)
    
    start_time = time.time()
    try:
        while time.time() - start_time < CONFIG['BENCH_DURATION']:
            rem = int(CONFIG['BENCH_DURATION'] - (time.time() - start_time))
            c = HardwareHAL.get_cpu_temp()
            g = HardwareHAL.get_gpu_temp()
            
            # 79C GOVERNOR (CPU Dependent)
            if not throttled and c >= CONFIG['TEMP_LIMIT']:
                throttle_ev.clear(); throttled = True
            elif throttled and c <= CONFIG['TEMP_RESUME']:
                throttle_ev.set(); throttled = False

            status = "MAX POWER" if not throttled else "COOLING DOWN"
            print(f"\r >> Time: {rem}s | CPU: {c:.1f}C | GPU: {g:.1f}C | Status: {status}    ", end="")
            time.sleep(1)
            
    except KeyboardInterrupt: pass
    
    stop_ev.set()
    for p in procs: p.terminate()
    print("\n\n[SUCCESS] Benchmark Complete. Initializing Miner...")
    time.sleep(2)

# ==============================================================================
# SECTION 6: MINING CORE (UPDATED BLOCK FINDER)
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
        while True:
            try:
                self.sock = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=10)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                # Subscribe & Authorize
                self.send("mining.subscribe", ["KXT-v20"])
                self.send("mining.authorize", [f"{CONFIG['WALLET']}.rig1", CONFIG['PASS']])
                self.state.connected = True
                self.log_q.put(("NET", "Connected to Pool"))
                
                while True:
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
                self.log_q.put(("ERR", f"Connection Lost: {e}"))
                time.sleep(5)
                
    def send(self, method, params):
        try:
            payload = json.dumps({"id": self.msg_id, "method": method, "params": params}) + "\n"
            self.sock.sendall(payload.encode())
            self.msg_id += 1
        except: pass
        
    def handle_message(self, msg):
        # Handle ExtraNonce
        if msg.get('id') == 1 and msg.get('result'):
            self.state.extranonce1 = msg['result'][1]
            self.state.extranonce2_size = msg['result'][2]
            
        # Handle Share Status
        if msg.get('id') and msg.get('id') > 2:
            if msg.get('result'):
                self.state.accepted += 1
                self.log_q.put(("RX", "Share ACCEPTED"))
            else:
                self.state.rejected += 1
                self.log_q.put(("RX", "Share REJECTED"))
                
        # Handle New Job
        if msg.get('method') == 'mining.notify':
            params = msg['params']
            job_id = params[0]
            clean_jobs = params[8]
            
            self.log_q.put(("JOB", f"New Block: {job_id[:8]}"))
            
            if clean_jobs:
                # Clear queue if job is clean
                while not self.job_q.empty():
                    try: self.job_q.get_nowait()
                    except: pass
            
            # Dispatch to workers
            job_package = (params, self.state.extranonce1, self.state.extranonce2_size)
            for _ in range(mp.cpu_count() * 2):
                self.job_q.put(job_package)

def mining_worker(worker_id, job_q, res_q, stop_event, throttle_event):
    """
    The updated miner worker with fixed Block Hashing logic.
    Includes Endian flipping and Double-SHA256 verification.
    """
    nonce_start = worker_id * 10000000
    nonce = nonce_start
    
    current_job = None
    
    while not stop_event.is_set():
        throttle_event.wait() # Governor Check
        
        try:
            # Get Job (Non-blocking)
            try:
                job_data = job_q.get(timeout=0.1)
                params, en1, en2_size = job_data
                
                # Unpack
                job_id, prev_hash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean = params
                
                if clean:
                    nonce = nonce_start # Reset nonce for clean job
                    current_job = job_data
                else:
                    current_job = job_data
                    
            except queue.Empty:
                if current_job is None: continue
                
            # Construct Block Header
            # 1. Generate Random ExtraNonce2
            en2 = binascii.hexlify(os.urandom(en2_size)).decode()
            
            # 2. Build Coinbase
            coinbase = binascii.unhexlify(coinb1 + en1 + en2 + coinb2)
            coinbase_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            
            # 3. Calculate Merkle Root
            merkle_root = coinbase_hash
            for branch in merkle_branch:
                branch_bin = binascii.unhexlify(branch)
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + branch_bin).digest()).digest()
                
            # 4. Construct Header (Endian Flips applied)
            # Version (LE), PrevHash (LE), Merkle (LE), Time (LE), Bits (LE), Nonce (LE)
            header_prefix = (
                binascii.unhexlify(version)[::-1] +
                binascii.unhexlify(prev_hash)[::-1] +
                merkle_root +
                binascii.unhexlify(ntime)[::-1] +
                binascii.unhexlify(nbits)[::-1]
            )
            
            # 5. Hash Loop
            target_bin = b'\x00\x00' # Approximation for share difficulty
            
            for n in range(nonce, nonce + 1000):
                nonce_bin = struct.pack('<I', n) # Little Endian Nonce
                header = header_prefix + nonce_bin
                block_hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                
                # Check for share (Trailing zeros in reversed hash / Leading zeros in BE)
                if block_hash.endswith(target_bin):
                    res_q.put({
                        'jid': job_id,
                        'en2': en2,
                        'ntime': ntime,
                        'nonce': binascii.hexlify(nonce_bin).decode() # Send as Hex string
                    })
                    # Don't break, keep mining same extranonce for a bit
            
            nonce += 1000
            
        except Exception:
            continue

# ==============================================================================
# SECTION 7: PROXY SERVER
# ==============================================================================

class ProxyServer(threading.Thread):
    def __init__(self, log_q, state):
        super().__init__()
        self.log_q = log_q
        self.state = state
        self.daemon = True
        
    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("0.0.0.0", CONFIG['PROXY_PORT']))
            server.listen(50)
            self.log_q.put(("PRX", f"Proxy Listening on {CONFIG['PROXY_PORT']}"))
            
            while True:
                client_sock, addr = server.accept()
                t = threading.Thread(target=self.handle_client, args=(client_sock, addr))
                t.daemon = True
                t.start()
        except Exception as e:
            self.log_q.put(("ERR", f"Proxy Bind Failed: {e}"))

    def handle_client(self, c_sock, addr):
        self.state.proxy_count += 1
        p_sock = None
        try:
            p_sock = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']))
            inputs = [c_sock, p_sock]
            
            while True:
                readable, _, _ = select.select(inputs, [], [], 1)
                if c_sock in readable:
                    data = c_sock.recv(4096)
                    if not data: break
                    p_sock.sendall(data)
                if p_sock in readable:
                    data = p_sock.recv(4096)
                    if not data: break
                    c_sock.sendall(data)
        except: pass
        finally:
            try: c_sock.close()
            except: pass
            try: p_sock.close()
            except: pass
            self.state.proxy_count -= 1

# ==============================================================================
# SECTION 8: DASHBOARD UI
# ==============================================================================

def draw_window(stdscr, y, x, h, w, title, color):
    try:
        stdscr.attron(color)
        stdscr.box()
        # Custom borders
        stdscr.addch(y, x, curses.ACS_ULCORNER)
        stdscr.addch(y, x+w-1, curses.ACS_URCORNER)
        stdscr.addch(y+h-1, x, curses.ACS_LLCORNER)
        stdscr.addch(y+h-1, x+w-1, curses.ACS_LRCORNER)
        stdscr.hline(y, x+1, curses.ACS_HLINE, w-2)
        stdscr.hline(y+h-1, x+1, curses.ACS_HLINE, w-2)
        stdscr.vline(y+1, x, curses.ACS_VLINE, h-2)
        stdscr.vline(y+1, x+w-1, curses.ACS_VLINE, h-2)
        stdscr.addstr(y, x+2, f" {title} ")
        stdscr.attroff(color)
    except: pass

def dashboard(stdscr, state, job_q, res_q, log_q):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    
    stdscr.nodelay(True)
    curses.curs_set(0)
    
    # Start Services
    stratum = StratumProtocol(state, job_q, res_q, log_q)
    stratum.start()
    
    proxy = ProxyServer(log_q, state)
    proxy.start()
    
    # Start Workers
    workers = []
    stop_workers = mp.Event()
    throttle_workers = mp.Event()
    throttle_workers.set()
    
    for i in range(mp.cpu_count()):
        p = mp.Process(target=mining_worker, args=(i, job_q, res_q, stop_workers, throttle_workers))
        p.start()
        workers.append(p)
        
    log_buffer = []
    throttled = False
    
    while True:
        # Process Logs
        while not log_q.empty():
            log_buffer.append(log_q.get())
            if len(log_buffer) > 20: log_buffer.pop(0)
            
        # Get Sensors
        cpu_t = HardwareHAL.get_cpu_temp()
        gpu_t = HardwareHAL.get_gpu_temp()
        
        # 79C Governor (Mining Phase)
        if not throttled and cpu_t >= CONFIG['TEMP_LIMIT']:
            throttle_workers.clear()
            throttled = True
            log_q.put(("WARN", "Temp > 79C. Throttling..."))
        elif throttled and cpu_t <= CONFIG['TEMP_RESUME']:
            throttle_workers.set()
            throttled = False
            log_q.put(("INFO", "Cooling complete. Resuming."))

        # Draw UI
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        # Header
        title = " KXT v20 GOLDEN EDITION "
        stdscr.addstr(0, (w-len(title))//2, title, curses.A_REVERSE | curses.color_pair(4))
        
        # 1. Local Stats
        draw_window(stdscr, 2, 1, 6, 30, "SYSTEM", curses.color_pair(4))
        stdscr.addstr(3, 3, f"CPU Temp: {cpu_t:.1f}C")
        stdscr.addstr(4, 3, f"GPU Temp: {gpu_t:.1f}C")
        if throttled:
            stdscr.addstr(5, 3, "STATUS: THROTTLED", curses.color_pair(3))
        else:
            stdscr.addstr(5, 3, "STATUS: MINING", curses.color_pair(1))
            
        # 2. Network
        draw_window(stdscr, 2, 32, 6, 30, "NETWORK", curses.color_pair(4))
        stdscr.addstr(3, 34, f"Pool: {CONFIG['POOL_URL']}")
        stdscr.addstr(4, 34, f"Connected: {state.connected}")
        stdscr.addstr(5, 34, f"TX Shares: {state.tx_count}")
        
        # 3. Performance
        draw_window(stdscr, 2, 63, 6, 30, "RESULTS", curses.color_pair(4))
        stdscr.addstr(3, 65, f"Accepted: {state.accepted}", curses.color_pair(1))
        stdscr.addstr(4, 65, f"Rejected: {state.rejected}", curses.color_pair(3))
        
        # 4. Proxy
        draw_window(stdscr, 2, 94, 6, 20, "PROXY", curses.color_pair(4))
        stdscr.addstr(3, 96, f"Clients: {state.proxy_count}")
        stdscr.addstr(4, 96, f"Port: {CONFIG['PROXY_PORT']}")
        
        # Logs
        stdscr.hline(8, 0, curses.ACS_HLINE, w)
        for i, (lvl, msg) in enumerate(log_buffer):
            if 9+i >= h-1: break
            color = curses.color_pair(4)
            if lvl == "RX": color = curses.color_pair(1)
            if lvl == "ERR": color = curses.color_pair(3)
            if lvl == "JOB": color = curses.color_pair(2)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            stdscr.addstr(9+i, 2, f"[{timestamp}] [{lvl}] {msg}", color)
            
        stdscr.refresh()
        
        # Input
        try:
            key = stdscr.getch()
            if key == ord('q'): break
        except: pass
        
        time.sleep(0.1)
        
    stop_workers.set()
    for p in workers: p.terminate()

# ==============================================================================
# SECTION 9: ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # 1. Run Text Benchmark
    run_benchmark()
    
    # 2. Setup Shared Memory
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
    
    # 3. Launch UI
    try:
        curses.wrapper(dashboard, state, job_queue, res_queue, log_queue)
    except KeyboardInterrupt:
        print("\n[KXT] Exiting...")
    except Exception as e:
        print(f"\n[CRASH] {e}")
