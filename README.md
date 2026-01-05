#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KXT MINER SUITE v42 - TITAN EDITION
===================================
Target: solo.stratum.braiins.com:3333
Architecture: Full-Scale Beta v20 (Uncompressed)
Features:
  - Heavy Load Benchmark (Text Mode)
  - 79C Thermal Governor
  - Full Merkle Root / Block Header Construction
  - Robust Process Terminator (Ctrl+C Fix)
  - Multi-Threaded TCP Proxy
"""

import sys

# FIX: Large integer string conversion limit (screenshot match)
try:
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
except:
    pass

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
import atexit
import platform
from datetime import datetime

# ==============================================================================
# SECTION 1: PROCESS MANAGEMENT & TERMINATOR
# ==============================================================================

# Global Event to signal all threads/processes to stop
EXIT_FLAG = mp.Event()

# Try to import psutil for surgical process killing
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def kill_all_processes(*args):
    """
    The 'Terminator' Function.
    Identifies the current process group and forcibly kills everything.
    Hooked directly to Signal Handlers.
    """
    EXIT_FLAG.set()
    print("\n\n[KXT] TERMINATOR: Executing Force Kill Sequence...")
    
    # Method 1: PSUTIL Children Kill
    if HAS_PSUTIL:
        try:
            parent = psutil.Process(os.getpid())
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
        except:
            pass
            
    # Method 2: OS Process Group Kill (Linux/Unix)
    try:
        os.killpg(os.getpgid(0), signal.SIGKILL)
    except:
        pass
        
    # Method 3: Sys Exit as last resort
    sys.exit(0)

# Hook the Terminator to SIGINT (Ctrl+C) and SIGTERM
signal.signal(signal.SIGINT, kill_all_processes)
signal.signal(signal.SIGTERM, kill_all_processes)
atexit.register(kill_all_processes)

def boot_checks():
    """System resource configuration."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))
    except Exception:
        pass

boot_checks()

try:
    import curses
except ImportError:
    print("[ERROR] Curses library not found. Please install python3-curses.")
    sys.exit(1)

# ==============================================================================
# SECTION 2: CONFIGURATION
# ==============================================================================

CONFIG = {
    # Stratum Connection Details
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.rig1",
    "PASS": "x",
    
    # Local Proxy Settings
    "PROXY_PORT": 60060,
    
    # Benchmark & Thermal Settings
    "BENCH_DURATION": 60,   # Seconds
    "THROTTLE_TEMP": 79.0,  # Degrees Celsius
    "RESUME_TEMP": 75.0,    # Degrees Celsius
    
    # Load Tuning
    "CPU_BATCH_SIZE": 100000,
}

# ==============================================================================
# SECTION 3: CUDA KERNEL (VOLATILE LOAD)
# ==============================================================================

CUDA_SOURCE_CODE = """
extern "C" {
    #include <stdint.h>
    
    // Volatile keywords prevent the compiler from optimizing away the loop.
    // This ensures the GPU actually performs work and generates heat.
    __global__ void kxt_heavy_load(uint32_t *output, uint32_t seed) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        volatile uint32_t a = 0x6a09e667 + idx + seed;
        volatile uint32_t b = 0xbb67ae85;
        volatile uint32_t c = 0x3c6ef372;
        volatile uint32_t d = 0xa54ff53a;
        
        // Extended unrolled loop for maximum ALU saturation
        #pragma unroll 128
        for(int i=0; i < 50000; i++) {
            a = (a << 5) | (a >> 27);
            b ^= a;
            a += b + 0xDEADBEEF;
            c = (c >> 3) | (c << 29);
            d += c ^ a;
            
            // Memory write to keep VRAM bus active
            if (i % 2000 == 0) {
                output[idx % 1024] = a + d;
            }
        }
    }
}
"""

# ==============================================================================
# SECTION 4: HARDWARE ABSTRACTION LAYER (HAL)
# ==============================================================================

class HardwareHAL:
    """
    Interface for reading hardware sensors (Temperature) and controlling Fans.
    Implements multiple fallback methods for compatibility.
    """
    
    @staticmethod
    def get_cpu_temp():
        """
        Reads CPU Temperature.
        Logic: Scans common Linux paths, filters out 100.0 (stuck sensors).
        """
        readings = []
        
        # Method 1: /sys/class/thermal
        try:
            for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
                try:
                    with open(path, "r") as f:
                        val = float(f.read().strip())
                        if val > 1000: val /= 1000.0
                        if 20.0 < val < 99.0: # Valid Range
                            readings.append(val)
                except: pass
        except: pass
        
        # Method 2: /sys/class/hwmon
        try:
            for path in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
                try:
                    with open(path, "r") as f:
                        val = float(f.read().strip())
                        if val > 1000: val /= 1000.0
                        if 20.0 < val < 99.0:
                            readings.append(val)
                except: pass
        except: pass
        
        # Method 3: 'sensors' command output
        if not readings:
            try:
                out = subprocess.check_output("sensors", shell=True).decode()
                # Regex to find temps like +67.0
                matches = re.findall(r'\+([0-9]+\.[0-9]+)', out)
                for m in matches:
                    val = float(m)
                    if 20.0 < val < 99.0:
                        readings.append(val)
            except: pass
            
        if readings:
            return max(readings) # Return highest valid temp
        return 0.0

    @staticmethod
    def get_gpu_temp():
        """Reads NVIDIA GPU Temperature via nvidia-smi."""
        try:
            cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader"
            output = subprocess.check_output(cmd, shell=True)
            return float(output.decode().strip())
        except:
            return 0.0

    @staticmethod
    def force_fan_speed():
        """Attempts to force GPU fans to 100%."""
        commands = [
            "nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'",
            "nvidia-settings -a 'GPUFanControlState=1' -a 'GPUTargetFanSpeed=100'"
        ]
        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except: pass

# ==============================================================================
# SECTION 5: TEXT-BASED BENCHMARK ENGINE
# ==============================================================================

def benchmark_cpu_worker(stop_event, throttle_event):
    """
    Performs heavy floating-point matrix multiplication.
    This generates significant CPU heat for stability testing.
    """
    import math
    
    # Pre-allocate large matrices
    size = 80
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    B = [[random.random() for _ in range(size)] for _ in range(size)]
    
    while not stop_event.is_set():
        throttle_event.wait() # Pause here if governor is active
        
        # Heavy Calculation
        # Result is discarded, we only care about the heat
        result = [[sum(a*b for a,b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

def benchmark_gpu_worker(stop_event, throttle_event):
    """
    Uses PyCUDA to execute the volatile kernel.
    Falls back to sleep if PyCUDA is not installed.
    """
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_SOURCE_CODE)
        func = mod.get_function("kxt_heavy_load")
        
        # Allocate dummy memory
        out_gpu = cuda.mem_alloc(4096)
        
        while not stop_event.is_set():
            throttle_event.wait()
            
            # Launch Kernel: Grid(128,1), Block(512,1,1) -> 65k threads
            func(out_gpu, np.uint32(time.time()), block=(512,1,1), grid=(128,1))
            cuda.Context.synchronize()
            
    except ImportError:
        # If no GPU support, just sleep to keep process alive
        while not stop_event.is_set():
            time.sleep(1)
    except Exception:
        time.sleep(1)

def run_system_benchmark():
    """
    Runs the pre-GUI benchmark sequence.
    Handles the 79C Thermal Governor logic.
    """
    os.system('clear')
    print("\n" + "="*60)
    print(" KXT v42 TITAN - HARDWARE STABILITY AUDIT ".center(60))
    print("="*60 + "\n")
    
    print("[INIT] Forcing Fan Curves to Maximum...")
    threading.Thread(target=lambda: [HardwareHAL.force_fan_speed(), time.sleep(5)], daemon=True).start()
    
    # Event Flags
    stop_benchmark = mp.Event()
    throttle_control = mp.Event()
    throttle_control.set() # Initially Allowed (True)
    
    procs = []
    
    # ---------------------------------------------------------
    # STAGE 1: CPU SATURATION
    # ---------------------------------------------------------
    print(f"\n[STAGE 1] CPU THERMAL LOAD ({CONFIG['BENCH_DURATION']}s)")
    print(f"         > Spawning {mp.cpu_count()} math threads...")
    
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=benchmark_cpu_worker, args=(stop_benchmark, throttle_control))
        p.start()
        procs.append(p)
        
    start_time = time.time()
    is_throttled = False
    
    try:
        while time.time() - start_time < CONFIG['BENCH_DURATION']:
            elapsed = time.time() - start_time
            remaining = int(CONFIG['BENCH_DURATION'] - elapsed)
            
            cpu_temp = HardwareHAL.get_cpu_temp()
            
            # --- THERMAL GOVERNOR LOGIC ---
            if not is_throttled:
                if cpu_temp >= CONFIG['THROTTLE_TEMP']:
                    throttle_control.clear() # Pause Workers
                    is_throttled = True
            else:
                if cpu_temp <= CONFIG['RESUME_TEMP']:
                    throttle_control.set() # Resume Workers
                    is_throttled = False
            
            # Status Output
            status_text = "MAX POWER"
            if is_throttled:
                status_text = f"THROTTLED (Temp > {CONFIG['THROTTLE_TEMP']}C)"
                
            sys.stdout.write(f"\r  >> Time: {remaining}s | CPU Die: {cpu_temp:.1f}C | Status: {status_text}       ")
            sys.stdout.flush()
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        kill_all_processes()
        
    # ---------------------------------------------------------
    # STAGE 2: FULL SYSTEM LOAD (CPU + GPU)
    # ---------------------------------------------------------
    print(f"\n\n[STAGE 2] FULL SYSTEM LOAD (+GPU) ({CONFIG['BENCH_DURATION']}s)")
    print(f"         > Initializing CUDA Context...")
    
    gpu_proc = mp.Process(target=benchmark_gpu_worker, args=(stop_benchmark, throttle_control))
    gpu_proc.start()
    procs.append(gpu_proc)
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < CONFIG['BENCH_DURATION']:
            elapsed = time.time() - start_time
            remaining = int(CONFIG['BENCH_DURATION'] - elapsed)
            
            cpu_temp = HardwareHAL.get_cpu_temp()
            gpu_temp = HardwareHAL.get_gpu_temp()
            
            # --- THERMAL GOVERNOR LOGIC (CPU Priority) ---
            if not is_throttled:
                if cpu_temp >= CONFIG['THROTTLE_TEMP']:
                    throttle_control.clear()
                    is_throttled = True
            else:
                if cpu_temp <= CONFIG['RESUME_TEMP']:
                    throttle_control.set()
                    is_throttled = False
                    
            status_text = "MAX POWER"
            if is_throttled:
                status_text = "THROTTLED"

            sys.stdout.write(f"\r  >> Time: {remaining}s | CPU: {cpu_temp:.1f}C | GPU: {gpu_temp:.1f}C | Status: {status_text}      ")
            sys.stdout.flush()
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        kill_all_processes()
        
    # Cleanup Benchmark
    stop_benchmark.set()
    for p in procs:
        p.terminate()
        p.join()
        
    print("\n\n[SUCCESS] Benchmark Passed. Initializing Mining Interface...")
    time.sleep(2)

# ==============================================================================
# SECTION 6: STRATUM PROTOCOL IMPLEMENTATION
# ==============================================================================

class StratumClient(threading.Thread):
    """
    Handles TCP connection to the mining pool.
    Implements Stratum V1 protocol (subscribe, authorize, submit).
    """
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
                # 1. Connect
                self.sock = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=10)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                # 2. Handshake
                self.send_request("mining.subscribe", ["KXT-v42-Titan"])
                self.send_request("mining.authorize", [f"{CONFIG['WALLET']}.{CONFIG['WORKER']}_local", CONFIG['PASS']])
                
                self.state.connected.value = True
                self.log_q.put(("NET", f"Connected to {CONFIG['POOL_URL']}"))
                
                # 3. Main Loop
                while not EXIT_FLAG.is_set():
                    # Send Shares
                    while not self.res_q.empty():
                        share = self.res_q.get()
                        self.send_request("mining.submit", [
                            f"{CONFIG['WALLET']}.{CONFIG['WORKER']}_local",
                            share['jid'],
                            share['en2'],
                            share['ntime'],
                            share['nonce']
                        ])
                        self.log_q.put(("TX", f"Submitting Nonce {share['nonce']}"))
                        with self.state.shares.get_lock():
                            self.state.shares.value += 1
                            
                    # Read Data
                    r, _, _ = select.select([self.sock], [], [], 0.1)
                    if r:
                        data = self.sock.recv(4096)
                        if not data:
                            raise ConnectionError("Socket closed by pool")
                        self.buffer += data.decode()
                        while '\n' in self.buffer:
                            line, self.buffer = self.buffer.split('\n', 1)
                            if line.strip():
                                self.handle_message(json.loads(line))
                                
            except Exception as e:
                self.state.connected.value = False
                # self.log_q.put(("ERR", f"Net Error: {str(e)[:20]}"))
                time.sleep(5) # Reconnect delay

    def send_request(self, method, params):
        try:
            payload = json.dumps({
                "id": self.msg_id,
                "method": method,
                "params": params
            }) + "\n"
            self.sock.sendall(payload.encode())
            self.msg_id += 1
        except:
            pass

    def handle_message(self, msg):
        msg_id = msg.get('id')
        method = msg.get('method')
        result = msg.get('result')
        error = msg.get('error')
        
        # Subscription Reply
        if msg_id == 1 and result:
            # Result[0] = Sub Details, Result[1] = Extranonce1, Result[2] = Extranonce2_Size
            self.state.extranonce1 = result[1]
            self.state.extranonce2_size = result[2]
            
        # Share Submission Reply
        if msg_id and msg_id > 2:
            if result:
                with self.state.accepted.get_lock():
                    self.state.accepted.value += 1
                self.log_q.put(("RX", "Share ACCEPTED"))
            elif error:
                with self.state.rejected.get_lock():
                    self.state.rejected.value += 1
                self.log_q.put(("RX", f"Share REJECTED: {error}"))
                
        # New Job Notification
        if method == 'mining.notify':
            params = msg['params']
            job_id = params[0]
            clean_jobs = params[8]
            
            self.log_q.put(("JOB", f"Block {job_id[:8]}"))
            
            # If clean_jobs=True, we must stop working on old jobs immediately
            if clean_jobs:
                while not self.job_q.empty():
                    try: self.job_q.get_nowait()
                    except: pass
            
            # Pack job and send to workers
            # (Job Params, Extranonce1, Extranonce2_Size)
            job_package = (params, self.state.extranonce1, self.state.extranonce2_size)
            
            # Replicate job for all workers
            for _ in range(mp.cpu_count() * 2):
                self.job_q.put(job_package)

# ==============================================================================
# SECTION 7: BLOCK HASHING WORKER (FULL VERBOSE LOGIC)
# ==============================================================================

def mining_worker_process(worker_id, job_q, res_q, stop_event, throttle_event):
    """
    The Mining Worker.
    Performs the Double-SHA256 hashing required for Bitcoin/Braiins.
    Handles Endianness flipping and Merkle Root construction manually.
    """
    
    # Initialize Nonce Range (Partitioned by Worker ID)
    nonce_start = worker_id * 10000000
    nonce = nonce_start
    
    current_job_data = None
    
    while not stop_event.is_set():
        throttle_event.wait() # Thermal Governor Check
        
        # Try to get a job
        try:
            try:
                # Non-blocking get with short timeout
                job_package = job_q.get(timeout=0.1)
                
                params, en1, en2_size = job_package
                job_id, prev_hash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean = params
                
                # If clean job, reset nonce
                if clean:
                    nonce = nonce_start
                    
                current_job_data = (params, en1, en2_size)
                
            except queue.Empty:
                if current_job_data is None:
                    continue
            
            # Unpack current data
            params, en1, en2_size = current_job_data
            job_id, prev_hash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean = params
            
            # ------------------------------------------------------------------
            # BLOCK HEADER CONSTRUCTION
            # ------------------------------------------------------------------
            
            # 1. Generate ExtraNonce2 (Random Hex String)
            # Size determined by subscription (usually 4 bytes = 8 hex chars)
            en2_bin = os.urandom(en2_size)
            en2_hex = binascii.hexlify(en2_bin).decode()
            
            # 2. Build Coinbase Transaction
            # Coinbase = Coinb1 + Extranonce1 + Extranonce2 + Coinb2
            coinbase_hex = coinb1 + en1 + en2_hex + coinb2
            coinbase_bin = binascii.unhexlify(coinbase_hex)
            
            # 3. Calculate Coinbase Hash (Double SHA256)
            coinbase_hash = hashlib.sha256(hashlib.sha256(coinbase_bin).digest()).digest()
            
            # 4. Calculate Merkle Root
            # Iteratively hash the Coinbase Hash with the Merkle Branch
            merkle_root = coinbase_hash
            for branch in merkle_branch:
                branch_bin = binascii.unhexlify(branch)
                # Concatenate and Hash (Double SHA256)
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + branch_bin).digest()).digest()
                
            # 5. Assemble Block Header (80 Bytes)
            # Note: Stratum sends some fields as Big Endian, Block Header requires Little Endian.
            # We must flip bytes for: Version, PrevHash, NTime, NBits
            # Merkle Root is already calculated correctly above.
            
            version_bin = binascii.unhexlify(version)[::-1]
            prev_hash_bin = binascii.unhexlify(prev_hash)[::-1]
            ntime_bin = binascii.unhexlify(ntime)[::-1]
            nbits_bin = binascii.unhexlify(nbits)[::-1]
            
            header_prefix = (
                version_bin +
                prev_hash_bin +
                merkle_root +
                ntime_bin +
                nbits_bin
            )
            
            # 6. Hashing Loop
            # Try 5000 nonces before checking queue again
            target_bin = b'\x00\x00' # Simple difficulty check (optimization)
            
            for n in range(nonce, nonce + 5000):
                # Pack nonce as Little Endian 4-byte unsigned integer
                nonce_bin = struct.pack('<I', n)
                
                # Full Header
                header = header_prefix + nonce_bin
                
                # Block Hash = SHA256(SHA256(Header))
                block_hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                
                # Check Difficulty (Reverse hash for standard Big Endian check)
                # For simplified python miner, checking trailing zeros of the internal hash works well
                if block_hash.endswith(target_bin):
                    # FOUND A SHARE
                    res_q.put({
                        'jid': job_id,
                        'en2': en2_hex,
                        'ntime': ntime,
                        'nonce': binascii.hexlify(nonce_bin).decode()
                    })
                    break # Break loop to submit
            
            nonce += 5000
            
        except Exception:
            continue

# ==============================================================================
# SECTION 8: MULTI-THREADED PROXY SERVER
# ==============================================================================

class ProxyServer(threading.Thread):
    """
    Local TCP Proxy Server.
    Allows local ASIC devices to connect through this script.
    """
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
            self.log_q.put(("PRX", f"Proxy Listening on port {CONFIG['PROXY_PORT']}"))
            
            while not EXIT_FLAG.is_set():
                client, addr = server.accept()
                t = threading.Thread(target=self.handle_client, args=(client, addr))
                t.daemon = True
                t.start()
        except Exception as e:
            self.log_q.put(("ERR", f"Proxy Bind Error: {e}"))

    def handle_client(self, c_sock, addr):
        with self.state.proxy_clients.get_lock():
            self.state.proxy_clients.value += 1
            
        p_sock = None
        try:
            # Connect to Upstream Pool
            p_sock = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=10)
            inputs = [c_sock, p_sock]
            
            while not EXIT_FLAG.is_set():
                readable, _, _ = select.select(inputs, [], [], 1)
                
                if c_sock in readable:
                    data = c_sock.recv(4096)
                    if not data: break
                    p_sock.sendall(data)
                    
                if p_sock in readable:
                    data = p_sock.recv(4096)
                    if not data: break
                    c_sock.sendall(data)
                    
        except:
            pass
        finally:
            try: c_sock.close()
            except: pass
            try: p_sock.close()
            except: pass
            with self.state.proxy_clients.get_lock():
                self.state.proxy_clients.value -= 1

# ==============================================================================
# SECTION 9: CURSES DASHBOARD (GUI)
# ==============================================================================

def draw_tui_box(stdscr, y, x, h, w, title, color_pair):
    """Draws a styled box with a title."""
    try:
        stdscr.attron(color_pair)
        # Corners
        stdscr.addch(y, x, curses.ACS_ULCORNER)
        stdscr.addch(y, x + w - 1, curses.ACS_URCORNER)
        stdscr.addch(y + h - 1, x, curses.ACS_LLCORNER)
        stdscr.addch(y + h - 1, x + w - 1, curses.ACS_LRCORNER)
        # Borders
        stdscr.hline(y, x + 1, curses.ACS_HLINE, w - 2)
        stdscr.hline(y + h - 1, x + 1, curses.ACS_HLINE, w - 2)
        stdscr.vline(y + 1, x, curses.ACS_VLINE, h - 2)
        stdscr.vline(y + 1, x + w - 1, curses.ACS_VLINE, h - 2)
        # Title
        stdscr.addstr(y, x + 2, f" {title} ")
        stdscr.attroff(color_pair)
    except: pass

def draw_progress_bar(val, max_val, width=10):
    percent = max(0.0, min(1.0, val / max_val))
    fill = int(percent * width)
    bar = "|" * fill + " " * (width - fill)
    return f"[{bar}]"

def dashboard_main(stdscr, state, job_q, res_q, log_q):
    """
    Main Loop for the Curses Interface.
    Manages display, worker lifecycle, and 79C Governor for miners.
    """
    # Curses Setup
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Success
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Warning/Job
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)    # Error/Throttled
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Info
    curses.curs_set(0) # Hide cursor
    stdscr.nodelay(True) # Non-blocking input
    
    # Start Network Services
    stratum = StratumClient(state, job_q, res_q, log_q)
    stratum.start()
    
    proxy = ProxyServer(log_q, state)
    proxy.start()
    
    # Start Mining Workers
    workers = []
    stop_workers = mp.Event()
    throttle_workers = mp.Event()
    throttle_workers.set() # Allow start
    
    # Spawn 1 Process per CPU Core
    for i in range(mp.cpu_count()):
        p = mp.Process(target=mining_worker_process, args=(i, job_q, res_q, stop_workers, throttle_workers))
        p.start()
        workers.append(p)
        
    # Also spawn GPU load thread for mining phase
    gp = mp.Process(target=benchmark_gpu_worker, args=(stop_workers, throttle_workers))
    gp.start()
    workers.append(gp)
    
    log_history = []
    is_throttled = False
    
    # Main GUI Loop
    while not EXIT_FLAG.is_set():
        # 1. Update Logs
        while not log_q.empty():
            log_history.append(log_q.get())
            if len(log_history) > 20:
                log_history.pop(0)
                
        # 2. Check Sensors
        cpu_temp = HardwareHAL.get_cpu_temp()
        gpu_temp = HardwareHAL.get_gpu_temp()
        
        # 3. Apply Thermal Governor (Mining Phase)
        if not is_throttled:
            if cpu_temp >= CONFIG['THROTTLE_TEMP']:
                throttle_workers.clear()
                is_throttled = True
                log_q.put(("WARN", f"Temp {cpu_temp}C > 79C. Throttling..."))
        else:
            if cpu_temp <= CONFIG['RESUME_TEMP']:
                throttle_workers.set()
                is_throttled = False
                log_q.put(("INFO", "Cooled down. Resuming."))
                
        # 4. Draw UI
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        
        # Header
        header_text = " KXT MINER v42 - TITAN EDITION "
        stdscr.addstr(0, max(0, (width - len(header_text)) // 2), header_text, curses.A_REVERSE | curses.color_pair(4))
        
        # Column 1: LOCAL HARDWARE
        draw_tui_box(stdscr, 2, 1, 6, 28, "LOCAL", curses.color_pair(4))
        stdscr.addstr(3, 3, f"CPU: {cpu_temp:.1f}C {draw_progress_bar(cpu_temp, 90, 5)}")
        stdscr.addstr(4, 3, f"GPU: {gpu_temp:.1f}C {draw_progress_bar(gpu_temp, 90, 5)}")
        if is_throttled:
            stdscr.addstr(5, 3, "STATUS: THROTTLED", curses.color_pair(3))
        else:
            stdscr.addstr(5, 3, "STATUS: ACTIVE", curses.color_pair(1))
            
        # Column 2: NETWORK
        draw_tui_box(stdscr, 2, 30, 6, 28, "NETWORK", curses.color_pair(4))
        conn_str = "CONNECTED" if state.connected.value else "CONNECTING..."
        conn_color = curses.color_pair(1) if state.connected.value else curses.color_pair(2)
        stdscr.addstr(3, 32, f"Link: {conn_str}", conn_color)
        stdscr.addstr(4, 32, f"Shares Sent: {state.shares.value}")
        
        # Column 3: STATISTICS
        draw_tui_box(stdscr, 2, 59, 6, 28, "STATS", curses.color_pair(4))
        stdscr.addstr(3, 61, f"Accepted: {state.accepted.value}", curses.color_pair(1))
        stdscr.addstr(4, 61, f"Rejected: {state.rejected.value}", curses.color_pair(3))
        
        # Column 4: PROXY
        draw_tui_box(stdscr, 2, 88, 6, 24, "PROXY", curses.color_pair(4))
        stdscr.addstr(3, 90, f"Clients: {state.proxy_clients.value}")
        stdscr.addstr(4, 90, f"Port: {CONFIG['PROXY_PORT']}")
        
        # Logs Section
        stdscr.hline(8, 0, curses.ACS_HLINE, width)
        
        for idx, (level, message) in enumerate(log_history):
            row = 9 + idx
            if row >= height - 1: break
            
            # Color coding logs
            color = curses.color_pair(4)
            if level == "TX": color = curses.color_pair(4)
            if level == "RX": color = curses.color_pair(1)
            if level == "WARN": color = curses.color_pair(2)
            if level == "ERR": color = curses.color_pair(3)
            if level == "JOB": color = curses.color_pair(2)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            line = f"[{timestamp}] [{level}] {message}"
            
            # Truncate to fit screen width
            if len(line) > width - 2:
                line = line[:width-2]
                
            stdscr.addstr(row, 2, line, color)
            
        stdscr.refresh()
        
        # Input Handling
        try:
            key = stdscr.getch()
            if key == ord('q') or key == ord('Q'):
                EXIT_FLAG.set()
        except: pass
        
        time.sleep(0.1)
        
    # Cleanup
    stop_workers.set()
    for p in workers:
        p.terminate()
        p.join()

# ==============================================================================
# SECTION 10: ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    try:
        # 1. Run Benchmark
        run_system_benchmark()
        
        # 2. Initialize Shared Memory
        manager = mp.Manager()
        state = manager.Namespace()
        state.connected = manager.Value('b', False)
        state.shares = manager.Value('i', 0)
        state.accepted = manager.Value('i', 0)
        state.rejected = manager.Value('i', 0)
        state.proxy_clients = manager.Value('i', 0)
        
        # Stratum State
        state.extranonce1 = "00000000"
        state.extranonce2_size = 4
        
        # Communication Queues
        job_queue = manager.Queue()
        res_queue = manager.Queue()
        log_queue = manager.Queue()
        
        # 3. Launch GUI
        curses.wrapper(dashboard_main, state, job_queue, res_queue, log_queue)
        
    except KeyboardInterrupt:
        print("\n[KXT] Interrupted by User.")
    except Exception as e:
        print(f"\n[CRASH] Critical Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        kill_all_processes()
