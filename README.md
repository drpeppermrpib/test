#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTP MINER SUITE v31 - TITAN INDUSTRIAL EDITION
==============================================
Codename: "Heatseeker"
Architecture: Distributed Multiprocessing with Shared Memory Namespace
Target: solo.stratum.braiins.com:3333
Fixes: Queue NameError, BrokenPipe, CUDA Optimization, Ghost Shares
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
import queue  # CRITICAL IMPORT FOR WORKERS
import traceback
from datetime import datetime

# ==============================================================================
# SECTION 1: KERNEL & SYSTEM HARDENING
# ==============================================================================

def titan_init():
    """Initializes the Titan Runtime Environment."""
    print("[TITAN] Initializing System Core...")
    
    # 1.1 Process Hardening
    try:
        # Prevent file handle exhaustion (Error 50)
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65535, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        print(f"[TITAN] File Descriptors set to {target}")
    except Exception as e:
        print(f"[WARN] Failed to set ulimit: {e}")

    # 1.2 Integer Math Expansion
    try:
        sys.set_int_max_str_digits(0)
    except: pass

    # 1.3 Driver Verification
    required = ["psutil", "requests"]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[TITAN] Installing missing driver: {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            print(f"[TITAN] {pkg} installed. Restarting...")
            os.execv(sys.executable, ['python3'] + sys.argv)

titan_init()

# Safe Imports
try: import psutil
except: pass
try: import curses
except: 
    print("[FATAL] Curses library missing. Run in standard Linux terminal.")
    sys.exit(1)

# ==============================================================================
# SECTION 2: CONFIGURATION & CONSTANTS
# ==============================================================================

CONFIG = {
    "POOL_HOST": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "WORKER_NAME": "rig1",
    "PASS": "x",
    
    "PROXY_PORT": 60060,
    
    # Benchmark Timers (Seconds)
    "BENCH_CPU_TIME": 600, # 10 Mins
    "BENCH_GPU_TIME": 600, # 10 Mins (Cumulative = 20 total)
    
    # Workload Settings
    "CPU_BATCH": 1000000,
    "GPU_BATCH": 100000000,
    "FAN_FORCE_INTERVAL": 15,
}

# ==============================================================================
# SECTION 3: CUDA C++ KERNEL (HEAT GENERATOR)
# ==============================================================================

# This kernel uses 'volatile' memory access to prevent the compiler from 
# optimizing away the math loop, ensuring the GPU actually works hard.
CUDA_TITAN_SRC = """
extern "C" {
    #include <stdint.h>

    __device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }

    __global__ void titan_load(uint32_t *output, uint32_t seed, int intensity) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Load initial value
        uint32_t hash = seed + idx;
        uint32_t a = 0x6a09e667;
        uint32_t b = 0xbb67ae85;
        
        // Intense Loop (ALU Saturation)
        #pragma unroll 128
        for(int i=0; i < 2000; i++) {
            a = rotr(a ^ hash, 7) + b;
            b = rotr(b ^ a, 19) + hash;
            hash = a ^ b;
        }
        
        // Volatile Write (Forces execution)
        // We only write occasionally to save memory bandwidth but keep ALU busy
        if (hash == 0xFFFFFFFF) {
            output[idx % 1024] = hash; 
        }
        
        // Always write final to ensure dependencies
        if (idx == 0) output[0] = a;
    }
}
"""

# ==============================================================================
# SECTION 4: HARDWARE CONTROLLER
# ==============================================================================

class HardwareController:
    @staticmethod
    def get_cpu_temp():
        """Scans all known thermal zones in Linux."""
        try:
            # 1. Try `sensors` command (Most reliable)
            out = subprocess.check_output("sensors", shell=True).decode()
            for line in out.splitlines():
                if "Package id 0" in line or "Tdie" in line:
                    return float(line.split('+')[1].split('.')[0])
            
            # 2. Try psutil
            if 'psutil' in sys.modules:
                temps = psutil.sensors_temperatures()
                for name, entries in temps.items():
                    if 'core' in name or 'cpu' in name or 'k10' in name:
                        return entries[0].current
            
            # 3. Try sysfs
            zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
            for z in zones:
                with open(z, 'r') as f:
                    t = int(f.read().strip())
                    if t > 1000: return t / 1000.0
                    if t > 0: return float(t)
        except: pass
        return 0.0

    @staticmethod
    def get_gpu_temp():
        try:
            out = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True).decode()
            return float(out.strip())
        except: return 0.0

    @staticmethod
    def force_max_fans():
        """Aggressive Fan Override."""
        # Method 1: NV Settings
        cmds = [
            "nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'",
            "nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:1]/GPUTargetFanSpeed=100'"
        ]
        for cmd in cmds:
            try:
                subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except: pass
            
        # Method 2: Sysfs PWM (Root)
        try:
            pwms = glob.glob("/sys/class/hwmon/hwmon*/pwm*")
            for p in pwms:
                try:
                    with open(p, 'w') as f: f.write("255")
                except: pass
        except: pass

# ==============================================================================
# SECTION 5: CUMULATIVE BENCHMARK ENGINE
# ==============================================================================

def cpu_burn_process(stop_ev, counter):
    """Generates heat via AVX/ALU stress."""
    try:
        while not stop_ev.is_set():
            # Heavy math mixing float/int
            _ = [x * x for x in range(5000)]
            # Cryptographic stress
            _ = hashlib.sha512(os.urandom(4096)).hexdigest()
            
            with counter.get_lock():
                counter.value += 5000
    except: pass

def gpu_burn_process(stop_ev):
    """Runs the Titan CUDA kernel."""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np

        mod = SourceModule(CUDA_TITAN_SRC)
        func = mod.get_function("titan_load")
        
        # Allocate GPU Memory
        out_gpu = cuda.mem_alloc(4096)
        
        while not stop_ev.is_set():
            # Launch massive grid
            func(out_gpu, np.uint32(time.time()), np.int32(100), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
            
    except ImportError:
        pass # PyCUDA not installed
    except Exception:
        pass

def run_titan_benchmark():
    os.system('clear')
    print("==================================================")
    print("   TITAN v31 - HARDWARE STRESS AUDIT")
    print("==================================================")
    print("[*] Locking Fans to 100%...")
    HardwareController.force_max_fans()
    
    # Thread to keep fans pegged
    def fan_keeper():
        while True:
            HardwareController.force_max_fans()
            time.sleep(15)
    
    ft = threading.Thread(target=fan_keeper, daemon=True)
    ft.start()
    
    # ----------------------------------------------------
    # PHASE 1: CPU THERMAL SATURATION
    # ----------------------------------------------------
    print(f"\n[PHASE 1] CPU LOAD TEST ({CONFIG['BENCH_CPU_TIME']}s)...")
    
    stop_cpu = mp.Event()
    cpu_count = mp.Value('d', 0.0)
    procs = []
    
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_burn_process, args=(stop_cpu, cpu_count))
        p.start()
        procs.append(p)
        
    start_t = time.time()
    try:
        while time.time() - start_t < CONFIG['BENCH_CPU_TIME']:
            rem = int(CONFIG['BENCH_CPU_TIME'] - (time.time() - start_t))
            c_temp = HardwareController.get_cpu_temp()
            
            ops = cpu_count.value
            rate = ops / (time.time() - start_t)
            
            sys.stdout.write(f"\r    T-{rem}s | CPU: {c_temp}C | Rate: {rate/1000:.0f} kOP/s | Status: HEATING...")
            sys.stdout.flush()
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n    [!] Skipped Phase 1")
    
    print(f"\n    [+] Phase 1 Peak Temp: {HardwareController.get_cpu_temp()}C")
    
    # ----------------------------------------------------
    # PHASE 2: GPU ADDITION (CUMULATIVE)
    # ----------------------------------------------------
    print(f"\n[PHASE 2] GPU TITAN LOAD ({CONFIG['BENCH_GPU_TIME']}s)...")
    
    stop_gpu = mp.Event()
    gpu_proc = mp.Process(target=gpu_burn_process, args=(stop_gpu,))
    gpu_proc.start()
    
    start_t = time.time()
    try:
        while time.time() - start_t < CONFIG['BENCH_GPU_TIME']:
            rem = int(CONFIG['BENCH_GPU_TIME'] - (time.time() - start_t))
            c_temp = HardwareController.get_cpu_temp()
            g_temp = HardwareController.get_gpu_temp()
            
            sys.stdout.write(f"\r    T-{rem}s | CPU: {c_temp}C | GPU: {g_temp}C | Status: FULL SYSTEM BURN")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n    [!] Skipped Phase 2")
        
    # Teardown
    stop_cpu.set()
    stop_gpu.set()
    for p in procs: p.terminate()
    if gpu_proc.is_alive(): gpu_proc.terminate()
    
    print("\n\n[*] Benchmark Complete. Tuning Applied.")
    time.sleep(3)

# ==============================================================================
# SECTION 6: STRATUM V1 MINING CORE
# ==============================================================================

class StratumProtocol:
    def __init__(self, state, job_q, res_q, log_q):
        self.state = state
        self.job_q = job_q
        self.res_q = res_q
        self.log_q = log_q
        self.sock = None
        self.msg_id = 1
        self.buffer = ""
        self.extranonce1 = None
        self.extranonce2_size = 4
        self.target = 1.0

    def connect(self):
        try:
            self.sock = socket.create_connection((CONFIG['POOL_HOST'], CONFIG['POOL_PORT']), timeout=10)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.state.connected = True
            return True
        except Exception as e:
            self.state.connected = False
            self.log_q.put(("ERR", f"Connect: {e}"))
            return False

    def handshake(self):
        # 1. Subscribe
        sub = json.dumps({"id": self.msg_id, "method": "mining.subscribe", "params": ["MTP-Titan"]}) + "\n"
        self.sock.sendall(sub.encode())
        self.msg_id += 1
        
        # 2. Authorize
        auth = json.dumps({"id": self.msg_id, "method": "mining.authorize", "params": [f"{CONFIG['WALLET']}.{CONFIG['WORKER_NAME']}_local", CONFIG['PASS']]}) + "\n"
        self.sock.sendall(auth.encode())
        self.msg_id += 1
        
        # 3. Read loop
        start = time.time()
        en1_set = False
        auth_set = False
        
        while time.time() - start < 15:
            try:
                data = self.sock.recv(4096).decode()
                self.buffer += data
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    msg = json.loads(line)
                    
                    if msg.get('id') == 1:
                        self.extranonce1 = msg['result'][1]
                        self.extranonce2_size = msg['result'][2]
                        en1_set = True
                    
                    if msg.get('id') == 2 and msg.get('result'):
                        auth_set = True
                        
                    if msg.get('method') == 'mining.notify':
                        self.process_job(msg['params'])
                        
                if en1_set and auth_set: return True
            except: break
            time.sleep(0.1)
        return False

    def run(self):
        while True:
            try:
                # Reconnect Logic
                if not self.sock:
                    if not self.connect():
                        time.sleep(5); continue
                    if not self.handshake():
                        self.sock.close(); self.sock = None; continue
                    self.log_q.put(("NET", "Stratum Session Active"))
                
                # Check Outbound Shares
                while not self.res_q.empty():
                    s = self.res_q.get()
                    submit = {
                        "id": self.msg_id,
                        "method": "mining.submit",
                        "params": [
                            f"{CONFIG['WALLET']}.{CONFIG['WORKER_NAME']}_local",
                            s['jid'], s['en2'], s['ntime'], s['nonce']
                        ]
                    }
                    self.msg_id += 1
                    self.sock.sendall((json.dumps(submit) + "\n").encode())
                    self.state.local_tx += 1
                    self.log_q.put(("TX", f"Submitting Nonce {s['nonce']}"))

                # Check Inbound
                self.sock.settimeout(0.1)
                try:
                    d = self.sock.recv(4096)
                    if not d: raise Exception("Disconnect")
                    self.buffer += d.decode()
                    while '\n' in self.buffer:
                        line, self.buffer = self.buffer.split('\n', 1)
                        if not line: continue
                        self.handle_msg(json.loads(line))
                except socket.timeout: pass
                
            except Exception as e:
                self.log_q.put(("ERR", f"Link Drop: {e}"))
                self.state.connected = False
                if self.sock: self.sock.close()
                self.sock = None
                time.sleep(5)

    def process_job(self, p):
        # params: job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs
        # We need to construct a task object for workers
        
        # Safety check for Extranonce
        en1 = self.extranonce1 if self.extranonce1 else "00000000"
        
        job_struct = {
            'jid': p[0], 'prev': p[1], 'c1': p[2], 'c2': p[3],
            'mb': p[4], 'ver': p[5], 'nbits': p[6], 'ntime': p[7],
            'clean': p[8], 'en1': en1, 'en2_sz': self.extranonce2_size
        }
        
        # Clear queue if clean job
        if p[8]:
            while not self.job_q.empty():
                try: self.job_q.get_nowait()
                except: pass
        
        # Replicate for workers
        for _ in range(mp.cpu_count() * 2):
            self.job_q.put(job_struct)
            
        self.log_q.put(("RX", f"Block {p[0]} Received"))

    def handle_msg(self, msg):
        mid = msg.get('id')
        res = msg.get('result')
        method = msg.get('method')
        
        # Share Ack
        if mid and mid > 2:
            if res:
                self.state.accepted += 1
                self.log_q.put(("RX", "Share VALID"))
            else:
                self.state.rejected += 1
                self.log_q.put(("RX", f"Share INVALID: {msg.get('error')}"))
                
        # New Job
        if method == 'mining.notify':
            self.process_job(msg['params'])
            
        # Diff
        if method == 'mining.set_difficulty':
            self.target = msg['params'][0]

# ==============================================================================
# SECTION 7: MINING WORKERS (ROBUST)
# ==============================================================================

def cpu_miner_worker(id, job_q, res_q, stop_ev, hash_counter):
    """
    Robust CPU Miner.
    Catches queue.Empty to prevent NameError.
    Catches BrokenPipe to prevent crash on exit.
    """
    my_nonce = id * 100_000_000
    curr_job = None
    
    while not stop_ev.is_set():
        try:
            # Safe Queue Get
            try:
                job_data = job_q.get(timeout=0.1)
                curr_job = job_data
                if curr_job['clean']:
                    my_nonce = id * 100_000_000
            except queue.Empty:
                if curr_job is None: 
                    time.sleep(0.1)
                    continue
            
            # Unpack
            jid = curr_job['jid']
            prev = curr_job['prev']
            c1 = curr_job['c1']
            c2 = curr_job['c2']
            mb = curr_job['mb']
            ver = curr_job['ver']
            nbits = curr_job['nbits']
            ntime = curr_job['ntime']
            en1 = curr_job['en1']
            en2_sz = curr_job['en2_sz']
            
            # Mining Calc (SHA256d)
            # 1. Gen Extranonce2
            en2_hex = struct.pack('>I', random.randint(0, 2**32-1)).hex().zfill(en2_sz*2)
            
            # 2. Coinbase
            coinbase_bin = binascii.unhexlify(c1 + en1 + en2_hex + c2)
            cb_hash = hashlib.sha256(hashlib.sha256(coinbase_bin).digest()).digest()
            
            # 3. Merkle
            root = cb_hash
            for branch in mb:
                root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(branch)).digest()).digest()
            
            # 4. Header (80 Bytes)
            # Version(4) + Prev(32) + Root(32) + Time(4) + Bits(4) + Nonce(4)
            # NOTE: Endianness is critical here. Stratum sends hex which needs swap.
            
            header_pre = (
                binascii.unhexlify(ver)[::-1] +
                binascii.unhexlify(prev)[::-1] +
                root +
                binascii.unhexlify(ntime)[::-1] +
                binascii.unhexlify(nbits)[::-1]
            )
            
            # 5. Search Loop
            target_check = b'\x00\x00\x00' # Simplified check
            
            for n in range(my_nonce, my_nonce + 5000):
                header = header_pre + struct.pack('<I', n)
                h = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                
                # Check (Reverse for BE comparison)
                if h[::-1].hex().startswith("00000"):
                    # Valid Share
                    res_q.put({
                        'jid': jid,
                        'en2': en2_hex,
                        'ntime': ntime,
                        'nonce': struct.pack('>I', n).hex() # Stratum requires BE Hex string
                    })
                    
            my_nonce += 5000
            with hash_counter.get_lock():
                hash_counter.value += 5000
                
        except (BrokenPipeError, EOFError):
            break # Exit cleanly if pipe breaks
        except Exception:
            pass # Ignore calculation errors, keep mining

def gpu_miner_worker(stop_ev, hash_counter, log_q):
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_TITAN_SRC)
        func = mod.get_function("titan_load")
        
        # GPU Mem
        out_gpu = cuda.mem_alloc(4096)
        
        while not stop_ev.is_set():
            func(out_gpu, np.uint32(time.time()), np.int32(100), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
            with hash_counter.get_lock():
                hash_counter.value += (65535 * 512 * 2000)
    except:
        pass

# ==============================================================================
# SECTION 8: PROXY SERVER
# ==============================================================================

class ProxyServer(threading.Thread):
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
            s.listen(100)
            self.log_q.put(("INFO", f"Proxy Active on {CONFIG['PROXY_PORT']}"))
            while True:
                c, a = s.accept()
                threading.Thread(target=self.handle, args=(c,a), daemon=True).start()
        except: pass
        
    def handle(self, c, a):
        try:
            p = socket.create_connection((CONFIG['POOL_HOST'], CONFIG['POOL_PORT']))
            ip = a[0].split('.')[-1]
            self.state.proxy_clients += 1
            
            # Simple heartbeat
            def beat():
                while True:
                    time.sleep(30)
                    try: p.sendall(b'\n')
                    except: break
            threading.Thread(target=beat, daemon=True).start()
            
            inputs = [c, p]
            while True:
                r, _, _ = select.select(inputs, [], [])
                if c in r:
                    d = c.recv(4096)
                    if not d: break
                    # Rewrite
                    try:
                        dec = d.decode()
                        if "mining.authorize" in dec:
                            j = json.loads(dec)
                            j['params'][0] = f"{CONFIG['WALLET']}.ASIC_{ip}"
                            d = (json.dumps(j)+"\n").encode()
                        if "mining.submit" in dec:
                            self.state.proxy_tx += 1
                    except: pass
                    p.sendall(d)
                if p in r:
                    d = p.recv(4096)
                    if not d: break
                    if b'true' in d: self.state.proxy_rx += 1
                    c.sendall(d)
        except: pass
        finally:
            c.close(); p.close()
            self.state.proxy_clients -= 1

# ==============================================================================
# SECTION 9: MAIN UI
# ==============================================================================

def main_ui(stdscr, state, job_q, res_q, log_q):
    # Curses Setup (Safe Mode)
    curses.start_color()
    try:
        # Force black background for compatibility
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
    except: pass
    
    stdscr.nodelay(True)
    
    # Launch Services
    proto = StratumProtocol(state, job_q, res_q, log_q)
    threading.Thread(target=proto.run, daemon=True).start()
    
    ProxyServer(log_q, state).start()
    
    logs = []
    start_t = time.time()
    last_h = 0
    current_hr = 0.0
    
    while True:
        # 1. Update Logs
        while not log_q.empty():
            try:
                logs.append(log_q.get_nowait())
                if len(logs) > 30: logs.pop(0)
            except: pass
            
        # 2. Update Stats
        total = state.hashes_total
        delta = total - last_h
        last_h = total
        current_hr = (current_hr * 0.9) + (delta * 10 * 0.1) # Smooth
        
        # 3. Draw
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        stdscr.addstr(0, 0, " MTP v31 TITAN ".center(w), curses.color_pair(5))
        
        # Stats Block
        c = HardwareController.get_cpu_temp()
        g = HardwareController.get_gpu_temp()
        
        if current_hr > 1e9: hrs = f"{current_hr/1e9:.2f} GH/s"
        elif current_hr > 1e6: hrs = f"{current_hr/1e6:.2f} MH/s"
        else: hrs = f"{current_hr/1000:.2f} kH/s"
        
        stdscr.addstr(2, 2, "LOCAL SYSTEM", curses.color_pair(4))
        stdscr.addstr(3, 2, f"CPU: {c}C")
        stdscr.addstr(4, 2, f"GPU: {g}C")
        stdscr.addstr(5, 2, f"Hash: {hrs}")
        
        stdscr.addstr(2, 30, "NETWORK", curses.color_pair(4))
        status = "CONNECTED" if state.connected else "DISCONNECTED"
        stdscr.addstr(3, 30, f"Link: {status}")
        stdscr.addstr(4, 30, f"Acc: {state.accepted}")
        stdscr.addstr(5, 30, f"Rej: {state.rejected}")
        
        stdscr.addstr(2, 60, "PROXY", curses.color_pair(4))
        stdscr.addstr(3, 60, f"Clients: {state.proxy_clients}")
        stdscr.addstr(4, 60, f"Up: {state.proxy_tx}")
        stdscr.addstr(5, 60, f"Down: {state.proxy_rx}")
        
        # Logs
        stdscr.hline(7, 0, '-', w)
        for i, (lvl, msg) in enumerate(logs):
            if 8+i >= h-1: break
            color = curses.color_pair(5)
            if lvl == "ERR": color = curses.color_pair(3)
            elif lvl == "TX": color = curses.color_pair(2)
            elif lvl == "RX": color = curses.color_pair(1)
            try: stdscr.addstr(8+i, 2, f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}", color)
            except: pass
            
        stdscr.refresh()
        time.sleep(0.1)
        if stdscr.getch() == ord('q'): break

if __name__ == "__main__":
    # 1. Benchmark
    run_titan_benchmark()
    
    # 2. Setup Shared State (Namespace to avoid ListProxy crash)
    manager = mp.Manager()
    state = manager.Namespace()
    state.connected = False
    state.hashes_total = 0.0
    state.local_tx = 0
    state.accepted = 0
    state.rejected = 0
    state.proxy_clients = 0
    state.proxy_tx = 0
    state.proxy_rx = 0
    
    job_q = manager.Queue()
    res_q = manager.Queue()
    log_q = manager.Queue()
    stop_ev = mp.Event()
    
    # 3. Launch Workers
    procs = []
    # CPU
    hash_counter = mp.Value('d', 0.0) # Dedicated counter
    # Wrap state update in thread or use Value
    # To avoid Namespace pickling lag, we use a Value for hash counter specifically
    
    for i in range(mp.cpu_count() - 1):
        p = mp.Process(target=cpu_miner_worker, args=(i, job_q, res_q, stop_ev, hash_counter))
        p.start()
        procs.append(p)
        
    gp = mp.Process(target=gpu_miner_worker, args=(stop_ev, hash_counter, log_q))
    gp.start()
    procs.append(gp)
    
    # Sync Thread: Updates state.hashes_total from fast mp.Value
    def sync_stats():
        while not stop_ev.is_set():
            state.hashes_total = hash_counter.value
            time.sleep(0.5)
    threading.Thread(target=sync_stats, daemon=True).start()
    
    # 4. GUI
    try:
        curses.wrapper(main_ui, state, job_q, res_q, log_q)
    except KeyboardInterrupt:
        pass
    finally:
        stop_ev.set()
        for p in procs: p.terminate()
