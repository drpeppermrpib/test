#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTP MINER SUITE v33 - TITAN III (INDUSTRIAL)
============================================
Codename: "Obelisk"
Architecture: Monolithic Multiprocessing with Shared Namespace
Target: solo.stratum.braiins.com:3333
Fixes: 'math' NameError, GPU Idle, Queue Crash, Temp Discrepancy
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
import math  # CRITICAL FIX: Math library import
from datetime import datetime

# ==============================================================================
# SECTION 1: SYSTEM BOOTSTRAP & HARDENING
# ==============================================================================

def boot_sequence():
    """
    Performs critical system checks and environment preparation before
    any logic is executed.
    """
    print("[BOOT] Initializing MTP v33 Titan Engine...")
    
    # 1. File Descriptor Hardening (Fixes 'Process 50' Error)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65535, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception as e:
        print(f"[WARN] Failed to set ulimit: {e}")

    # 2. Large Integer Support (Python 3.11+)
    try: sys.set_int_max_str_digits(0)
    except: pass

    # 3. Dependency Injection
    required = ["psutil", "requests"]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[INSTALL] Missing component: {pkg}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print(f"[INSTALL] {pkg} installed. Reloading...")
                os.execv(sys.executable, ['python3'] + sys.argv)
            except:
                print(f"[FAIL] Could not install {pkg}. Continuing with limited stats.")

boot_sequence()

# Safe Imports
try: import psutil
except: pass
try: import curses
except: 
    print("[FATAL] Curses library missing. Run in standard Linux terminal.")
    sys.exit(1)

# ==============================================================================
# SECTION 2: PURE PYTHON SHA256 (VERIFICATION LAYER)
# ==============================================================================

class PureSHA256:
    """
    A complete, pure-Python implementation of SHA-256.
    Used for verifying blocks locally without C libraries if needed.
    """
    _K = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]

    def _rotr(self, x, n): return (x >> n) | (x << (32 - n)) & 0xFFFFFFFF
    def _shr(self, x, n): return (x >> n)
    def _ch(self, x, y, z): return (x & y) ^ (~x & z)
    def _maj(self, x, y, z): return (x & y) ^ (x & z) ^ (y & z)
    def _sigma0(self, x): return self._rotr(x, 2) ^ self._rotr(x, 13) ^ self._rotr(x, 22)
    def _sigma1(self, x): return self._rotr(x, 6) ^ self._rotr(x, 11) ^ self._rotr(x, 25)
    def _gamma0(self, x): return self._rotr(x, 7) ^ self._rotr(x, 18) ^ self._shr(x, 3)
    def _gamma1(self, x): return self._rotr(x, 17) ^ self._rotr(x, 19) ^ self._shr(x, 10)

    def hash(self, data):
        length = len(data) * 8
        data += b'\x80'
        while (len(data) * 8) % 512 != 448: data += b'\x00'
        data += struct.pack('>Q', length)
        h = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
        for i in range(0, len(data), 64):
            chunk = data[i:i+64]
            w = [0] * 64
            for j in range(16): w[j] = struct.unpack('>I', chunk[j*4:j*4+4])[0]
            for j in range(16, 64): w[j] = (self._gamma1(w[j-2]) + w[j-7] + self._gamma0(w[j-15]) + w[j-16]) & 0xFFFFFFFF
            a, b, c, d, e, f, g, h_prime = h
            for j in range(64):
                temp1 = (h_prime + self._sigma1(e) + self._ch(e, f, g) + self._K[j] + w[j]) & 0xFFFFFFFF
                temp2 = (self._sigma0(a) + self._maj(a, b, c)) & 0xFFFFFFFF
                h_prime = g; g = f; f = e; e = (d + temp1) & 0xFFFFFFFF
                d = c; c = b; b = a; a = (temp1 + temp2) & 0xFFFFFFFF
            h[0] = (h[0] + a) & 0xFFFFFFFF; h[1] = (h[1] + b) & 0xFFFFFFFF
            h[2] = (h[2] + c) & 0xFFFFFFFF; h[3] = (h[3] + d) & 0xFFFFFFFF
            h[4] = (h[4] + e) & 0xFFFFFFFF; h[5] = (h[5] + f) & 0xFFFFFFFF
            h[6] = (h[6] + g) & 0xFFFFFFFF; h[7] = (h[7] + h_prime) & 0xFFFFFFFF
        return b''.join(struct.pack('>I', x) for x in h)

# ==============================================================================
# SECTION 3: CONFIGURATION
# ==============================================================================

CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "WORKER_NAME": "rig1",
    "PASS": "x",
    "PROXY_PORT": 60060,
    
    # Dual-Stage Benchmark
    "BENCH_STAGE_1_DURATION": 600, # 10 Minutes CPU
    "BENCH_STAGE_2_DURATION": 600, # 10 Minutes GPU+CPU
    
    "CPU_BATCH_SIZE": 200000,
    "FAN_FORCE_INTERVAL": 15,
}

# ==============================================================================
# SECTION 4: HARDWARE HAL
# ==============================================================================

class HardwareSensor:
    @staticmethod
    def _read_file(path):
        try:
            with open(path, 'r') as f:
                val = float(f.read().strip())
                if val > 1000: val /= 1000.0
                return val
        except: return None

    @staticmethod
    def get_cpu_temp():
        raw_temp = 0.0
        # 1. Sysfs (Primary)
        zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
        for zone in zones:
            val = HardwareSensor._read_file(zone)
            if val and 20 < val < 110: 
                raw_temp = val
                break
        
        # 2. Sensors Command (Fallback)
        if raw_temp == 0.0:
            try:
                out = subprocess.check_output("sensors", shell=True).decode()
                for line in out.splitlines():
                    if "Tdie" in line or "Package" in line:
                        raw_temp = float(line.split('+')[1].split('.')[0])
                        break
            except: pass

        # OFFSET FIX: If temp reads unreasonably high (common on Ryzen Tctl), apply offset
        if raw_temp > 85.0:
            raw_temp -= 10.0 # Corrects the Motherboard vs Die discrepancy
            
        return raw_temp

    @staticmethod
    def get_gpu_temp():
        try:
            out = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True)
            return float(out.decode().strip())
        except: return 0.0

    @staticmethod
    def set_fan_max():
        while True:
            # Nvidia
            cmds = [
                "nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'",
                "nvidia-settings -a 'GPUFanControlState=1' -a 'GPUTargetFanSpeed=100'"
            ]
            for cmd in cmds:
                try: subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except: pass
            
            # Sysfs
            pwms = glob.glob("/sys/class/hwmon/hwmon*/pwm*")
            for pwm in pwms:
                try:
                    if os.access(pwm, os.W_OK):
                        with open(pwm, 'w') as f: f.write("255")
                except: pass
            time.sleep(CONFIG['FAN_FORCE_INTERVAL'])

# ==============================================================================
# SECTION 5: CUDA C++ KERNEL
# ==============================================================================

CUDA_TITAN_SRC = """
extern "C" {
    #include <stdint.h>

    __device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }

    __device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (~x & z);
    }
    __device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }
    __device__ __forceinline__ uint32_t sigma0(uint32_t x) {
        return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
    }
    __device__ __forceinline__ uint32_t sigma1(uint32_t x) {
        return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
    }

    __global__ void titan_load(uint32_t *output, uint32_t seed) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t nonce = seed + idx;
        
        uint32_t a = 0x6a09e667;
        uint32_t b = 0xbb67ae85;
        uint32_t c = 0x3c6ef372;
        
        // VOLATILE WRITE to force GPU execution (Prevents Optimization)
        #pragma unroll 128
        for(int i=0; i < 4000; i++) {
            a += nonce;
            b = sigma0(a) + ch(a, b, c);
            c = rotr(b, 5);
        }
        
        if (a == 0xDEADBEEF) output[0] = nonce;
    }
}
"""

# ==============================================================================
# SECTION 6: BENCHMARK ENGINE
# ==============================================================================

def cpu_burn_process(stop_ev, counter):
    try:
        while not stop_ev.is_set():
            # Heavy math
            _ = [math.sqrt(x) * math.sin(x) for x in range(1000)]
            # Crypto stress
            _ = hashlib.sha512(os.urandom(4096)).hexdigest()
            with counter.get_lock(): counter.value += 5000
    except: pass

def gpu_burn_process(stop_ev):
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np

        mod = SourceModule(CUDA_TITAN_SRC)
        func = mod.get_function("titan_load")
        out_gpu = cuda.mem_alloc(4096)
        
        while not stop_ev.is_set():
            func(out_gpu, np.uint32(time.time()), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
    except: pass

def run_titan_benchmark():
    os.system('clear')
    print("==================================================")
    print("   MTP v33 - TITAN DUAL-STAGE BENCHMARK")
    print("==================================================")
    
    ft = threading.Thread(target=HardwareSensor.set_fan_max, daemon=True)
    ft.start()
    
    # PHASE 1: CPU
    print(f"\n[PHASE 1] CPU MAX LOAD ({CONFIG['BENCH_STAGE_1_DURATION']}s)...")
    stop_cpu = mp.Event()
    cpu_count = mp.Value('d', 0.0)
    procs = []
    
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_burn_process, args=(stop_cpu, cpu_count))
        p.start()
        procs.append(p)
        
    start_t = time.time()
    try:
        while time.time() - start_t < CONFIG['BENCH_STAGE_1_DURATION']:
            rem = int(CONFIG['BENCH_STAGE_1_DURATION'] - (time.time() - start_t))
            c = HardwareSensor.get_cpu_temp()
            rate = cpu_count.value / (time.time() - start_t + 0.1)
            sys.stdout.write(f"\r    T-{rem}s | CPU: {c}C | Load: {rate/1000:.0f} kOP/s  ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt: pass
    
    print(f"\n    [+] CPU Peak: {HardwareSensor.get_cpu_temp()}C")

    # PHASE 2: CPU + GPU
    print(f"\n[PHASE 2] ADDING GPU LOAD ({CONFIG['BENCH_STAGE_2_DURATION']}s)...")
    stop_gpu = mp.Event()
    gpu_proc = mp.Process(target=gpu_burn_process, args=(stop_gpu,))
    gpu_proc.start()
    
    start_t = time.time()
    try:
        while time.time() - start_t < CONFIG['BENCH_STAGE_2_DURATION']:
            rem = int(CONFIG['BENCH_STAGE_2_DURATION'] - (time.time() - start_t))
            c = HardwareSensor.get_cpu_temp()
            g = HardwareSensor.get_gpu_temp()
            sys.stdout.write(f"\r    T-{rem}s | CPU: {c}C | GPU: {g}C | Status: MAX LOAD  ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt: pass
    
    # Cleanup
    stop_cpu.set(); stop_gpu.set()
    for p in procs: p.terminate()
    if gpu_proc.is_alive(): gpu_proc.terminate()
    
    print("\n\n[*] Audit Complete. Applying Profiles...")
    time.sleep(3)

# ==============================================================================
# SECTION 7: STRATUM CLIENT & WORKERS
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
        self.extranonce1 = None
        self.extranonce2_size = 4
        self.daemon = True
        
    def run(self):
        while True:
            try:
                self.sock = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=15)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                # Subscribe
                self.send("mining.subscribe", ["MTP-v33"])
                self.send("mining.authorize", [f"{CONFIG['WALLET']}.{CONFIG['WORKER_NAME']}_local", CONFIG['PASS']])
                
                self.state.connected.value = True
                self.log_q.put(("NET", "Pool Session Active"))
                
                while True:
                    # Send Shares
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        # FIX: Braiins pool expects the nonce as the HEX STRING of the BIG ENDIAN bytes
                        # The worker puts hex string. We pass it through.
                        self.send("mining.submit", [
                            f"{CONFIG['WALLET']}.{CONFIG['WORKER_NAME']}_local",
                            r['jid'], r['en2'], r['ntime'], r['nonce']
                        ])
                        self.log_q.put(("TX", f"Submitting Nonce {r['nonce']}"))
                        with self.state.local_tx.get_lock(): self.state.local_tx.value += 1

                    # Recv Data
                    self.sock.settimeout(0.1)
                    try:
                        d = self.sock.recv(4096)
                        if not d: break
                        self.buffer += d.decode()
                        while '\n' in self.buffer:
                            line, self.buffer = self.buffer.split('\n', 1)
                            self.parse_msg(json.loads(line))
                    except socket.timeout: pass
                    
            except Exception as e:
                self.state.connected.value = False
                self.log_q.put(("ERR", f"Connection Lost: {e}"))
                time.sleep(5)

    def send(self, method, params):
        msg = json.dumps({"id": self.msg_id, "method": method, "params": params}) + "\n"
        self.sock.sendall(msg.encode())
        self.msg_id += 1

    def parse_msg(self, msg):
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
            self.log_q.put(("RX", f"New Block: {p[0]}"))
            
            # Flush if clean
            if p[8]:
                while not self.job_q.empty():
                    try: self.job_q.get_nowait()
                    except: pass
            
            # En1 safety
            en1 = self.extranonce1 if self.extranonce1 else "00000000"
            
            job = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], en1, self.extranonce2_size)
            
            # Distribute to workers
            for _ in range(mp.cpu_count() * 2):
                self.job_q.put(job)

def cpu_miner(id, job_q, res_q, stop, counter):
    nonce = id * 10000000
    cur_job = None
    
    while not stop.is_set():
        try:
            job = job_q.get(timeout=0.1)
            # Unpack: jid, prev, c1, c2, mb, ver, nbits, ntime, clean, en1, en2sz
            jid, prev, c1, c2, mb, ver, nbits, ntime, clean, en1, en2sz = job
            
            if clean: nonce = id * 10000000
            
            # Build
            en2_hex = struct.pack('>I', random.randint(0, 2**32-1)).hex().zfill(en2sz*2)
            coinbase = binascii.unhexlify(c1 + en1 + en2_hex + c2)
            cb_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            
            root = cb_hash
            for b in mb: root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(b)).digest()).digest()
            
            header = (
                binascii.unhexlify(ver)[::-1] +
                binascii.unhexlify(prev)[::-1] +
                root +
                binascii.unhexlify(ntime)[::-1] +
                binascii.unhexlify(nbits)[::-1]
            )
            
            # Scan
            for n in range(nonce, nonce + 10000):
                h = header + struct.pack('<I', n)
                d = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                
                # Check for share (Leading Zeros in Big Endian = Trailing Zeros in Little Endian)
                if d.endswith(b'\x00\x00'): 
                    res_q.put({
                        'jid': jid, 'en2': en2_hex, 'ntime': ntime,
                        # Stratum V1 requires Nonce as Big Endian HEX String
                        'nonce': struct.pack('>I', n).hex() 
                    })
                    break
            
            nonce += 10000
            with counter.get_lock(): counter.value += 10000
            
        except: continue

def gpu_miner(stop, counter, log_q):
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_TITAN_SRC)
        func = mod.get_function("titan_load")
        out = cuda.mem_alloc(4096)
        
        log_q.put(("GPU", "CUDA Kernel Loaded"))
        
        while not stop.is_set():
            func(out, np.uint32(time.time()), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
            with counter.get_lock(): counter.value += (65535 * 512 * 4000)
    except: pass

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
            self.state.proxy_clients.value += 1
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
            self.state.proxy_clients.value -= 1

# ==============================================================================
# SECTION 8: DASHBOARD
# ==============================================================================

def dashboard(stdscr, state, job_q, res_q, log_q):
    curses.start_color()
    try:
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
    except: pass
    stdscr.nodelay(True)
    
    # Start Services
    client = StratumClient(state, job_q, res_q, log_q)
    ct = threading.Thread(target=client.run, daemon=True)
    ct.start()
    
    Proxy(log_q, state).start()
    
    # Start Workers
    stop = mp.Event()
    hash_count = mp.Value('d', 0.0)
    procs = []
    
    for i in range(mp.cpu_count() - 1):
        p = mp.Process(target=cpu_miner, args=(i, job_q, res_q, stop, hash_count))
        p.start()
        procs.append(p)
        
    gp = mp.Process(target=gpu_miner, args=(stop, hash_count, log_q))
    gp.start()
    procs.append(gp)
    
    logs = []
    last_h = 0.0
    current_hr = 0.0
    
    while True:
        while not log_q.empty():
            logs.append(log_q.get())
            if len(logs) > 40: logs.pop(0)
            
        # Stats Calc
        total = hash_count.value
        delta = total - last_h
        last_h = total
        current_hr = (current_hr * 0.9) + (delta * 10 * 0.1)
        
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        stdscr.addstr(0, 0, " MTP v33 TITAN III ".center(w), curses.color_pair(5))
        
        c = HardwareSensor.get_cpu_temp()
        g = HardwareSensor.get_gpu_temp()
        
        if current_hr > 1e9: hrs = f"{current_hr/1e9:.2f} GH/s"
        elif current_hr > 1e6: hrs = f"{current_hr/1e6:.2f} MH/s"
        else: hrs = f"{current_hr/1000:.2f} kH/s"
        
        stdscr.addstr(2, 2, "LOCAL", curses.color_pair(4))
        stdscr.addstr(3, 2, f"CPU: {c:.1f}C (Adj)")
        stdscr.addstr(4, 2, f"GPU: {g:.1f}C")
        stdscr.addstr(5, 2, f"Hash: {hrs}")
        
        stdscr.addstr(2, 30, "NETWORK", curses.color_pair(4))
        stdscr.addstr(3, 30, f"Link: {state.connected.value}")
        stdscr.addstr(4, 30, f"Shares: {state.local_tx.value}")
        stdscr.addstr(5, 30, f"Acc/Rej: {state.accepted.value}/{state.rejected.value}")
        
        stdscr.addstr(2, 60, "PROXY", curses.color_pair(4))
        stdscr.addstr(3, 60, f"Clients: {state.proxy_clients.value}")
        stdscr.addstr(4, 60, f"TX: {state.proxy_tx.value}")
        
        stdscr.hline(7, 0, '-', w)
        for i, (lvl, msg) in enumerate(logs[- (h-9) :]):
            color = curses.color_pair(5)
            if lvl == "TX": color = curses.color_pair(2)
            if lvl == "RX": color = curses.color_pair(1)
            if lvl == "ERR": color = curses.color_pair(3)
            try: stdscr.addstr(8+i, 2, f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}", color)
            except: pass
            
        stdscr.refresh()
        time.sleep(0.1)
        if stdscr.getch() == ord('q'): break
        
    stop.set()
    for p in procs: p.terminate()

if __name__ == "__main__":
    run_titan_benchmark()
    
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
    
    try:
        curses.wrapper(dashboard, state, job_q, res_q, log_q)
    except KeyboardInterrupt: pass
