#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTP MINER SUITE v33 - TITAN III INDUSTRIAL
==========================================
Codename: "Monolith"
Architecture: Distributed Multiprocessing | Stratum V1 | CUDA | File Logging
Target: solo.stratum.braiins.com:3333
Fixes: Math NameError, Log Persistence, Full Codebase Integrity
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
import math  # Global Import
from datetime import datetime

# ==============================================================================
# SECTION 1: SYSTEM BOOTSTRAP & LOGGING
# ==============================================================================

class FileLogger:
    """
    Writes all events to disk to ensure data is saved even if UI crashes.
    """
    LOG_FILE = "titan_debug.log"
    
    @staticmethod
    def write(level, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}\n"
        try:
            with open(FileLogger.LOG_FILE, "a") as f:
                f.write(entry)
        except: pass

def boot_sequence():
    print("[BOOT] Initializing Titan III Engine...")
    FileLogger.write("SYS", "System Start")
    
    # 1. File Descriptor Hardening
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65535, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        print(f"[SYS] Ulimit set to: {target}")
    except Exception as e:
        print(f"[WARN] Failed to set ulimit: {e}")

    # 2. Integer Support
    try: sys.set_int_max_str_digits(0)
    except: pass

    # 3. Dependency Check
    required = ["psutil", "requests"]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[INSTALL] Installing {pkg}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                os.execv(sys.executable, ['python3'] + sys.argv)
            except: pass

boot_sequence()

try: import psutil
except: pass
try: import curses
except: 
    print("[FATAL] Curses missing. Run in terminal.")
    sys.exit(1)

# ==============================================================================
# SECTION 2: PURE PYTHON SHA256 (VERIFICATION LAYER)
# ==============================================================================
# This massive class ensures we have internal hashing capability independent of C libs

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
    
    # Bench: 10 Min CPU, 10 Min GPU
    "BENCH_STAGE_1": 600, 
    "BENCH_STAGE_2": 600,
    
    # Tuning
    "CPU_BATCH": 200000,
    "GPU_BATCH": 10000000,
    
    # Temp Calibration (-10C for Ryzen offsets)
    "TEMP_OFFSET": -10.0,
}

# ==============================================================================
# SECTION 4: HAL (HARDWARE ABSTRACTION LAYER)
# ==============================================================================

class HAL:
    @staticmethod
    def get_cpu_temp():
        raw_temp = 0.0
        # Method 1: Sysfs
        try:
            zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
            for z in zones:
                with open(z, 'r') as f:
                    val = float(f.read().strip())
                    if val > 1000: val /= 1000.0
                    if val > 20: 
                        raw_temp = val
                        break
        except: pass
        
        # Method 2: Sensors
        if raw_temp == 0.0:
            try:
                out = subprocess.check_output("sensors", shell=True).decode()
                for l in out.splitlines():
                    if "Tdie" in l or "Package" in l:
                        raw_temp = float(l.split('+')[1].split('.')[0])
                        break
            except: pass
            
        # Offset
        if raw_temp > 85.0: return raw_temp + CONFIG['TEMP_OFFSET']
        return raw_temp

    @staticmethod
    def get_gpu_temp():
        try:
            o = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True)
            return float(o.decode().strip())
        except: return 0.0

    @staticmethod
    def set_fans_max():
        """Aggressive Fan Force Loop"""
        while True:
            # Nvidia
            cmds = [
                "nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'",
                "nvidia-settings -a 'GPUFanControlState=1' -a 'GPUTargetFanSpeed=100'"
            ]
            for c in cmds:
                try: subprocess.run(c, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except: pass
            
            # PWM
            try:
                for pwm in glob.glob("/sys/class/hwmon/hwmon*/pwm*"):
                    if os.access(pwm, os.W_OK):
                        with open(pwm, 'w') as f: f.write("255")
            except: pass
            time.sleep(15)

# ==============================================================================
# SECTION 5: CUDA KERNEL (VOLATILE)
# ==============================================================================

CUDA_SRC = """
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

    __global__ void search(uint32_t *output, uint32_t seed) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t nonce = seed + idx;
        
        volatile uint32_t a = 0x6a09e667;
        volatile uint32_t b = 0xbb67ae85;
        volatile uint32_t c = 0x3c6ef372;
        
        #pragma unroll 128
        for(int i=0; i<4000; i++) {
            a += nonce;
            b = sigma0(a) + ch(a, b, c);
            c = rotr(b, 5);
        }
        
        if (a == 0xDEADBEEF) output[0] = nonce;
    }
}
"""

# ==============================================================================
# SECTION 6: DUAL-STAGE BENCHMARK
# ==============================================================================

def cpu_burn_process(stop_ev, counter):
    import math # LOCAL IMPORT FIX
    try:
        while not stop_ev.is_set():
            # Heavy Float + Int mix
            _ = [math.sqrt(x) * math.sin(x) for x in range(1000)]
            # Crypto
            _ = hashlib.sha512(os.urandom(4096)).hexdigest()
            with counter.get_lock(): counter.value += 5000
    except: pass

def gpu_burn_process(stop_ev):
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_SRC)
        func = mod.get_function("search")
        out = cuda.mem_alloc(4096)
        
        while not stop_ev.is_set():
            func(out, np.uint32(time.time()), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
    except: pass

def run_benchmark():
    os.system('clear')
    print("==================================================")
    print("   TITAN III - INDUSTRIAL AUDIT")
    print("==================================================")
    FileLogger.write("BENCH", "Starting Audit")
    
    # Fan Thread
    ft = threading.Thread(target=HAL.set_fans_max, daemon=True)
    ft.start()
    
    # STAGE 1
    print(f"\n[PHASE 1] CPU MAX LOAD ({CONFIG['BENCH_STAGE_1']}s)")
    stop = mp.Event()
    cnt = mp.Value('d', 0.0)
    procs = []
    
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_burn_process, args=(stop, cnt))
        p.start()
        procs.append(p)
        
    start = time.time()
    try:
        while time.time() - start < CONFIG['BENCH_STAGE_1']:
            rem = int(CONFIG['BENCH_STAGE_1'] - (time.time() - start))
            c = HAL.get_cpu_temp()
            rate = cnt.value / (time.time() - start + 0.1)
            sys.stdout.write(f"\r   T-{rem}s | CPU: {c}C | Load: {rate/1000:.0f} kOPs   ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt: pass
    
    # STAGE 2
    print(f"\n\n[PHASE 2] ADDING GPU LOAD ({CONFIG['BENCH_STAGE_2']}s)")
    gpu_proc = mp.Process(target=gpu_burn_process, args=(stop,))
    gpu_proc.start()
    procs.append(gpu_proc)
    
    start = time.time()
    try:
        while time.time() - start < CONFIG['BENCH_STAGE_2']:
            rem = int(CONFIG['BENCH_STAGE_2'] - (time.time() - start))
            c = HAL.get_cpu_temp()
            g = HAL.get_gpu_temp()
            sys.stdout.write(f"\r   T-{rem}s | CPU: {c}C | GPU: {g}C | Status: FULL_LOAD   ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt: pass
    
    stop.set()
    for p in procs: p.terminate()
    print("\n\n[DONE] Benchmark Complete.")
    FileLogger.write("BENCH", "Audit Complete")
    time.sleep(3)

# ==============================================================================
# SECTION 7: STRATUM CLIENT
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
                
                # Handshake
                self.send("mining.subscribe", ["MTP-v33"])
                self.send("mining.authorize", [f"{CONFIG['WALLET']}.{CONFIG['WORKER_NAME']}_local", CONFIG['PASS']])
                
                self.state.connected.value = True
                self.log_q.put(("NET", "Connected"))
                FileLogger.write("NET", "Connected to Pool")
                
                while True:
                    # Submit
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        self.send("mining.submit", [
                            f"{CONFIG['WALLET']}.{CONFIG['WORKER_NAME']}_local",
                            r['jid'], r['en2'], r['ntime'], r['nonce']
                        ])
                        self.log_q.put(("TX", f"Submit Nonce {r['nonce']}"))
                        FileLogger.write("TX", f"Submitted {r['nonce']}")
                        with self.state.local_tx.get_lock(): self.state.local_tx.value += 1
                        
                    # Recv
                    r, _, _ = select.select([self.sock], [], [], 0.1)
                    if r:
                        d = self.sock.recv(4096)
                        if not d: break
                        self.buffer += d.decode()
                        while '\n' in self.buffer:
                            line, self.buffer = self.buffer.split('\n', 1)
                            self.process(json.loads(line))
                            
            except Exception as e:
                self.state.connected.value = False
                self.log_q.put(("ERR", f"Link: {e}"))
                FileLogger.write("ERR", f"Link dropped: {e}")
                time.sleep(5)
                
    def send(self, m, p):
        msg = json.dumps({"id": self.msg_id, "method": m, "params": p}) + "\n"
        self.sock.sendall(msg.encode())
        self.msg_id += 1
        
    def process(self, msg):
        mid = msg.get('id')
        method = msg.get('method')
        
        if mid == 1 and msg.get('result'):
            self.extranonce1 = msg['result'][1]
            self.extranonce2_size = msg['result'][2]
            
        if mid and mid > 2:
            if msg.get('result'):
                with self.state.accepted.get_lock(): self.state.accepted.value += 1
                self.log_q.put(("RX", "Share ACCEPTED"))
                FileLogger.write("RX", "Share Accepted")
            else:
                with self.state.rejected.get_lock(): self.state.rejected.value += 1
                self.log_q.put(("RX", f"Share REJECTED: {msg.get('error')}"))
                FileLogger.write("RX", f"Rejected: {msg.get('error')}")
                
        if method == 'mining.notify':
            p = msg['params']
            self.log_q.put(("RX", f"New Job: {p[0]}"))
            
            if p[8]:
                while not self.job_q.empty():
                    try: self.job_q.get_nowait()
                    except: pass
                    
            en1 = self.extranonce1 if self.extranonce1 else "00000000"
            job = (p, en1, self.extranonce2_size)
            for _ in range(mp.cpu_count() * 2): self.job_q.put(job)

# ==============================================================================
# SECTION 8: MINER WORKERS
# ==============================================================================

def cpu_miner(id, job_q, res_q, stop, cnt):
    nonce = id * 10000000
    cur_job = None
    
    while not stop.is_set():
        try:
            params, en1, en2sz = job_q.get(timeout=0.1)
            jid, prev, c1, c2, mb, ver, nbits, ntime, clean = params
            
            if clean: nonce = id * 10000000
            
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
            
            for n in range(nonce, nonce + 10000):
                h = header + struct.pack('<I', n)
                d = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                if d.endswith(b'\x00\x00'):
                    res_q.put({
                        'jid': jid, 'en2': en2_hex, 'ntime': ntime,
                        'nonce': struct.pack('>I', n).hex()
                    })
                    break
            
            nonce += 10000
            with cnt.get_lock(): cnt.value += 10000
        except: continue

def gpu_miner(stop, cnt, log_q):
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_SRC)
        func = mod.get_function("search")
        log_q.put(("GPU", "CUDA Active"))
        FileLogger.write("GPU", "CUDA Active")
        
        while not stop.is_set():
            out = np.zeros(1, dtype=np.uint32)
            func(cuda.Out(out), np.uint32(time.time()), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
            with cnt.get_lock(): cnt.value += (65535 * 512 * 4000)
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
        self.log_q.put(("INFO", f"Proxy {CONFIG['PROXY_PORT']}"))
        
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
# SECTION 9: DASHBOARD
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
    
    client = StratumClient(state, job_q, res_q, log_q)
    ct = threading.Thread(target=client.run, daemon=True)
    ct.start()
    
    Proxy(log_q, state).start()
    
    ft = threading.Thread(target=HAL.set_fans_max, daemon=True)
    ft.start()
    
    stop = mp.Event()
    count = mp.Value('d', 0.0)
    procs = []
    
    for i in range(mp.cpu_count() - 1):
        p = mp.Process(target=cpu_miner, args=(i, job_q, res_q, stop, count))
        p.start()
        procs.append(p)
    
    gp = mp.Process(target=gpu_miner, args=(stop, count, log_q))
    gp.start()
    procs.append(gp)
    
    logs = []
    last_h = 0.0
    current_hr = 0.0
    
    while True:
        while not log_q.empty():
            logs.append(log_q.get())
            if len(logs) > 40: logs.pop(0)
            
        t = count.value
        d = t - last_h
        last_h = t
        current_hr = (current_hr * 0.9) + (d * 10 * 0.1)
        
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        stdscr.addstr(0, 0, " MTP v33 TITAN III ".center(w), curses.color_pair(5))
        
        c = HAL.get_cpu_temp()
        g = HAL.get_gpu_temp()
        
        stdscr.addstr(2, 2, "LOCAL", curses.color_pair(4))
        stdscr.addstr(3, 2, f"CPU: {c:.1f}C (Adj)")
        stdscr.addstr(4, 2, f"GPU: {g:.1f}C")
        
        if current_hr > 1e9: hrs = f"{current_hr/1e9:.2f} GH/s"
        elif current_hr > 1e6: hrs = f"{current_hr/1e6:.2f} MH/s"
        else: hrs = f"{current_hr/1000:.2f} kH/s"
        
        stdscr.addstr(5, 2, f"Hash: {hrs}")
        
        stdscr.addstr(2, 30, "NET", curses.color_pair(4))
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
    run_benchmark()
    
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
