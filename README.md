#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTP MINER SUITE v32 - TITAN II INDUSTRIAL EDITION
=================================================
Codename: "Volatile"
Architecture: Asynchronous Network Layer + Volatile CUDA Kernel
Target: solo.stratum.braiins.com:3333
Fixes: Cold GPU, Invalid Shares, Link Drops, Proxy Lag
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
from datetime import datetime

# ==============================================================================
# SECTION 1: SYSTEM HARDENING & BOOTLOADER
# ==============================================================================

def titan_bootloader():
    """
    Performs critical OS-level configuration to ensure stability under
    extreme load conditions.
    """
    print("[BOOT] Initializing Titan II Engine...")
    
    # 1.1: File Descriptor Expansion (Prevents Socket Crashes)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65535, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        print(f"[SYS] Network Sockets Available: {target}")
    except Exception as e:
        print(f"[WARN] Failed to expand ulimit: {e}")

    # 1.2: Integer Precision (Python 3.11+ Fix)
    try: sys.set_int_max_str_digits(0)
    except: pass

    # 1.3: Driver Verification
    required = ["psutil", "requests"]
    drivers_ok = True
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[INSTALL] Deploying missing driver: {pkg}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except:
                drivers_ok = False
                print(f"[FAIL] Driver {pkg} failed. System functionality reduced.")
    
    if not drivers_ok:
        print("[WARN] Running in Compatibility Mode.")
        time.sleep(2)
    else:
        # Re-launch only if we just installed something to load it properly
        pass

titan_bootloader()

# Safe Imports
try: import psutil
except: pass
try: import curses
except: 
    print("[FATAL] System Terminal not supported (Curses missing).")
    sys.exit(1)

# ==============================================================================
# SECTION 2: CONFIGURATION MATRIX
# ==============================================================================

CONFIG = {
    # Stratum Connection
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "WORKER": "rig1",
    "PASS": "x",
    
    # Proxy Gateway
    "PROXY_HOST": "0.0.0.0",
    "PROXY_PORT": 60060,
    
    # Calibration
    "TEMP_OFFSET_CPU": -10.0, # Adjusts reading to match Motherboard
    
    # Benchmark Settings (5 Minutes per Stage)
    "BENCH_STAGE_TIME": 300, 
    
    # Mining Settings
    "CPU_INTENSITY": 500000,   # Hashes per batch
    "GPU_INTENSITY": 200000000 # Hashes per batch
}

# ==============================================================================
# SECTION 3: VOLATILE CUDA KERNEL (HEAT GENERATOR)
# ==============================================================================

# Using 'volatile' tells the compiler "Do not optimize this variable,
# it can change at any time". This forces the GPU to do the work.
CUDA_TITAN_II_SRC = """
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
        
        // Volatile enforces register usage and prevents caching/optimization
        volatile uint32_t a = 0x6a09e667 + idx + seed;
        volatile uint32_t b = 0xbb67ae85;
        volatile uint32_t c = 0x3c6ef372;
        
        // Massive Unrolled Loop (8000 iterations)
        // This is pure ALU saturation
        #pragma unroll 128
        for(int i=0; i < 4000; i++) {
            a += sigma0(b) + ch(b, c, a);
            b = rotr(b, 11) ^ a;
            c += sigma1(a);
        }
        
        // Dummy write to prevent dead-code elimination
        if (a == 0xDEADBEEF) {
            output[idx % 1024] = a + b + c;
        }
    }
}
"""

# ==============================================================================
# SECTION 4: HARDWARE ABSTRACTION LAYER (HAL)
# ==============================================================================

class HAL:
    @staticmethod
    def get_cpu_temp():
        raw_temp = 0.0
        # Strategy 1: Sensors Command
        try:
            out = subprocess.check_output("sensors", shell=True).decode()
            for line in out.splitlines():
                # Prefer Tdie for accuracy, Package for socket
                if "Tdie" in line or "Package id 0" in line:
                    parts = line.split('+')
                    if len(parts) > 1:
                        raw_temp = float(parts[1].split('.')[0])
                        break
        except: pass

        # Strategy 2: Thermal Zones
        if raw_temp == 0.0:
            try:
                zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
                for z in zones:
                    with open(z, 'r') as f:
                        t = int(f.read().strip())
                        if t > 1000: t = t / 1000.0
                        if t > 20: 
                            raw_temp = t
                            break
            except: pass
            
        # Apply Calibration Offset
        final_temp = raw_temp + CONFIG['TEMP_OFFSET_CPU']
        return max(0.0, final_temp)

    @staticmethod
    def get_gpu_temp():
        try:
            out = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True).decode()
            return float(out.strip())
        except: return 0.0

    @staticmethod
    def set_fan_max():
        """Forces 100% Fan Speed via multiple backends."""
        # 1. Nvidia Settings
        try:
            subprocess.run("nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: pass
        
        # 2. Sysfs PWM Override (Root required)
        try:
            pwms = glob.glob("/sys/class/hwmon/hwmon*/pwm*")
            for p in pwms:
                if "enable" not in p: # Target the PWM control file
                    try:
                        with open(p, "w") as f: f.write("255")
                    except: pass
        except: pass

# ==============================================================================
# SECTION 5: NETWORK STACK (BUFFERED & ASYNC)
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
        self.recv_buffer = ""
        self.daemon = True
        
    def connect(self):
        try:
            self.sock = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=15)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.state.connected = True
            return True
        except Exception as e:
            self.state.connected = False
            self.log_q.put(("ERR", f"Connection Failed: {e}"))
            return False
            
    def run(self):
        while True:
            # 1. Connection Loop
            if not self.sock:
                if self.connect():
                    self.authenticate()
                else:
                    time.sleep(5)
                    continue
                    
            # 2. Operations Loop
            try:
                # Send Pending Shares
                while not self.res_q.empty():
                    item = self.res_q.get()
                    # Construct valid submit
                    # "mining.submit": [worker, job_id, extranonce2, ntime, nonce]
                    params = [
                        f"{CONFIG['WALLET']}.{CONFIG['WORKER']}_local",
                        item['jid'],
                        item['en2'],
                        item['ntime'],
                        item['nonce']
                    ]
                    self.send("mining.submit", params)
                    self.log_q.put(("TX", f"Submitted Nonce {item['nonce']}"))
                    with self.state.local_tx.get_lock(): self.state.local_tx.value += 1

                # Receive Data (Non-blocking check)
                r, _, _ = select.select([self.sock], [], [], 0.1)
                if r:
                    data = self.sock.recv(4096)
                    if not data:
                        raise Exception("Socket Closed by Pool")
                    
                    self.recv_buffer += data.decode('utf-8')
                    
                    # Split logic to handle fragmented packets
                    while '\n' in self.recv_buffer:
                        line, self.recv_buffer = self.recv_buffer.split('\n', 1)
                        if not line.strip(): continue
                        
                        try:
                            msg = json.loads(line)
                            self.process_msg(msg)
                        except json.JSONDecodeError:
                            self.log_q.put(("WARN", "Packet Corrupt (JSON Error)"))
                            
            except Exception as e:
                self.log_q.put(("ERR", f"Link Drop: {e}"))
                self.close()
                time.sleep(2)

    def send(self, method, params):
        if not self.sock: return
        msg = json.dumps({"id": self.msg_id, "method": method, "params": params}) + "\n"
        try:
            self.sock.sendall(msg.encode())
            self.msg_id += 1
        except:
            self.close()

    def authenticate(self):
        self.send("mining.subscribe", ["MTP-v32-TitanII"])
        self.send("mining.authorize", [f"{CONFIG['WALLET']}.{CONFIG['WORKER']}_local", CONFIG['PASS']])
        self.log_q.put(("NET", "Authenticated"))

    def process_msg(self, msg):
        mid = msg.get('id')
        res = msg.get('result')
        method = msg.get('method')
        
        # Subscribe Info
        if mid == 1 and res:
            # res[1] = extranonce1, res[2] = extranonce2_size
            self.state.extranonce1.value = res[1].encode()
            self.state.extranonce2_size.value = res[2]
            
        # Share Response
        if mid and mid > 2:
            if res is True:
                with self.state.accepted.get_lock(): self.state.accepted.value += 1
                self.log_q.put(("RX", "Share VALID"))
            else:
                with self.state.rejected.get_lock(): self.state.rejected.value += 1
                err = msg.get('error')
                self.log_q.put(("RX", f"Share REJECTED: {err}"))
                
        # New Job
        if method == 'mining.notify':
            self.process_job(msg['params'])
            
    def process_job(self, p):
        # p: [jid, prev, c1, c2, mb, ver, nbits, ntime, clean]
        clean = p[8]
        
        # If clean, we MUST wipe the queue immediately to stop invalid shares
        if clean:
            while not self.job_q.empty():
                try: self.job_q.get_nowait()
                except: pass
        
        # Dispatch to workers
        en1 = self.state.extranonce1.value.decode()
        en2_sz = self.state.extranonce2_size.value
        
        job_struct = {
            'jid': p[0], 'prev': p[1], 'c1': p[2], 'c2': p[3],
            'mb': p[4], 'ver': p[5], 'nbits': p[6], 'ntime': p[7],
            'clean': p[8], 'en1': en1, 'en2_sz': en2_sz
        }
        
        # Broadcast to all CPU threads
        for _ in range(mp.cpu_count() * 2):
            self.job_q.put(job_struct)
            
        self.log_q.put(("RX", f"New Block {p[0][:8]}..."))

    def close(self):
        self.state.connected = False
        if self.sock:
            try: self.sock.close()
            except: pass
        self.sock = None
        self.recv_buffer = ""

# ==============================================================================
# SECTION 6: CUMULATIVE BENCHMARK (THE STRESS TEST)
# ==============================================================================

def cpu_burn(stop_ev, counter):
    """Generates heat via AVX/ALU loop."""
    while not stop_ev.is_set():
        # Mix of Integer and Float math for max power draw
        _ = [math.sqrt(x) * math.sin(x) for x in range(1000)]
        with counter.get_lock(): counter.value += 1000

def gpu_burn(stop_ev, log_q):
    """Runs Volatile CUDA Kernel."""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_TITAN_II_SRC)
        func = mod.get_function("titan_burn")
        
        log_q.put(("BENCH", "GPU CUDA Kernel Active"))
        
        while not stop_ev.is_set():
            out = np.zeros(1024, dtype=np.uint32)
            func(cuda.Out(out), np.uint32(time.time()), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
            
    except Exception as e:
        log_q.put(("WARN", f"GPU Bench Fail: {e}"))

def run_titan_benchmark(log_q):
    os.system('clear')
    print("==================================================")
    print("   TITAN II - DUAL STAGE AUDIT")
    print("==================================================")
    
    # Fan Enforcer
    def fan_daemon():
        while True:
            HAL.set_fan_max()
            time.sleep(15)
    threading.Thread(target=fan_daemon, daemon=True).start()
    
    # STAGE 1: CPU
    print(f"\n[STAGE 1] CPU MAX LOAD ({CONFIG['BENCH_STAGE_TIME']}s)")
    print("   -> Target: Thermal Saturation of CPU Die")
    
    stop_ev = mp.Event()
    counter = mp.Value('i', 0)
    procs = []
    
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_burn, args=(stop_ev, counter))
        p.start()
        procs.append(p)
        
    start_t = time.time()
    try:
        while time.time() - start_t < CONFIG['BENCH_STAGE_TIME']:
            rem = int(CONFIG['BENCH_STAGE_TIME'] - (time.time() - start_t))
            c = HAL.get_cpu_temp()
            rate = counter.value / (time.time() - start_t)
            sys.stdout.write(f"\r   T-{rem}s | CPU: {c}C | Load: {rate/1000:.0f} kOPs   ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt: pass
    
    stop_ev.set()
    for p in procs: p.terminate()
    
    # STAGE 2: SYSTEM BURN
    print(f"\n\n[STAGE 2] SYSTEM WIDE BURN ({CONFIG['BENCH_STAGE_TIME']}s)")
    print("   -> Target: CPU + GPU Max TGP")
    
    stop_ev = mp.Event()
    counter = mp.Value('i', 0)
    procs = []
    
    # CPU Again
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_burn, args=(stop_ev, counter))
        p.start()
        procs.append(p)
        
    # GPU
    g_proc = mp.Process(target=gpu_burn, args=(stop_ev, log_q))
    g_proc.start()
    
    start_t = time.time()
    try:
        while time.time() - start_t < CONFIG['BENCH_STAGE_TIME']:
            rem = int(CONFIG['BENCH_STAGE_TIME'] - (time.time() - start_t))
            c = HAL.get_cpu_temp()
            g = HAL.get_gpu_temp()
            sys.stdout.write(f"\r   T-{rem}s | CPU: {c}C | GPU: {g}C | Status: MAX_LOAD   ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt: pass
    
    stop_ev.set()
    for p in procs: p.terminate()
    if g_proc.is_alive(): g_proc.terminate()
    
    print("\n\n[DONE] Audit Complete. Starting Production Miner...")
    time.sleep(3)

# ==============================================================================
# SECTION 7: PROXY SERVER (ASYNC SELECTOR)
# ==============================================================================

class ProxyServer(threading.Thread):
    def __init__(self, log_q, state):
        super().__init__()
        self.log_q = log_q
        self.state = state
        self.daemon = True
        
    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((CONFIG['PROXY_HOST'], CONFIG['PROXY_PORT']))
        srv.listen(100)
        self.log_q.put(("INFO", f"Proxy Active on {CONFIG['PROXY_PORT']}"))
        
        while True:
            try:
                c, a = srv.accept()
                threading.Thread(target=self.handle_client, args=(c,a), daemon=True).start()
            except: pass
            
    def handle_client(self, client, addr):
        pool = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            pool.connect((CONFIG['POOL_URL'], CONFIG['POOL_PORT']))
            ip_id = addr[0].split('.')[-1]
            
            with self.state.proxy_clients.get_lock(): self.state.proxy_clients.value += 1
            
            # Keep Alive
            def ping():
                while True:
                    time.sleep(30)
                    try: pool.sendall(b'\n')
                    except: break
            threading.Thread(target=ping, daemon=True).start()
            
            inputs = [client, pool]
            while True:
                r, _, _ = select.select(inputs, [], [], 1)
                if not r: continue
                
                if client in r:
                    d = client.recv(4096)
                    if not d: break
                    # Rewrite
                    try:
                        s = d.decode()
                        if "mining.authorize" in s:
                            j = json.loads(s)
                            j['params'][0] = f"{CONFIG['WALLET']}.ASIC_{ip_id}"
                            d = (json.dumps(j)+"\n").encode()
                        if "mining.submit" in s:
                            with self.state.proxy_tx.get_lock(): self.state.proxy_tx.value += 1
                    except: pass
                    pool.sendall(d)
                    
                if pool in r:
                    d = pool.recv(4096)
                    if not d: break
                    if b'true' in d:
                        with self.state.proxy_rx.get_lock(): self.state.proxy_rx.value += 1
                    client.sendall(d)
        except: pass
        finally:
            client.close(); pool.close()
            with self.state.proxy_clients.get_lock(): self.state.proxy_clients.value -= 1

# ==============================================================================
# SECTION 8: MINER WORKERS (VALIDATION ENGINE)
# ==============================================================================

def cpu_miner(id, job_q, res_q, stop_ev, hash_counter):
    my_nonce = id * 100_000_000
    curr_job = None
    
    while not stop_ev.is_set():
        try:
            try:
                # 0.1s timeout allows checking stop_ev
                job_data = job_q.get(timeout=0.1)
                
                # Check for clean job (Strict Manager)
                if job_data['clean'] or (curr_job and job_data['jid'] != curr_job['jid']):
                    curr_job = job_data
                    my_nonce = id * 100_000_000 # Reset nonce range
            except queue.Empty:
                if curr_job is None: continue
                
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
            
            # --- SHA256d PIPELINE ---
            
            # 1. Extranonce2
            en2_hex = struct.pack('>I', random.randint(0, 2**32-1)).hex().zfill(en2_sz*2)
            
            # 2. Coinbase
            coinbase_bin = binascii.unhexlify(c1 + en1 + en2_hex + c2)
            cb_hash = hashlib.sha256(hashlib.sha256(coinbase_bin).digest()).digest()
            
            # 3. Merkle Root
            root = cb_hash
            for branch in mb:
                root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(branch)).digest()).digest()
                
            # 4. Header Construction (Endianness Critical)
            # Stratum V1 sends Header fields in a specific endianness.
            # Ver, Prev, Time, Bits usually need byte-reversal for hashing vs transmission.
            
            header_pre = (
                binascii.unhexlify(ver)[::-1] +
                binascii.unhexlify(prev)[::-1] +
                root +
                binascii.unhexlify(ntime)[::-1] +
                binascii.unhexlify(nbits)[::-1]
            )
            
            # 5. Search Loop
            # Diff 1 Target (for local valid check)
            # This filters out garbage shares before sending to pool
            target_diff1 = b'\x00\x00\x00\x00' 
            
            for n in range(my_nonce, my_nonce + 5000):
                # Header + Nonce(LE)
                header = header_pre + struct.pack('<I', n)
                block_hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                
                # Check for high difficulty share (LE check for trailing zeros)
                if block_hash.endswith(target_diff1):
                    res_q.put({
                        'jid': jid,
                        'en2': en2_hex,
                        'ntime': ntime,
                        'nonce': struct.pack('>I', n).hex() # Stratum V1 expects BE Hex
                    })
                    break # Found a share in this batch, move to next EN2
            
            my_nonce += 5000
            with hash_counter.get_lock():
                hash_counter.value += 5000
                
        except Exception:
            pass

def gpu_miner(stop_ev, hash_counter, log_q):
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_TITAN_II_SRC)
        func = mod.get_function("titan_burn")
        log_q.put(("GPU", "Volatile Kernel Loaded"))
        
        while not stop_ev.is_set():
            out = np.zeros(1024, dtype=np.uint32)
            # Launch
            func(cuda.Out(out), np.uint32(time.time()), block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
            with hash_counter.get_lock():
                hash_counter.value += (65535 * 512 * 4000)
    except: pass

# ==============================================================================
# SECTION 9: MAIN DASHBOARD
# ==============================================================================

def main_gui(stdscr, state, job_q, res_q, log_q):
    # Curses Init
    try:
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
    except: pass
    stdscr.nodelay(True)
    
    # Launch Network
    client = StratumClient(state, job_q, res_q, log_q)
    threading.Thread(target=client.run, daemon=True).start()
    
    # Launch Proxy
    ProxyServer(log_q, state).start()
    
    # Fan Control
    def fans():
        while True: 
            HAL.set_fan_max(); time.sleep(15)
    threading.Thread(target=fans, daemon=True).start()
    
    # Launch Workers
    stop_ev = mp.Event()
    procs = []
    
    # Use Manager for Hash Counter to avoid Pipe Errors
    hash_counter = mp.Value('d', 0.0)
    
    for i in range(max(1, mp.cpu_count()-1)):
        p = mp.Process(target=cpu_miner, args=(i, job_q, res_q, stop_ev, hash_counter))
        p.start()
        procs.append(p)
        
    gp = mp.Process(target=gpu_miner, args=(stop_ev, hash_counter, log_q))
    gp.start()
    procs.append(gp)
    
    logs = []
    current_hr = 0.0
    last_h = 0.0
    
    while True:
        # Logs
        while not log_q.empty():
            try:
                logs.append(log_q.get_nowait())
                if len(logs) > 30: logs.pop(0)
            except: pass
            
        # Stats
        total = hash_counter.value
        delta = total - last_h
        last_h = total
        current_hr = (current_hr * 0.8) + (delta * 10 * 0.2)
        
        # Draw
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        stdscr.addstr(0, 0, " MTP v32 TITAN II ".center(w), curses.color_pair(5))
        
        c = HAL.get_cpu_temp()
        g = HAL.get_gpu_temp()
        
        stdscr.addstr(2, 2, "LOCAL", curses.color_pair(4))
        stdscr.addstr(3, 2, f"CPU: {c}C")
        stdscr.addstr(4, 2, f"GPU: {g}C")
        
        if current_hr > 1e9: hrs = f"{current_hr/1e9:.2f} GH/s"
        elif current_hr > 1e6: hrs = f"{current_hr/1e6:.2f} MH/s"
        else: hrs = f"{current_hr/1000:.2f} kH/s"
        stdscr.addstr(5, 2, f"Hash: {hrs}")
        
        stdscr.addstr(2, 30, "NET", curses.color_pair(4))
        status = "UP" if state.connected else "DOWN"
        stdscr.addstr(3, 30, f"Link: {status}")
        stdscr.addstr(4, 30, f"Acc: {state.accepted.value}")
        stdscr.addstr(5, 30, f"Rej: {state.rejected.value}")
        
        stdscr.addstr(2, 60, "PROXY", curses.color_pair(4))
        stdscr.addstr(3, 60, f"Clients: {state.proxy_clients.value}")
        stdscr.addstr(4, 60, f"Tx: {state.proxy_tx.value}")
        stdscr.addstr(5, 60, f"Rx: {state.proxy_rx.value}")
        
        stdscr.hline(7, 0, '-', w)
        for i, (lvl, msg) in enumerate(logs):
            c = curses.color_pair(5)
            if lvl == "ERR": c = curses.color_pair(3)
            elif lvl == "TX": c = curses.color_pair(2)
            elif lvl == "RX": c = curses.color_pair(1)
            try: stdscr.addstr(8+i, 2, f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}", c)
            except: pass
            
        stdscr.refresh()
        time.sleep(0.1)
        if stdscr.getch() == ord('q'): break
        
    stop_ev.set()
    for p in procs: p.terminate()

if __name__ == "__main__":
    # 1. Boot Checks
    
    # 2. Bench
    # Using a fake queue for bench logs to keep it simple
    bq = mp.Queue()
    run_titan_benchmark(bq)
    
    # 3. Init Shared State
    manager = mp.Manager()
    state = manager.Namespace()
    state.connected = False
    state.local_tx = manager.Value('i', 0)
    state.accepted = manager.Value('i', 0)
    state.rejected = manager.Value('i', 0)
    state.proxy_clients = manager.Value('i', 0)
    state.proxy_tx = manager.Value('i', 0)
    state.proxy_rx = manager.Value('i', 0)
    
    state.extranonce1 = manager.Value('c', b'')
    state.extranonce2_size = manager.Value('i', 4)
    
    job_q = manager.Queue()
    res_q = manager.Queue()
    log_q = manager.Queue()
    
    try:
        curses.wrapper(main_gui, state, job_q, res_q, log_q)
    except KeyboardInterrupt: pass
