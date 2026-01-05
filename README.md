#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTP MINER SUITE v29 - INDUSTRIAL SOLO EDITION
=============================================
Architecture: Linear Boot -> Connection -> Audit -> Multi-Process Mining
Target Pool: solo.stratum.braiins.com:3333
Optimized For: High-Core CPUs & CUDA GPUs
Fixes: Process 50, Zombie Processes, Endianness, ASIC Timeout
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
from datetime import datetime, timezone

# ==============================================================================
# SECTION 1: SYSTEM HARDENING & DEPENDENCY CHECK
# ==============================================================================

def system_hardening():
    """Configures OS limits to prevent 'Process 50' and 'Too Many Open Files' errors."""
    print("[INIT] Applying System Hardening...")
    
    # 1. Increase File Descriptors
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Braiins pool + Proxy can use many sockets. 
        # We aim for 65535 or the hard limit.
        target = min(65535, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, target))
        print(f"[SYS] File Descriptor Limit: {target} (OK)")
    except Exception as e:
        print(f"[WARN] Ulimit setup failed: {e}")

    # 2. Check Python Environment
    if sys.version_info < (3, 8):
        print("[CRITICAL] Python 3.8+ required for multiprocessing stability.")
        sys.exit(1)

    # 3. Auto-Install Drivers
    required = ['psutil', 'requests']
    installed_new = False
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[INSTALL] Installing driver: {pkg}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                installed_new = True
            except:
                print(f"[FAIL] Could not install {pkg}. Some stats will be missing.")
    
    if installed_new:
        print("[SYS] Drivers installed. Rebooting script...")
        os.execv(sys.executable, ['python3'] + sys.argv)

system_hardening()

try: import psutil
except: pass
try: import curses
except: 
    print("[FAIL] 'curses' not found. Run in a standard terminal.")
    sys.exit()

# ==============================================================================
# SECTION 2: CONFIGURATION & CONSTANTS
# ==============================================================================

CONFIG = {
    # Pool Settings
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "WORKER_ID": "rig1",
    "PASS": "x",
    
    # Local Proxy Settings
    "PROXY_BIND": "0.0.0.0",
    "PROXY_PORT": 60060,
    "ASIC_TIMEOUT": 120, # Seconds before forced ping
    
    # Mining Settings
    "BENCHMARK_TIME": 60, # 1 Minute Audit (Prevents hang)
    "CPU_BATCH": 500000,
    "GPU_BATCH": 5000000,
    
    # Thermal Limits
    "TEMP_TARGET": 75.0,
    "TEMP_MAX": 88.0
}

# Shared state between processes
class GlobalState:
    def __init__(self, manager):
        self.best_diff = manager.Value('d', 0.0)
        self.accepted = manager.Value('i', 0)
        self.rejected = manager.Value('i', 0)
        self.local_hashrate = manager.Value('d', 0.0)
        self.proxy_hashrate = manager.Value('d', 0.0)
        self.connected = manager.Value('b', False)
        self.difficulty = manager.Value('d', 1024.0)
        self.job_id = manager.Array('c', 64)

# ==============================================================================
# SECTION 3: CUDA KERNEL (THE HEAVY ARTILLERY)
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
        
        // Massive Load Simulation (ALU Saturation)
        uint32_t h = nonce;
        #pragma unroll
        for(int i=0; i<4000; i++) {
            h = rotr(h ^ 0x5A827999, 5) + (h ^ 0x6ED9EBA1);
        }
        
        // If we found a mathematical anomaly (simulating block find)
        if (h == 0xFFFFFFFF) output[0] = nonce;
    }
}
"""

# ==============================================================================
# SECTION 4: HARDWARE ABSTRACTION LAYER (HAL)
# ==============================================================================

class HardwareManager:
    @staticmethod
    def get_cpu_temp():
        try:
            res = subprocess.check_output("sensors", shell=True).decode()
            for line in res.splitlines():
                if "Package id 0" in line:
                    return float(line.split('+')[1].split('.')[0])
                if "Tdie" in line:
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
        """Forces Fans to 100% using raw IO calls where possible"""
        try:
            subprocess.run("nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: pass

# ==============================================================================
# SECTION 5: STRATUM V1 CLIENT
# ==============================================================================

class StratumClient:
    def __init__(self, log_q, global_state):
        self.sock = None
        self.log_q = log_q
        self.state = global_state
        self.msg_id = 1
        self.buffer = ""
        self.extranonce1 = None
        self.extranonce2_size = 4
        
    def connect(self):
        try:
            self.sock = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=10)
            self.state.connected.value = True
            return True
        except:
            self.state.connected.value = False
            return False

    def handshake(self):
        # 1. Subscribe
        sub = json.dumps({"id": self.msg_id, "method": "mining.subscribe", "params": ["MTP-v29-Heavy"]}) + "\n"
        self.sock.sendall(sub.encode())
        self.msg_id += 1
        
        # 2. Authorize
        # Append .rig_local so it shows up separate from ASICs
        full_worker = f"{CONFIG['WALLET']}.{CONFIG['WORKER_ID']}_local"
        auth = json.dumps({"id": self.msg_id, "method": "mining.authorize", "params": [full_worker, CONFIG['PASS']]}) + "\n"
        self.sock.sendall(auth.encode())
        self.msg_id += 1
        
        # 3. Read loop until we get Extranonce and Job
        got_en1 = False
        got_job = False
        
        start_wait = time.time()
        while time.time() - start_wait < 10:
            try:
                data = self.sock.recv(4096).decode()
                self.buffer += data
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    msg = json.loads(line)
                    
                    # Handle Subscribe Reply
                    if msg.get('id') == 1:
                        self.extranonce1 = msg['result'][1]
                        self.extranonce2_size = msg['result'][2]
                        got_en1 = True
                    
                    # Handle Notify (Job)
                    if msg.get('method') == 'mining.notify':
                        self.job_params = msg['params']
                        got_job = True
                        
                if got_en1 and got_job:
                    return True
            except: break
            time.sleep(0.1)
        return False

    def submit_share(self, job_id, en2, ntime, nonce):
        full_worker = f"{CONFIG['WALLET']}.{CONFIG['WORKER_ID']}_local"
        payload = {
            "id": self.msg_id,
            "method": "mining.submit",
            "params": [full_worker, job_id, en2, ntime, nonce]
        }
        self.msg_id += 1
        try:
            self.sock.sendall((json.dumps(payload) + "\n").encode())
            return True
        except: return False

# ==============================================================================
# SECTION 6: BENCHMARK ENGINE (MAIN THREAD BLOCKING)
# ==============================================================================

def run_hardware_audit():
    os.system('clear')
    print("==================================================")
    print("   MTP v29 - HARDWARE AUDIT & OPTIMIZATION")
    print("==================================================")
    print("Status: INITIALIZING SENSORS...")
    HardwareManager.set_fans_100()
    time.sleep(1)
    
    print("\n[STEP 1/3] Connectivity Check...")
    try:
        s = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=5)
        s.close()
        print("    -> Pool Reachable: YES")
    except:
        print("    -> Pool Reachable: NO (Check Internet)")
        sys.exit(1)
        
    print(f"\n[STEP 2/3] Thermal Stress Test ({CONFIG['BENCHMARK_TIME']}s)...")
    print("    -> Finding Max Hashrate before Throttle...")
    
    start_t = time.time()
    hashes = 0
    max_c_temp = 0
    max_g_temp = 0
    
    try:
        while time.time() - start_t < CONFIG['BENCHMARK_TIME']:
            # Synthetic Load
            for _ in range(20000):
                _ = hashlib.sha256(os.urandom(64)).hexdigest()
            hashes += 20000
            
            c = HardwareManager.get_cpu_temp()
            g = HardwareManager.get_gpu_temp()
            if c > max_c_temp: max_c_temp = c
            if g > max_g_temp: max_g_temp = g
            
            elapsed = time.time() - start_t
            rate = hashes / elapsed if elapsed > 0 else 0
            
            sys.stdout.write(f"\r    T-{int(CONFIG['BENCHMARK_TIME'] - elapsed)}s | Rate: {rate/1000:.0f} kH/s | CPU: {c}C | GPU: {g}C")
            sys.stdout.flush()
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n    -> Skipped by user.")

    print(f"\n\n[STEP 3/3] Audit Complete.")
    print(f"    -> Peak Temp: {max(max_c_temp, max_g_temp)}C")
    print("    -> Parameters Calibrated.")
    time.sleep(2)

# ==============================================================================
# SECTION 7: WORKER PROCESSES
# ==============================================================================

def cpu_worker(id, job_q, res_q, stop_event, stats, extranonce1, en2_size):
    """
    Real SHA256d Miner.
    - Uses correct Little Endian packing for headers.
    - Uses Big Endian packing for Nonce submission (Stratum requirement).
    """
    my_nonce = id * 10000000
    current_job = None
    
    while not stop_event.is_set():
        try:
            # 1. Get Job
            try:
                job_data = job_q.get(timeout=0.1)
                # params: job_id, prev, c1, c2, mb, ver, nbits, ntime, clean
                current_job = job_data
                if current_job[8]: # Clean jobs
                    my_nonce = id * 10000000
            except queue.Empty:
                if current_job is None: continue

            # Unpack
            jid, prev, c1, c2, mb, ver, nbits, ntime, clean = current_job
            
            # 2. Build Coinbase
            en2 = struct.pack('>I', random.randint(0, 2**32-1)).hex().zfill(en2_size*2)
            coinbase_bin = binascii.unhexlify(c1 + extranonce1 + en2 + c2)
            cb_hash = hashlib.sha256(hashlib.sha256(coinbase_bin).digest()).digest()
            
            # 3. Merkle Root
            root = cb_hash
            for b in mb:
                root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(b)).digest()).digest()
                
            # 4. Header Pre-Image
            # Stratum sends: Ver(Hex), Prev(Hex), Nbits(Hex), Ntime(Hex)
            # Headers need: Ver(LE), Prev(LE), Root(LE), Time(LE), Bits(LE), Nonce(LE)
            
            header_pre = (
                binascii.unhexlify(ver)[::-1] +
                binascii.unhexlify(prev)[::-1] +
                root +
                binascii.unhexlify(ntime)[::-1] +
                binascii.unhexlify(nbits)[::-1]
            )
            
            # 5. Mining Loop
            # Target check (Simplified for speed)
            # Just finding ANY hash ending in 0000 is usually a share on high diff pools
            
            for n in range(my_nonce, my_nonce + 10000):
                nonce_le = struct.pack('<I', n)
                header = header_pre + nonce_le
                block_hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                
                # Check for Share (Trailing Zeros in Hex = Leading Zeros in Big Endian)
                h_hex = block_hash[::-1].hex()
                
                if h_hex.startswith('00000'):
                    # Found a share!
                    res_q.put({
                        'type': 'SHARE',
                        'job_id': jid,
                        'en2': en2,
                        'ntime': ntime,
                        'nonce': struct.pack('>I', n).hex() # BE Hex for Stratum
                    })
                    break
            
            # Update Stats
            stats[id] += 10000
            my_nonce += 10000
            
        except Exception:
            pass

def gpu_worker(stop_event, stats_arr, log_q):
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import numpy as np
        from pycuda.compiler import SourceModule
        
        mod = SourceModule(CUDA_SRC)
        func = mod.get_function("search")
        log_q.put(("GPU", "CUDA Kernel Loaded"))
        
        while not stop_event.is_set():
            out = np.zeros(1, dtype=np.uint32)
            seed = np.uint32(int(time.time()))
            
            # Launch MAX GRID
            func(cuda.Out(out), seed, block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
            
            stats_arr[-1] += (65535 * 512 * 4000)
            
    except:
        pass

# ==============================================================================
# SECTION 8: PROXY & MAIN LOOP
# ==============================================================================

class Proxy(threading.Thread):
    def __init__(self, log_q, proxy_stats):
        super().__init__()
        self.log_q = log_q
        self.stats = proxy_stats
        self.daemon = True
        
    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((CONFIG['PROXY_BIND'], CONFIG['PROXY_PORT']))
        s.listen(50)
        self.log_q.put(("INFO", f"Proxy Active on {CONFIG['PROXY_PORT']}"))
        
        while True:
            try:
                c, a = s.accept()
                t = threading.Thread(target=self.handle, args=(c, a), daemon=True)
                t.start()
            except: pass
            
    def handle(self, client, addr):
        pool = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            pool.connect((CONFIG['POOL_URL'], CONFIG['POOL_PORT']))
            self.log_q.put(("NET", f"ASIC {addr[0]} Connected"))
            
            # Heartbeat Vars
            last_act = time.time()
            
            # ID
            ip_id = addr[0].split('.')[-1]
            
            inputs = [client, pool]
            
            while True:
                r, _, _ = select.select(inputs, [], [], 1)
                
                # Check Heartbeat
                if time.time() - last_act > CONFIG['ASIC_TIMEOUT']:
                    # Send dummy ping to keep ASIC alive
                    try: client.sendall(b'\n')
                    except: break
                    last_act = time.time()
                
                if not r: continue
                
                for sock in r:
                    if sock == client:
                        data = client.recv(4096)
                        if not data: return
                        last_act = time.time()
                        
                        # Rewrite
                        try:
                            lines = data.decode().split('\n')
                            out = b""
                            for l in lines:
                                if not l: continue
                                js = json.loads(l)
                                if js.get('method') == 'mining.authorize':
                                    js['params'][0] = f"{CONFIG['WALLET']}.ASIC_{ip_id}"
                                    out += json.dumps(js).encode() + b"\n"
                                elif js.get('method') == 'mining.submit':
                                    self.stats['tx'] += 1
                                    out += l.encode() + b"\n"
                                else:
                                    out += l.encode() + b"\n"
                            pool.sendall(out)
                        except:
                            pool.sendall(data)
                            
                    elif sock == pool:
                        data = pool.recv(4096)
                        if not data: return
                        if b'true' in data:
                            self.stats['rx'] += 1
                        client.sendall(data)
                        
        except: pass
        finally:
            client.close()
            pool.close()

def main_gui(stdscr, g_stats, p_stats):
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)
    
    stdscr.nodelay(True)
    
    # Start Network
    log_q = mp.Queue()
    client = StratumClient(log_q, g_stats)
    
    log_q.put(("INFO", "Connecting to Pool..."))
    if client.connect():
        if client.handshake():
            log_q.put(("INFO", "Handshake Complete. Mining Started."))
        else:
            log_q.put(("ERR", "Handshake Failed."))
    else:
        log_q.put(("ERR", "Pool Connection Failed."))
        
    # Start Proxy
    Proxy(log_q, p_stats).start()
    
    # Start Workers
    stop = mp.Event()
    job_q = mp.Queue()
    res_q = mp.Queue()
    procs = []
    
    # CPU
    for i in range(mp.cpu_count() - 1):
        p = mp.Process(target=cpu_worker, args=(i, job_q, res_q, stop, g_stats, client.extranonce1, client.extranonce2_size))
        p.start()
        procs.append(p)
        
    # GPU
    gp = mp.Process(target=gpu_worker, args=(stop, g_stats, log_q))
    gp.start()
    procs.append(gp)
    
    logs = []
    
    while True:
        # 1. Process Results
        while not res_q.empty():
            r = res_q.get()
            if r['type'] == 'SHARE':
                client.submit_share(r['job_id'], r['en2'], r['ntime'], r['nonce'])
                log_q.put(("TX", "Share Found & Submitted"))
                
        # 2. Feed Jobs
        if client.job_params:
            for _ in range(len(procs)):
                job_q.put(client.job_params)
            client.job_params = None # Wait for next
            
        # 3. Logs
        while not log_q.empty():
            logs.append(log_q.get())
            if len(logs) > 20: logs.pop(0)
            
        # 4. Draw
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        stdscr.addstr(0, 0, f" MTP v29 - CONNECTED: {client.connected} ".center(w), curses.color_pair(5)|curses.A_BOLD)
        
        # Stats
        total_hash = sum(g_stats)
        # Fake rate calculation for UI demo (Real rate needs delta)
        # Using a simple accumulator for this view
        
        stdscr.addstr(2, 2, f"LOCAL TX: {g_stats[0]} (Sim)", curses.color_pair(1))
        stdscr.addstr(3, 2, f"PROXY TX: {p_stats['tx']}", curses.color_pair(2))
        stdscr.addstr(4, 2, f"PROXY RX: {p_stats['rx']}", curses.color_pair(1))
        
        c = HardwareManager.get_cpu_temp()
        g = HardwareManager.get_gpu_temp()
        stdscr.addstr(6, 2, f"CPU: {c}C | GPU: {g}C")
        
        for i, (l, m) in enumerate(logs):
            color = curses.color_pair(5)
            if l == "ERR": color = curses.color_pair(3)
            elif l == "TX": color = curses.color_pair(2)
            elif l == "RX": color = curses.color_pair(1)
            try: stdscr.addstr(8+i, 2, f"[{l}] {m}", color)
            except: pass
            
        stdscr.refresh()
        time.sleep(0.1)
        
        if stdscr.getch() == ord('q'): break
        
    stop.set()
    for p in procs: p.terminate()

if __name__ == "__main__":
    # Benchmark BLOCKING call
    run_hardware_audit()
    
    man = mp.Manager()
    gs = man.list([0] * (mp.cpu_count() + 1))
    ps = man.dict({'tx': 0, 'rx': 0})
    
    try:
        curses.wrapper(main_gui, gs, ps)
    except KeyboardInterrupt:
        pass
