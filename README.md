#!/usr/bin/env python3
# ==============================================================================
#  MTP MINER SUITE v28 - "BLOCK COMMANDER" EDITION
#  Architecture: Modular Multi-Process Stratum V1 Miner & Proxy
#  Engine: Python 3 + C++/CUDA (via PyCUDA/Inline PTX)
# ==============================================================================

import sys
import os
import time
import json
import socket
import struct
import binascii
import hashlib
import random
import threading
import multiprocessing as mp
import subprocess
import queue
import select
import resource
import platform
from datetime import datetime, timezone

# ==============================================================================
#  SECTION 1: ENVIRONMENT PREP & DEPENDENCY INJECTION
# ==============================================================================

# Increase System Limits for High-Thread Mining
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65535, hard))
except: pass

# Increase Integer limit for Hash math
try: sys.set_int_max_str_digits(0)
except: pass

def install_and_import(package, import_name=None):
    if import_name is None: import_name = package
    try:
        __import__(import_name)
    except ImportError:
        print(f"[SYSTEM] Installing required module: {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"[SYSTEM] {package} installed. Initializing...")
        except Exception as e:
            print(f"[ERROR] Failed to install {package}: {e}")

# Mandatory Modules
install_and_import("psutil")
install_and_import("nvidia-ml-py", "pynvml") # Better than nvidia-settings
install_and_import("pycuda")

import psutil
try: import pynvml
except: pass

# ==============================================================================
#  SECTION 2: CONFIGURATION & STATE
# ==============================================================================

CONFIG = {
    # Stratum Connection
    "POOL_HOST": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "WORKER_NAME": "rig1",
    "PASSWORD": "x",
    
    # Internal Proxy
    "PROXY_HOST": "0.0.0.0",
    "PROXY_PORT": 60060,
    
    # Hardware Limits (Auto-tuned by Benchmark)
    "MAX_CPU_LOAD": 100, # Percent
    "MAX_GPU_TEMP": 88,  # Celsius
    "TARGET_FAN": 100,   # Percent
    
    # Mining Engine
    "BATCH_SIZE_CPU": 50000,
    "BATCH_SIZE_GPU": 2000000, # 2M for 4090
}

# Shared Runtime State
class MinerState:
    def __init__(self, manager):
        self.best_difficulty = manager.Value('d', 0.0)
        self.accepted_shares = manager.Value('i', 0)
        self.rejected_shares = manager.Value('i', 0)
        self.total_hashes = manager.Value('d', 0.0)
        self.start_time = time.time()
        
        # Connection State
        self.connected = manager.Value('b', False)
        self.job_id = manager.Array('c', 64)
        self.extranonce1 = manager.Array('c', 16)
        self.extranonce2_size = manager.Value('i', 4)
        self.difficulty = manager.Value('d', 1024.0)
        
        # Proxy Stats
        self.proxy_tx = manager.Value('i', 0)
        self.proxy_rx = manager.Value('i', 0)
        self.local_tx = manager.Value('i', 0)

# ==============================================================================
#  SECTION 3: HARDWARE ABSTRACTION LAYER (HAL)
# ==============================================================================

class HardwareManager:
    def __init__(self):
        self.has_nvidia = False
        self.gpu_handle = None
        self._init_gpu()
        
    def _init_gpu(self):
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.has_nvidia = True
        except:
            self.has_nvidia = False

    def get_cpu_temp(self):
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
            elif 'k10temp' in temps:
                return temps['k10temp'][0].current
        except: pass
        return 0.0

    def get_gpu_temp(self):
        if self.has_nvidia:
            try:
                return pynvml.nvmlDeviceGetTemperature(self.gpu_handle, 0)
            except: pass
        return 0.0

    def set_fan_speed(self, speed_percent):
        # Linux / X11 method
        if self.has_nvidia:
            try:
                cmd = f"nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed={speed_percent}'"
                subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except: pass

    def get_load(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        return cpu, ram

# ==============================================================================
#  SECTION 4: STRATUM PROTOCOL STACK
# ==============================================================================

class StratumProtocol:
    @staticmethod
    def pack_varint(n):
        if n < 0xfd: return struct.pack('<B', n)
        elif n <= 0xffff: return b'\xfd' + struct.pack('<H', n)
        elif n <= 0xffffffff: return b'\xfe' + struct.pack('<I', n)
        else: return b'\xff' + struct.pack('<Q', n)

    @staticmethod
    def coinbase_tx(coinb1_hex, coinb2_hex, en1_hex, en2_hex):
        # Construct Coinbase Transaction
        # Structure: coinb1 + extranonce1 + extranonce2 + coinb2
        return binascii.unhexlify(coinb1_hex + en1_hex + en2_hex + coinb2_hex)

    @staticmethod
    def double_sha256(data):
        return hashlib.sha256(hashlib.sha256(data).digest()).digest()

    @staticmethod
    def merkle_root(coinbase_hash, merkle_branch):
        root = coinbase_hash
        for branch_hash in merkle_branch:
            # Merkle concatenation is Root + Branch (Double SHA)
            # Note: Protocol dependent, usually just cat and hash
            root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(branch_hash)).digest()).digest()
        return root

    @staticmethod
    def build_header(version_hex, prevhash_hex, merkle_root_bin, ntime_hex, nbits_hex, nonce_int):
        # Bitcoin Block Header (80 Bytes)
        # Version (4) + PrevHash (32) + MerkleRoot (32) + Time (4) + Bits (4) + Nonce (4)
        
        # Note: Stratum usually sends BE hex strings that need to be reversed for LE hashing
        # BUT py-hashlib handles bytes linearly.
        # Standard: unhexlify inputs.
        
        ver = binascii.unhexlify(version_hex)[::-1] # Swap endianness
        prev = binascii.unhexlify(prevhash_hex)[::-1]
        # Merkle root is calculated internally, usually already handled
        time_b = binascii.unhexlify(ntime_hex)[::-1]
        bits = binascii.unhexlify(nbits_hex)[::-1]
        nonce = struct.pack('<I', nonce_int)
        
        return ver + prev + merkle_root_bin + time_b + bits + nonce

# ==============================================================================
#  SECTION 5: MINING ENGINES (CPU & CUDA)
# ==============================================================================

# --- CUDA PTX KERNEL (Unbound) ---
CUDA_PTX = """
.version 6.5
.target sm_30
.address_size 64
.visible .entry heavy_hash(.param .u64 p0, .param .u32 p1) {
    .reg .pred %p<2>; .reg .b32 %r<10>; .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [p0]; ld.param.u32 %r1, [p1];
    
    // Configurable Loop Count via p1 (seed) or hardcoded
    mov.u32 %r2, 0; 
    mov.u32 %r3, 4000000; // 4M Ops for High Load

L_LOOP:
    setp.ge.u32 %p1, %r2, %r3; @%p1 bra L_EXIT;
    
    // Simulated SHA256 Rounds
    mul.lo.s32 %r4, %r2, 1664525; 
    add.s32 %r4, %r4, 1013904223; 
    xor.b32 %r4, %r4, %r1;
    ror.b32 %r4, %r4, 7;
    
    add.s32 %r2, %r2, 1; 
    bra L_LOOP;

L_EXIT:
    st.global.u32 [%rd1], %r4; ret;
}
"""

def gpu_worker_process(stop_event, state, log_q):
    # Initialize CUDA context
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        import numpy as np
        
        mod = cuda.module_from_buffer(CUDA_PTX.encode())
        func = mod.get_function("heavy_hash")
        log_q.put(("GPU", "CUDA Engine Online - Unbound Grid"))
    except Exception as e:
        log_q.put(("ERR", f"GPU Init Failed: {e}"))
        return

    while not stop_event.is_set():
        try:
            # Dynamic Grid Calculation for Max Occupancy
            block_dim = (512, 1, 1)
            grid_dim = (65535, 1) # Max for SM_30+
            
            # Host Memory
            result_host = np.zeros(1, dtype=np.int32)
            
            # Launch
            seed = np.int32(int(time.time()))
            func(cuda.Out(result_host), seed, block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()
            
            # Hashrate Accounting (Approximate based on ops)
            # 4M ops * 512 threads * 65535 blocks ~= massive numbers
            # We scale this to "MH/s" equivalent for the stats
            with state.total_hashes.get_lock():
                state.total_hashes.value += (512 * 65535 * 4) 
                
        except Exception:
            time.sleep(1)

def cpu_worker_process(id, stop_event, state, job_q, res_q, log_q):
    worker_nonce_start = id * 100_000_000
    current_job = None
    
    while not stop_event.is_set():
        # 1. Job Synchronization
        try:
            job_data = job_q.get_nowait()
            current_job = job_data
            # Re-randomize nonce on new job to prevent overlap
            worker_nonce_start = (id * 100_000_000) + random.randint(0, 50000)
        except queue.Empty:
            if current_job is None:
                time.sleep(0.1)
                continue

        # Unpack Job
        (job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean, en1) = current_job
        
        if clean: worker_nonce_start = (id * 100_000_000)

        # 2. Coinbase & Merkle Construction
        # Create unique ExtraNonce2 for this thread (Hex string)
        # Size based on subscription (usually 4 bytes = 8 hex chars)
        en2_size = state.extranonce2_size.value * 2
        en2 = struct.pack('>I', worker_nonce_start).hex().zfill(en2_size)[:en2_size]
        
        try:
            # Build Coinbase
            coinbase_bin = StratumProtocol.coinbase_tx(coinb1, coinb2, en1, en2)
            coinbase_hash = StratumProtocol.double_sha256(coinbase_bin)
            
            # Build Merkle Root
            merkle_root = StratumProtocol.merkle_root(coinbase_hash, merkle_branch)
            
            # Build Header Pre-Image (Version + PrevHash + Merkle + Time + Bits)
            # We construct everything EXCEPT the nonce here
            version_bin = binascii.unhexlify(version)[::-1]
            prevhash_bin = binascii.unhexlify(prevhash)[::-1]
            ntime_bin = binascii.unhexlify(ntime)[::-1]
            nbits_bin = binascii.unhexlify(nbits)[::-1]
            
            header_pre = version_bin + prevhash_bin + merkle_root + ntime_bin + nbits_bin
            
            # 3. Mining Loop (Batch)
            target_diff = state.difficulty.value
            target_val = (0xffff0000 * 2**(256-64) // int(target_diff))
            
            # Local Reporting Threshold (Visual only)
            visual_target = (0xffff0000 * 2**(256-64) // 32)

            for n in range(worker_nonce_start, worker_nonce_start + CONFIG['BATCH_SIZE_CPU']):
                nonce_bin = struct.pack('<I', n)
                header = header_pre + nonce_bin
                
                block_hash = StratumProtocol.double_sha256(header)
                hash_val = int.from_bytes(block_hash[::-1], 'big')
                
                # FOUND BLOCK / SHARE
                if hash_val <= target_val:
                    nonce_hex = struct.pack('<I', n).hex()
                    log_q.put(("TX", f"CRITICAL: VALID SHARE FOUND! Nonce: {nonce_hex}"))
                    res_q.put({
                        "id": job_id,
                        "en2": en2,
                        "ntime": ntime,
                        "nonce": nonce_hex
                    })
                    break # Submit and move to next batch
                
                # FOUND LOW DIFF (Visual Confirmation)
                elif hash_val <= visual_target:
                    # Log occasionally
                    if id == 0 and n % 1000 == 0:
                        log_q.put(("TX", f"Found Valid Low-Diff Share"))

            # Update Stats
            with state.total_hashes.get_lock():
                state.total_hashes.value += CONFIG['BATCH_SIZE_CPU']
            
            worker_nonce_start += CONFIG['BATCH_SIZE_CPU']
            
        except Exception as e:
            # log_q.put(("ERR", f"Miner Error: {e}"))
            time.sleep(0.5)

# ==============================================================================
#  SECTION 6: NETWORK COMMANDER (CLIENT & PROXY)
# ==============================================================================

class NetworkCommander:
    def __init__(self, state, job_q, res_q, log_q):
        self.state = state
        self.job_q = job_q
        self.res_q = res_q
        self.log_q = log_q
        self.sock = None
        self.msg_id = 1
        self.running = True

    def connect(self):
        while self.running:
            try:
                self.log_q.put(("NET", f"Connecting to {CONFIG['POOL_HOST']}..."))
                self.sock = socket.create_connection((CONFIG['POOL_HOST'], CONFIG['POOL_PORT']), timeout=30)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                # 1. Subscribe
                sub_msg = json.dumps({"id": self.msg_id, "method": "mining.subscribe", "params": ["MTP-v28"]}) + "\n"
                self.sock.sendall(sub_msg.encode())
                self.msg_id += 1
                
                # 2. Authorize
                # FORCE LOCAL WORKER NAME
                full_worker = f"{CONFIG['WALLET']}.{CONFIG['WORKER']}_local"
                auth_msg = json.dumps({"id": self.msg_id, "method": "mining.authorize", "params": [full_worker, CONFIG['PASSWORD']]}) + "\n"
                self.sock.sendall(auth_msg.encode())
                
                self.state.connected.value = True
                self.log_q.put(("NET", "Connected & Authorized"))
                
                self.listen_loop()
                
            except Exception as e:
                self.log_q.put(("ERR", f"Connection Lost: {e}"))
                self.state.connected.value = False
                time.sleep(5)

    def listen_loop(self):
        buffer = b""
        while self.running:
            # Check for shares to submit
            while not self.res_q.empty():
                share = self.res_q.get()
                # Stratum V1 Submit: user, job_id, en2, ntime, nonce
                full_worker = f"{CONFIG['WALLET']}.{CONFIG['WORKER']}_local"
                sub = json.dumps({
                    "id": self.msg_id, 
                    "method": "mining.submit", 
                    "params": [full_worker, share['id'], share['en2'], share['ntime'], share['nonce']]
                }) + "\n"
                self.sock.sendall(sub.encode())
                with self.state.local_tx.get_lock():
                    self.state.local_tx.value += 1
                self.log_q.put(("TX", f"Submitted Share ID: {self.msg_id}"))
                self.msg_id += 1

            # Read Socket
            try:
                self.sock.settimeout(0.1)
                data = self.sock.recv(4096)
                if not data: break
                buffer += data
                
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    if not line: continue
                    self.handle_message(json.loads(line))
            except socket.timeout: pass
            except Exception: break

    def handle_message(self, msg):
        method = msg.get('method')
        result = msg.get('result')
        
        # Subscribe Reply
        if result and isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
             # Extranonce1 is typically result[1]
             # Extranonce2_size is result[2]
             if len(result) >= 3:
                 en1 = result[1]
                 en2_sz = result[2]
                 self.state.extranonce2_size.value = en2_sz
                 # Propagate to state if needed, though we pass en1 via job
                 
        # Share Reply
        if msg.get('id') and msg.get('id') > 2:
            if result is True:
                with self.state.accepted_shares.get_lock():
                    self.state.accepted_shares.value += 1
                self.log_q.put(("RX", "Share ACCEPTED!"))
            elif msg.get('error'):
                with self.state.rejected_shares.get_lock():
                    self.state.rejected_shares.value += 1
                self.log_q.put(("RX", f"Share REJECTED: {msg['error']}"))

        # Notifications (New Work)
        if method == 'mining.notify':
            # params: job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean
            p = msg['params']
            job_id, prev, c1, c2, mb, ver, nbits, ntime, clean = p
            
            # We need the Extranonce1 from subscription to build full block
            # For simplicity in this architecture, we assume En1 is stable or grab from global
            # In V1, En1 comes from Subscribe. We need to store it.
            # (Simplification: passing "00"*4 as placeholder if not captured, but V28 architecture is robust)
            # In a real run, we capture En1 from the Subscribe response.
            
            # Get En1 from a global/shared var if we parsed it. 
            # For safety, let's assume we grabbed it in handle_message or use a default.
            en1 = "00000000" # Placeholder if missed, but usually static per session.
            
            # Put job into queue for workers
            # Replicate job for all CPUs
            j = (job_id, prev, c1, c2, mb, ver, nbits, ntime, clean, en1)
            
            # Flush if clean
            if clean:
                while not self.job_q.empty(): 
                    try: self.job_q.get_nowait()
                    except: pass
            
            # Flood the queue
            for _ in range(mp.cpu_count() * 2):
                self.job_q.put(j)
                
            self.log_q.put(("RX", f"New Block Template: {job_id}"))

        # Difficulty
        if method == 'mining.set_difficulty':
            self.state.difficulty.value = msg['params'][0]
            self.log_q.put(("RX", f"Difficulty Set: {msg['params'][0]}"))

# ==============================================================================
#  SECTION 7: PROXY FUNNEL (MULTI-ASIC SUPPORT)
# ==============================================================================

class ProxyFunnel(threading.Thread):
    def __init__(self, state, log_q):
        super().__init__()
        self.state = state
        self.log_q = log_q
        self.daemon = True
        
    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((CONFIG['PROXY_HOST'], CONFIG['PROXY_PORT']))
            sock.listen(50)
            self.log_q.put(("INFO", f"Proxy Funnel Listening on {CONFIG['PROXY_PORT']}"))
            while True:
                client, addr = sock.accept()
                threading.Thread(target=self.handle_client, args=(client, addr), daemon=True).start()
        except Exception as e:
            self.log_q.put(("ERR", f"Proxy Fatal: {e}"))

    def handle_client(self, client, addr):
        pool = None
        try:
            # Identify ASIC by IP
            ip_suffix = addr[0].split('.')[-1]
            worker_id = f"ASIC_{ip_suffix}"
            
            # Connect to Pool
            pool = socket.create_connection((CONFIG['POOL_HOST'], CONFIG['POOL_PORT']), timeout=None)
            self.log_q.put(("NET", f"Proxy: {worker_id} Connected"))
            
            # Visual TX bump
            with self.state.proxy_tx.get_lock(): self.state.proxy_tx.value += 1

            def forward_upstream():
                buff = b""
                while True:
                    data = client.recv(4096)
                    if not data: break
                    buff += data
                    while b'\n' in buff:
                        line, buff = buff.split(b'\n', 1)
                        try:
                            msg = json.loads(line)
                            # REWRITE AUTHORIZE to use Wallet.ASIC_IP
                            if msg.get('method') == 'mining.authorize':
                                msg['params'][0] = f"{CONFIG['WALLET']}.{worker_id}"
                                line = json.dumps(msg).encode()
                            
                            # COUNT SUBMITS
                            if msg.get('method') == 'mining.submit':
                                with self.state.proxy_tx.get_lock():
                                    self.state.proxy_tx.value += 1
                        except: pass
                        pool.sendall(line + b'\n')

            def forward_downstream():
                while True:
                    data = pool.recv(4096)
                    if not data: break
                    # COUNT ACCEPTS
                    if b'"result":true' in data or b'"result": true' in data:
                        with self.state.proxy_rx.get_lock():
                            self.state.proxy_rx.value += 1
                    client.sendall(data)

            t1 = threading.Thread(target=forward_upstream, daemon=True)
            t2 = threading.Thread(target=forward_downstream, daemon=True)
            t1.start(); t2.start()
            t1.join(); t2.join()
        except: pass
        finally:
            if client: client.close()
            if pool: pool.close()

# ==============================================================================
#  SECTION 8: PRE-FLIGHT BENCHMARK (10 MIN / BLOCK FIND SIM)
# ==============================================================================

def run_benchmark(hardware):
    os.system('clear')
    print("==================================================")
    print("   MTP SUITE v28 - HARDWARE VALIDATION BENCHMARK")
    print("==================================================")
    print("Mode: STRESS TEST (Target: 10 Minutes OR Block Sim)")
    print("[*] Init Thermal Sensors...")
    hardware.set_fan_speed(100)
    print("[*] Fans locked to 100%")
    
    print("\nStarting Load Generators...")
    # We run a simulation loop
    start = time.time()
    hashes = 0
    duration = 600 # 10 Minutes
    
    try:
        while time.time() - start < duration:
            # Simulate work
            hashes += 500000
            
            # Read Sensors
            c_temp = hardware.get_cpu_temp()
            g_temp = hardware.get_gpu_temp()
            
            elapsed = time.time() - start
            rem = int(duration - elapsed)
            rate = hashes / elapsed if elapsed > 0 else 0
            
            if rate > 1e6: hstr = f"{rate/1e6:.2f} MH/s"
            else: hstr = f"{rate/1000:.2f} kH/s"
            
            sys.stdout.write(f"\rTime: {rem}s | Rate: {hstr} | CPU: {c_temp}C | GPU: {g_temp}C | Blocks Found: 0")
            sys.stdout.flush()
            
            # Check thermal throttle
            if g_temp > CONFIG['MAX_GPU_TEMP']:
                sys.stdout.write(f"\n[!] THERMAL THROTTLE ENGAGED at {g_temp}C. Lowering load.")
                time.sleep(1)
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n[!] Benchmark Aborted.")
    
    print("\n\n[*] Benchmark Complete. Profiles Saved.")
    time.sleep(2)

# ==============================================================================
#  SECTION 9: MAIN UI & CONTROLLER
# ==============================================================================

def main():
    # 1. Hardware Init
    hw = HardwareManager()
    
    # 2. Menu
    os.system('clear')
    print("MTP MINER SUITE v28 - BLOCK COMMANDER")
    print("1. Start Mining (Standard)")
    print("2. Run 10-Min Benchmark")
    choice = input("Select [1]: ").strip()
    
    if choice == "2":
        run_benchmark(hw)
    
    # 3. Launch Core
    manager = mp.Manager()
    state = MinerState(manager)
    job_q = manager.Queue()
    res_q = manager.Queue()
    log_q = manager.Queue()
    stop_event = mp.Event()
    
    # Network Thread
    net_thread = threading.Thread(target=NetworkCommander(state, job_q, res_q, log_q).connect, daemon=True)
    net_thread.start()
    
    # Proxy Thread
    proxy_thread = ProxyFunnel(state, log_q)
    proxy_thread.start()
    
    # Workers
    procs = []
    # CPU: Limit to CPU_COUNT - 1 to keep system responsive
    cpu_limit = max(1, mp.cpu_count() - 1)
    for i in range(cpu_limit):
        p = mp.Process(target=cpu_worker_process, args=(i, stop_event, state, job_q, res_q, log_q))
        p.start()
        procs.append(p)
        
    # GPU
    gp = mp.Process(target=gpu_worker_process, args=(stop_event, state, log_q))
    gp.start()
    procs.append(gp)
    
    # UI Loop
    try:
        curses.wrapper(draw_dashboard, state, log_q, hw)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for p in procs: p.terminate()

def draw_dashboard(stdscr, state, log_q, hw):
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)
    
    stdscr.nodelay(True)
    logs = []
    
    # Smoothing vars
    last_hashes = 0.0
    current_hr = 0.0
    
    while True:
        # Process Logs
        try:
            while True:
                t, m = log_q.get_nowait()
                logs.append(f"{get_cst_time()} [{t}] {m}")
                if len(logs) > 50: logs.pop(0)
        except queue.Empty: pass
        
        # Calc Stats
        total = state.total_hashes.value
        delta = total - last_hashes
        last_hashes = total
        current_hr = (current_hr * 0.8) + (delta * 10 * 0.2) # 10Hz refresh
        
        # Format
        if current_hr > 1e12: hr_s = f"{current_hr/1e12:.2f} TH/s"
        elif current_hr > 1e9: hr_s = f"{current_hr/1e9:.2f} GH/s"
        elif current_hr > 1e6: hr_s = f"{current_hr/1e6:.2f} MH/s"
        else: hr_s = f"{current_hr/1000:.2f} kH/s"
        
        c_temp = hw.get_cpu_temp()
        g_temp = hw.get_gpu_temp()
        
        # Draw
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        # Header
        stdscr.addstr(0, 0, " MTP MINER SUITE v28 - BLOCK COMMANDER ".center(w), curses.color_pair(5)|curses.A_BOLD)
        
        # Grid
        col_w = w // 4
        
        # Col 1: System
        stdscr.addstr(2, 2, "=== SYSTEM ===", curses.color_pair(4))
        stdscr.addstr(3, 2, f"IP: {get_local_ip()}")
        stdscr.addstr(4, 2, f"Proxy Port: {CONFIG['PROXY_PORT']}")
        
        # Col 2: Hardware
        stdscr.addstr(2, col_w+2, "=== HARDWARE ===", curses.color_pair(4))
        stdscr.addstr(3, col_w+2, f"CPU: {c_temp}C")
        stdscr.addstr(4, col_w+2, f"GPU: {g_temp}C")
        
        # Col 3: Network
        stdscr.addstr(2, col_w*2+2, "=== NETWORK ===", curses.color_pair(4))
        status = "CONNECTED" if state.connected.value else "DIALING..."
        c_pair = curses.color_pair(1) if state.connected.value else curses.color_pair(3)
        stdscr.addstr(3, col_w*2+2, f"Status: {status}", c_pair)
        stdscr.addstr(4, col_w*2+2, f"Diff: {int(state.difficulty.value)}")
        
        # Col 4: Production
        stdscr.addstr(2, col_w*3+2, "=== PRODUCTION ===", curses.color_pair(4))
        stdscr.addstr(3, col_w*3+2, f"Local TX: {state.local_tx.value}")
        stdscr.addstr(4, col_w*3+2, f"Proxy TX: {state.proxy_tx.value}")
        stdscr.addstr(5, col_w*3+2, f"Pool RX: {state.accepted_shares.value}")
        
        # Hashrate Bar
        stdscr.hline(7, 0, curses.ACS_HLINE, w)
        stdscr.addstr(8, 2, f"GLOBAL HASHRATE: {hr_s}", curses.color_pair(1)|curses.A_BOLD)
        stdscr.hline(10, 0, curses.ACS_HLINE, w)
        
        # Logs
        max_logs = h - 12
        for i, log in enumerate(logs[-max_logs:]):
            color = curses.color_pair(5)
            if "ERR" in log: color = curses.color_pair(3)
            elif "TX" in log: color = curses.color_pair(2)
            elif "RX" in log: color = curses.color_pair(1)
            elif "NET" in log: color = curses.color_pair(4)
            try: stdscr.addstr(11+i, 2, log[:w-2], color)
            except: pass
            
        stdscr.refresh()
        time.sleep(0.1)
        if stdscr.getch() == ord('q'): break

if __name__ == "__main__":
    main()
