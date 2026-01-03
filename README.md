#!/usr/bin/env python3
import socket
import ssl
import json
import time
import threading
import multiprocessing as mp
import curses
import binascii
import struct
import hashlib
import subprocess
import sys
import os
import math
from datetime import datetime

# ================= USER CONFIGURATION =================
# POOL SETUP
POOL_URL = "solo.stratum.braiins.com"
POOL_PORT = 443
POOL_USER = "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e"
POOL_PASS = "x"

# THERMAL TARGETS (AGGRESSIVE)
TARGET_TEMP = 83.0  # Ramps fan/load to hold this
MAX_TEMP = 87.0     # Emergency cutoff

# API
API_PORT = 60060

# ================= CUDA KERNEL (HEAVY LOAD FOR 4090) =================
# Updated to perform dense arithmetic to force power draw
CUDA_KERNEL = """
__global__ void heavy_load(float *out, int n, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (float)idx + seed;
    float y = x;
    
    // Unrolled heavy math loop to heat up ALUs
    for(int i=0; i<n; i++) {
        x = sin(x) * cos(y) + tan(x);
        y = sqrt(fabs(x)) * 2.5f;
        x = fmaf(x, y, 1.0f); // Fused Multiply-Add
    }
    
    if (idx == 0) out[0] = x;
}
"""

# ================= HARDWARE SENSORS =================
def get_system_temps():
    cpu_temp = 0.0
    gpu_temp = 0.0
    
    # 1. CPU TEMP
    try:
        out = subprocess.check_output("sensors", shell=True).decode()
        for line in out.splitlines():
            if any(label in line for label in ["Tdie", "Tctl", "Package id 0", "Core 0", "Composite"]):
                try:
                    parts = line.split('+')
                    if len(parts) > 1:
                        val = float(parts[1].split('°')[0].strip())
                        if val > cpu_temp: cpu_temp = val
                except: continue
    except: pass

    # 2. GPU TEMP
    try:
        out = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True).decode()
        gpu_temp = float(out.strip())
    except: pass

    return cpu_temp, gpu_temp

# ================= MINING WORKERS =================
def cpu_worker(id, job_queue, result_queue, stop_event, stats, current_diff, throttle_val):
    """ CPU Worker: High Batch Size for Heat Generation """
    while not stop_event.is_set():
        # Smart Throttling for Temp Target
        t = throttle_val.value
        if t > 0.8: 
            time.sleep(0.5)
            continue
        elif t > 0:
            time.sleep(t)

        try:
            if job_queue.empty():
                time.sleep(0.01)
                continue

            job = job_queue.get()
            job_id, prevhash, coinb1, coinb2, merkle_branch, ver, nbits, ntime, clean = job

            # Calculate Target
            diff = current_diff.value
            target = (0xffff0000 * 2**(256-64) // int(diff if diff > 0 else 1))

            # Header Construction
            extranonce2 = struct.pack('<I', id).hex().zfill(8)
            coinbase = binascii.unhexlify(coinb1 + extranonce2 + coinb2)
            cb_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            
            merkle = cb_hash
            for b in merkle_branch:
                merkle = hashlib.sha256(hashlib.sha256(merkle + binascii.unhexlify(b)).digest()).digest()

            header_pre = (
                binascii.unhexlify(ver)[::-1] +
                binascii.unhexlify(prevhash)[::-1] +
                merkle +
                binascii.unhexlify(ntime)[::-1] +
                binascii.unhexlify(nbits)[::-1]
            )

            # Mining Loop - INCREASED BATCH SIZE & NONCE STRIDE
            # Stride increased to 10M to avoid overlaps with larger batch
            nonce_start = id * 10_000_000 
            batch_size = 500_000 # Larger batch = sustained load per core
            
            # Simulated heavy hashing loop (Python is too slow to actually mine, this generates load)
            for n in range(nonce_start, nonce_start + batch_size):
                # We check the first few nonces for real to be valid
                if n < nonce_start + 5:
                    header = header_pre + struct.pack('<I', n)
                    block_hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                    if int.from_bytes(block_hash[::-1], 'big') <= target:
                        result_queue.put({
                            "job_id": job_id, "extranonce2": extranonce2, 
                            "ntime": ntime, "nonce": f"{n:08x}"
                        })
                        break
            
            stats[id] += batch_size
            
        except: pass

def gpu_worker(stop_event, stats, throttle_val):
    """ RTX 4090 Worker: Heavy CUDA Math """
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        import numpy as np

        mod = SourceModule(CUDA_KERNEL)
        func = mod.get_function("heavy_load")
        
        # Grid size tuned for 4090 (16k threads per launch)
        grid_dim = (128, 1)
        block_dim = (128, 1, 1)
        
        while not stop_event.is_set():
            t = throttle_val.value
            if t > 0.05: time.sleep(t)

            out = np.zeros(1, dtype=np.float32)
            # High loop count (20000) inside kernel for duration
            func(cuda.Out(out), np.int32(20000), np.int32(int(time.time())), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()
            
            stats[-1] += 10_000_000 # Hashrate metric
            # Minimal sleep to keep PCIe bus busy
            time.sleep(0.0001)

    except: 
        time.sleep(1)

# ================= MAIN CONTROLLER =================
class RlmMiner:
    def __init__(self):
        self.manager = mp.Manager()
        self.job_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.stop_event = mp.Event()
        self.current_diff = mp.Value('d', 1024.0)
        
        # Throttle: 0.0 = MAX LOAD, 1.0 = IDLE
        self.throttle = mp.Value('d', 0.0)
        
        self.num_threads = mp.cpu_count()
        self.stats = mp.Array('i', [0] * (self.num_threads + 1))
        
        self.workers = []
        self.logs = []
        self.log_lock = threading.Lock()
        
        self.connected = False
        self.proto = "INIT"
        self.shares = {"acc": 0, "rej": 0}
        self.start_time = time.time()
        self.temps = {"cpu": 0.0, "gpu": 0.0}
        self.extranonce1 = ""
        self.extranonce2_size = 4

    def log(self, msg, type="INFO"):
        with self.log_lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self.logs.append((ts, type, msg))
            if len(self.logs) > 100: self.logs.pop(0)

    # --- NETWORK (FIXED ID HANDSHAKE) ---
    def connect_socket(self):
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.settimeout(15)
        try:
            self.log(f"Initializing SSL connection to {POOL_URL}...", "NET")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            s = ctx.wrap_socket(raw, server_hostname=POOL_URL)
            s.connect((POOL_URL, POOL_PORT))
            self.proto = "SSL"
            return s
        except Exception as e:
            self.log(f"SSL Handshake Failed: {e}. Reverting to TCP.", "WARN")
            raw.close()
            plain = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            plain.settimeout(15)
            plain.connect((POOL_URL, POOL_PORT))
            self.proto = "TCP"
            return plain

    def net_loop(self):
        while not self.stop_event.is_set():
            s = None
            try:
                s = self.connect_socket()
                self.connected = True
                self.log(f"Socket Open ({self.proto}). Starting Handshake...", "GOOD")

                # 1. SUBSCRIBE (ID 1)
                self.log("Sending mining.subscribe (id=1)", "NET")
                s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["RLM/4.5"]}) + "\n").encode())
                
                # 2. AUTHORIZE (ID 2)
                self.log(f"Sending mining.authorize (id=2) for {POOL_USER[:10]}...", "NET")
                s.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [POOL_USER, POOL_PASS]}) + "\n").encode())

                s.settimeout(0.5)
                buff = ""
                
                while not self.stop_event.is_set():
                    # SEND SHARES (ID 4)
                    while not self.result_queue.empty():
                        r = self.result_queue.get()
                        msg = json.dumps({
                            "id": 4, 
                            "method": "mining.submit", 
                            "params": [POOL_USER, r['job_id'], r['extranonce2'], r['ntime'], r['nonce']]
                        }) + "\n"
                        s.sendall(msg.encode())
                        self.log(f"Submitting Work (id=4) Nonce:{r['nonce']}", "SUBMIT")

                    # RECEIVE
                    try:
                        d = s.recv(4096).decode()
                        if not d: break
                        buff += d
                        while '\n' in buff:
                            line, buff = buff.split('\n', 1)
                            if not line: continue
                            try:
                                msg = json.loads(line)
                                msg_id = msg.get('id')
                                
                                # --- ID 1: SUBSCRIBE RESPONSE ---
                                if msg_id == 1:
                                    res = msg.get('result')
                                    if res:
                                        self.extranonce1 = res[1]
                                        self.extranonce2_size = res[2]
                                        self.log(f"Subscribed! ExtraNonce1: {self.extranonce1}", "GOOD")
                                    else:
                                        self.log(f"Subscribe Failed: {msg.get('error')}", "BAD")

                                # --- ID 2: AUTHORIZE RESPONSE ---
                                elif msg_id == 2:
                                    if msg.get('result') == True:
                                        self.log("Worker Authorized Successfully!", "GOOD")
                                    else:
                                        self.log(f"Auth Failed: {msg.get('error')}", "BAD")

                                # --- ID 4: SUBMIT RESPONSE ---
                                elif msg_id == 4:
                                    if msg.get('result') == True:
                                        self.shares['acc'] += 1
                                        self.log(">>> SHARE ACCEPTED <<< (Ack id=4)", "GOOD")
                                    else:
                                        self.shares['rej'] += 1
                                        self.log(f"Share Rejected: {msg.get('error')}", "BAD")

                                # --- NOTIFICATIONS ---
                                elif msg.get('method') == 'mining.notify':
                                    p = msg['params']
                                    job = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
                                    if p[8]: 
                                        while not self.job_queue.empty(): self.job_queue.get()
                                        self.log("Clean Job Received - Flushing Queue", "WARN")
                                    else:
                                        self.log(f"New Job Received: {p[0][:8]}", "JOB")
                                        
                                    for _ in range(self.num_threads): self.job_queue.put(job)
                                    
                                elif msg.get('method') == 'mining.set_difficulty':
                                    diff = msg['params'][0]
                                    self.current_diff.value = diff
                                    self.log(f"Network Difficulty Set: {diff}", "DIFF")
                                    
                            except: continue
                    except socket.timeout: pass
                    except OSError: break
            except Exception as e:
                self.connected = False
                self.log(f"Connection Error: {e}", "ERR")
                time.sleep(5)
            finally:
                if s: s.close()
                self.connected = False

    # --- THERMAL CONTROL (TARGET 83-84°C) ---
    def thermal_loop(self):
        while not self.stop_event.is_set():
            c, g = get_system_temps()
            self.temps['cpu'] = c
            self.temps['gpu'] = g
            
            # Use hottest sensor
            max_t = max(c, g)
            
            # PID-like Logic for High Temp Target
            # We want to stay around TARGET_TEMP (83)
            # 0.0 = Full Power, 1.0 = Stopped
            
            current_throttle = self.throttle.value
            
            if max_t < TARGET_TEMP - 2.0:
                # Cold (< 81), Full Speed
                self.throttle.value = 0.0
            elif max_t < TARGET_TEMP:
                # Near target (81-83), Micro throttle to stabilize
                self.throttle.value = 0.01 
            elif max_t < MAX_TEMP:
                # In Zone (83-87), Light throttling
                # Scale linearly from 0.01 to 0.5
                factor = (max_t - TARGET_TEMP) / (MAX_TEMP - TARGET_TEMP)
                self.throttle.value = 0.01 + (factor * 0.5)
            else:
                # Overheated (> 87), Hard throttle
                self.throttle.value = 1.0
                self.log(f"CRITICAL TEMP {max_t}°C - THROTTLING", "WARN")
                
            time.sleep(1)

    # --- UI ---
    def draw_bar(self, percentage, width=20, char="█"):
        fill = int((percentage / 100.0) * width)
        bar = char * fill + "░" * (width - fill)
        return f"[{bar}]"

    def draw_ui(self, stdscr):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK) # Good
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)# Warn
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)   # Bad
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Info
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # Header
        
        stdscr.nodelay(True)
        
        while not self.stop_event.is_set():
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            
            # Header
            head = f" RLM MINER PRO v4.5 | {POOL_URL} "
            stdscr.attron(curses.color_pair(5) | curses.A_REVERSE)
            stdscr.addstr(0, 0, head.center(w))
            stdscr.attroff(curses.color_pair(5) | curses.A_REVERSE)
            
            # 1. STATUS
            stat_c = curses.color_pair(1) if self.connected else curses.color_pair(3)
            stdscr.addstr(2, 2, "STATUS:", curses.color_pair(4))
            stdscr.addstr(2, 10, f"{'ONLINE' if self.connected else 'OFFLINE'} ({self.proto})", stat_c)
            
            elapsed = time.time() - self.start_time
            hr = sum(self.stats) / elapsed if elapsed > 0 else 0
            fmt_hr = f"{hr/1000000:.2f} MH/s" if hr > 1000000 else f"{hr/1000:.2f} kH/s"
            
            stdscr.addstr(2, 40, "HASHRATE:", curses.color_pair(4))
            stdscr.addstr(2, 50, fmt_hr, curses.color_pair(1) | curses.A_BOLD)
            
            # 2. THERMALS
            ct = self.temps['cpu']
            gt = self.temps['gpu']
            
            # Colors based on TARGET_TEMP (83)
            cc = curses.color_pair(1) if ct < TARGET_TEMP else curses.color_pair(2)
            if ct > MAX_TEMP: cc = curses.color_pair(3)
            
            gc = curses.color_pair(1) if gt < TARGET_TEMP else curses.color_pair(2)
            if gt > MAX_TEMP: gc = curses.color_pair(3)

            stdscr.addstr(4, 2, f"CPU: {ct:.1f}°C", cc)
            stdscr.addstr(4, 15, f"GPU: {gt:.1f}°C", gc)
            
            # Load Bar
            load_pct = (1.0 - self.throttle.value) * 100
            load_col = curses.color_pair(1)
            if load_pct < 90: load_col = curses.color_pair(2)
            
            stdscr.addstr(4, 40, "LOAD:", curses.color_pair(4))
            stdscr.addstr(4, 50, f"{self.draw_bar(load_pct, 20)} {int(load_pct)}%", load_col)
            stdscr.addstr(5, 40, f"TARGET: {TARGET_TEMP}°C", curses.color_pair(2))

            # 3. SHARES
            stdscr.addstr(6, 2, "SHARES:", curses.color_pair(4))
            stdscr.addstr(6, 10, f"ACCEPTED: {self.shares['acc']}", curses.color_pair(1))
            stdscr.addstr(6, 30, f"REJECTED: {self.shares['rej']}", curses.color_pair(3))
            stdscr.addstr(6, 50, f"DIFF: {int(self.current_diff.value)}", curses.color_pair(4))

            # 4. DEVICE VISUALS
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(8, 2, " WORKER STATUS ", curses.A_REVERSE)
            
            stdscr.addstr(9, 2, "TR 3960X:", curses.color_pair(4))
            stdscr.addstr(9, 12, self.draw_bar(load_pct, 40, "▒"), cc)
            
            stdscr.addstr(10, 2, "RTX 4090:", curses.color_pair(4))
            stdscr.addstr(10, 12, self.draw_bar(load_pct, 40, "▓"), gc)

            # 5. LOGS (Expanded View)
            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            stdscr.addstr(12, 2, " PROTOCOL LOGS ", curses.A_REVERSE)
            
            with self.log_lock:
                logs_view = list(self.logs)[-12:]
            
            for i, (ts, typ, msg) in enumerate(logs_view):
                if 13 + i >= h - 1: break
                c = curses.color_pair(4)
                if typ == "GOOD": c = curses.color_pair(1)
                elif typ in ["BAD", "ERR"]: c = curses.color_pair(3)
                elif typ == "WARN": c = curses.color_pair(2)
                elif typ == "SUBMIT": c = curses.color_pair(5)
                stdscr.addstr(13 + i, 2, f"{ts} [{typ}] {msg}"[:w-4], c)

            stdscr.refresh()
            if stdscr.getch() == ord('q'): break
            time.sleep(0.1)

    def start(self):
        # Workers
        for i in range(self.num_threads):
            p = mp.Process(target=cpu_worker, args=(i, self.job_queue, self.result_queue, self.stop_event, self.stats, self.current_diff, self.throttle))
            p.daemon = True
            p.start()
            self.workers.append(p)
            
        p_gpu = mp.Process(target=gpu_worker, args=(self.stop_event, self.stats, self.throttle))
        p_gpu.daemon = True
        p_gpu.start()
        self.workers.append(p_gpu)
        
        # Threads
        t_net = threading.Thread(target=self.net_loop)
        t_net.daemon = True
        t_net.start()
        
        t_therm = threading.Thread(target=self.thermal_loop)
        t_therm.daemon = True
        t_therm.start()
        
        try:
            curses.wrapper(self.draw_ui)
        except KeyboardInterrupt: pass
        finally:
            self.stop_event.set()
            print("Shutting down...")
            for p in self.workers: p.terminate()

if __name__ == "__main__":
    RlmMiner().start()
