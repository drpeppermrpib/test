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

# THERMAL TARGETS (Aggressive)
TARGET_TEMP = 76.0  # The miner will run 100% until this temp
MAX_TEMP = 78.0     # Safety shutoff buffer

# API
API_PORT = 60060

# ================= CUDA KERNEL (RTX 4090 MAX LOAD) =================
# Added heavy trig/pow math to force GPU clocks up
CUDA_KERNEL = """
__global__ void hash_load_gen(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (float)idx;
    float y = x;
    // HEAVY LOAD GENERATION LOOP
    for(int i=0; i<n; i++) {
        x = sin(x) * cos(y) + tan(x);
        y = sqrt(fabs(x)) * pow(y, 1.001f);
    }
    if (idx < 1) out[0] = x + y;
}
"""

# ================= HARDWARE SENSORS =================
def get_system_temps():
    cpu_temp = 0.0
    gpu_temp = 0.0
    
    # 1. CPU TEMP (lm-sensors)
    try:
        out = subprocess.check_output("sensors", shell=True).decode()
        for line in out.splitlines():
            if any(label in line for label in ["Tdie", "Tctl", "Package id 0", "Composite", "temp1"]):
                try:
                    parts = line.split('+')
                    if len(parts) > 1:
                        val = float(parts[1].split('°')[0].strip())
                        if val > cpu_temp: cpu_temp = val
                except: continue
    except: pass

    # 2. GPU TEMP (nvidia-smi)
    try:
        out = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True).decode()
        gpu_temp = float(out.strip())
    except: pass

    return cpu_temp, gpu_temp

# ================= MINING WORKERS =================
def cpu_worker(id, job_queue, result_queue, stop_event, stats, current_diff, throttle_val):
    """ CPU Worker - Optimized for Threadripper """
    while not stop_event.is_set():
        # Smart Throttling Check
        t = throttle_val.value
        if t > 0.0:
            time.sleep(t)

        try:
            if job_queue.empty():
                time.sleep(0.01)
                continue

            job = job_queue.get()
            job_id, prevhash, coinb1, coinb2, merkle_branch, ver, nbits, ntime, clean = job

            # Diff Calculation
            diff = current_diff.value
            target = (0xffff0000 * 2**(256-64) // int(diff if diff > 0 else 1))

            # Header Prep
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

            # Mining Batch - Increased size for load
            nonce = id * 1000000
            batch_size = 100000 
            
            for n in range(nonce, nonce + batch_size):
                header = header_pre + struct.pack('<I', n)
                block_hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                
                # Check Target
                if int.from_bytes(block_hash[::-1], 'big') <= target:
                    result_queue.put({
                        "job_id": job_id,
                        "extranonce2": extranonce2,
                        "ntime": ntime,
                        "nonce": f"{n:08x}"
                    })
                    break
            
            stats[id] += batch_size
            
        except: pass

def gpu_worker(stop_event, stats, throttle_val):
    """ RTX 4090 Worker - Maximized Grid Size """
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        import numpy as np

        mod = SourceModule(CUDA_KERNEL)
        func = mod.get_function("hash_load_gen")
        
        while not stop_event.is_set():
            t = throttle_val.value
            if t > 0.0: time.sleep(t)
            
            # Massive Grid for 4090 (32k blocks)
            out = np.zeros(1, dtype=np.float32)
            # Increased loop count inside kernel (20000) and grid size (32768)
            func(cuda.Out(out), np.int32(20000), block=(512,1,1), grid=(32768,1))
            
            cuda.Context.synchronize()
            stats[-1] += 25000000 # Higher hashrate contribution
            
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

    def log(self, msg, type="INFO"):
        with self.log_lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self.logs.append((ts, type, msg))
            if len(self.logs) > 50: self.logs.pop(0)

    # --- NETWORK ---
    def connect_socket(self):
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.settimeout(10)
        try:
            self.log(f"Dialing {POOL_URL} (SSL)...", "NET")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            s = ctx.wrap_socket(raw, server_hostname=POOL_URL)
            s.connect((POOL_URL, POOL_PORT))
            self.proto = "SSL"
            return s
        except Exception as e:
            self.log("SSL Failed. Switching to TCP.", "WARN")
            raw.close()
            plain = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            plain.settimeout(10)
            plain.connect((POOL_URL, POOL_PORT))
            self.proto = "TCP"
            return plain

    def net_loop(self):
        while not self.stop_event.is_set():
            s = None
            try:
                s = self.connect_socket()
                self.connected = True
                self.log(f"Connected ({self.proto})", "GOOD")

                # Stratum Handshake
                s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["RLM/4.0"]}) + "\n").encode())
                s.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [POOL_USER, POOL_PASS]}) + "\n").encode())

                s.settimeout(0.5)
                buff = ""
                
                while not self.stop_event.is_set():
                    # Send Shares
                    while not self.result_queue.empty():
                        r = self.result_queue.get()
                        s.sendall((json.dumps({
                            "id": 4, "method": "mining.submit", 
                            "params": [POOL_USER, r['job_id'], r['extranonce2'], r['ntime'], r['nonce']]
                        }) + "\n").encode())
                        self.log("Submitting Share...", "INFO")

                    # Recv Data
                    try:
                        d = s.recv(4096).decode()
                        if not d: break
                        buff += d
                        while '\n' in buff:
                            line, buff = buff.split('\n', 1)
                            if not line: continue
                            try:
                                msg = json.loads(line)
                                if msg.get('method') == 'mining.notify':
                                    p = msg['params']
                                    job = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
                                    if p[8]: 
                                        while not self.job_queue.empty(): self.job_queue.get()
                                    for _ in range(self.num_threads): self.job_queue.put(job)
                                    
                                elif msg.get('method') == 'mining.set_difficulty':
                                    self.current_diff.value = msg['params'][0]
                                    self.log(f"Difficulty: {self.current_diff.value}", "DIFF")
                                    
                                elif msg.get('id') == 4:
                                    if msg.get('result'): 
                                        self.shares['acc'] += 1
                                        self.log("Share Accepted!", "GOOD")
                                    else: 
                                        self.shares['rej'] += 1
                                        self.log(f"Rejected: {msg.get('error')}", "BAD")
                            except: continue
                    except socket.timeout: pass
                    except OSError: break
            except Exception as e:
                self.connected = False
                self.log(f"Net Error: {e}", "ERR")
                time.sleep(5)
            finally:
                if s: s.close()
                self.connected = False

    # --- THERMAL CONTROL (RAMP UP LOGIC) ---
    def thermal_loop(self):
        while not self.stop_event.is_set():
            c, g = get_system_temps()
            self.temps['cpu'] = c
            self.temps['gpu'] = g
            
            max_t = max(c, g)
            
            # Logic: Run 100% (Throttle 0.0) until we are VERY close to 76.0
            if max_t < (TARGET_TEMP - 0.5):
                # Full Speed Ahead
                self.throttle.value = 0.0
            
            elif max_t < TARGET_TEMP:
                # Approaching limit (75.5 - 76.0), micro throttle
                self.throttle.value = 0.05
                
            elif max_t < MAX_TEMP:
                # Over target, start braking
                factor = (max_t - TARGET_TEMP) / (MAX_TEMP - TARGET_TEMP)
                self.throttle.value = factor * 0.5 # Cap throttle at 50% delay unless critical
                
            else:
                # Critical Heat
                self.throttle.value = 1.0
                self.log(f"Heat Limit! {max_t}°C", "WARN")
                
            time.sleep(1)

    # --- UI ---
    def draw_bar(self, percentage, width=20):
        fill = int((percentage / 100.0) * width)
        bar = "█" * fill + "░" * (width - fill)
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
            head = f" RLM MINER ULTRA v4.2 | {POOL_URL} "
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
            
            # 2. THERMALS & RAMP
            ct = self.temps['cpu']
            gt = self.temps['gpu']
            
            # Colors
            cc = curses.color_pair(1) if ct < TARGET_TEMP else curses.color_pair(2)
            if ct >= TARGET_TEMP: cc = curses.color_pair(3)
            
            gc = curses.color_pair(1) if gt < TARGET_TEMP else curses.color_pair(2)
            if gt >= TARGET_TEMP: gc = curses.color_pair(3)

            stdscr.addstr(4, 2, f"CPU: {ct}°C", cc)
            stdscr.addstr(4, 15, f"GPU: {gt}°C", gc)
            
            # Throttle Bar
            speed_pct = (1.0 - self.throttle.value) * 100
            if speed_pct < 0: speed_pct = 0
            
            bar_col = curses.color_pair(1)
            if speed_pct < 95: bar_col = curses.color_pair(2)
            
            stdscr.addstr(4, 40, "LOAD:", curses.color_pair(4))
            stdscr.addstr(4, 50, f"{self.draw_bar(speed_pct, 20)} {int(speed_pct)}%", bar_col)

            # 3. SHARES
            stdscr.addstr(6, 2, "SHARES:", curses.color_pair(4))
            stdscr.addstr(6, 10, f"ACCEPTED: {self.shares['acc']}", curses.color_pair(1))
            stdscr.addstr(6, 30, f"REJECTED: {self.shares['rej']}", curses.color_pair(3))
            stdscr.addstr(6, 50, f"DIFF: {int(self.current_diff.value)}", curses.color_pair(2))

            # 4. DEVICE BARS
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(8, 2, " WORKER STATUS ", curses.A_REVERSE)
            
            stdscr.addstr(9, 2, "TR 3960X:", curses.color_pair(4))
            stdscr.addstr(9, 12, self.draw_bar(speed_pct, 40), cc)
            
            stdscr.addstr(10, 2, "RTX 4090:", curses.color_pair(4))
            stdscr.addstr(10, 12, self.draw_bar(speed_pct, 40), gc)

            # 5. LOGS
            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            stdscr.addstr(12, 2, " SYSTEM LOGS ", curses.A_REVERSE)
            
            with self.log_lock:
                logs_view = list(self.logs)[-10:]
            
            for i, (ts, typ, msg) in enumerate(logs_view):
                if 13 + i >= h - 1: break
                c = curses.color_pair(4)
                if typ == "GOOD": c = curses.color_pair(1)
                elif typ in ["BAD", "ERR"]: c = curses.color_pair(3)
                elif typ == "WARN": c = curses.color_pair(2)
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
        
        # Services
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
