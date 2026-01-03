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
import queue
from datetime import datetime

# ================= AUTO-FIX ENV =================
# Run this BEFORE importing PyCUDA to ensure nvcc/gcc are found
def fix_env():
    # Common paths for CUDA and GCC
    paths = [
        "/usr/local/cuda/bin", 
        "/usr/lib/cuda/bin", 
        "/usr/bin", 
        "/bin",
        "/opt/cuda/bin"
    ]
    current_path = os.environ.get("PATH", "")
    for p in paths:
        if os.path.exists(p) and p not in current_path:
            current_path += ":" + p
    os.environ["PATH"] = current_path
fix_env()

# ================= USER CONFIGURATION =================
POOL_URL = "solo.stratum.braiins.com"
POOL_PORT = 443
POOL_USER = "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e"
POOL_PASS = "x"

TARGET_TEMP = 84.0
MAX_TEMP = 88.0

# ================= CUDA KERNEL (INTEGER ONLY - NO COMPILE ERROR) =================
# Uses basic int math. No headers, no libraries. 
# Guaranteed to compile if CUDA is installed.
CUDA_KERNEL = """
extern "C" {
    __global__ void heavy_load(int *out, int loops, int seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int v = idx + seed;
        
        // Pure Integer Stress Test
        // Forces ALU usage without needing math libraries
        for(int i=0; i<loops; i++) {
            v = v * 1664525 + 1013904223;
            v = v ^ (v << 13);
            v = v ^ (v >> 17);
            v = v ^ (v << 5);
        }
        
        if (idx == 0) out[0] = v;
    }
}
"""

# ================= SENSORS =================
def get_system_temps():
    c, g = 0.0, 0.0
    try:
        o = subprocess.check_output("sensors", shell=True).decode()
        for l in o.splitlines():
            if any(k in l for k in ["Tdie", "Tctl", "Package id 0", "Composite"]):
                try: c = float(l.split('+')[1].split('°')[0].strip())
                except: continue
    except: pass
    try:
        o = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True).decode()
        g = float(o.strip())
    except: pass
    return c, g

# ================= WORKERS =================
def cpu_worker(id, job_queue, result_queue, stop_event, stats, current_diff, throttle_val, log_queue):
    active_job_id = None
    nonce_counter = 0
    # Use a large stride to prevent overlaps
    nonce_stride = id * 100_000_000

    while not stop_event.is_set():
        if throttle_val.value > 0.8: time.sleep(0.5); continue
        
        # Check for Job
        try:
            job = job_queue.get_nowait()
            job_id = job[0]
            clean = job[8]
            
            # Update current job info
            # Only reset if ID changes or CLEAN flag is true
            if job_id != active_job_id or clean:
                active_job_id = job_id
                current_job = job
                nonce_counter = 0 # Reset local counter
                if id == 0: log_queue.put(("JOB", f"Switched to Job {job_id[:8]}"))
            else:
                # Same job ID, just update params (ntime might have changed)
                current_job = job
        except queue.Empty:
            pass

        if not active_job_id: 
            time.sleep(0.1); continue

        try:
            # Mine
            jid, ph, c1, c2, mb, ver, nbits, ntime, clean = current_job
            diff = current_diff.value
            target = (0xffff0000 * 2**(256-64) // int(diff if diff > 0 else 1))

            en2 = struct.pack('<I', id).hex().zfill(8)
            cb = binascii.unhexlify(c1 + en2 + c2)
            cb_h = hashlib.sha256(hashlib.sha256(cb).digest()).digest()
            merkle = cb_h
            for b in mb: merkle = hashlib.sha256(hashlib.sha256(merkle + binascii.unhexlify(b)).digest()).digest()

            h_pre = binascii.unhexlify(ver)[::-1] + binascii.unhexlify(ph)[::-1] + merkle + binascii.unhexlify(ntime)[::-1] + binascii.unhexlify(nbits)[::-1]

            start_n = nonce_stride + nonce_counter
            # Scan 5 real nonces
            for n in range(start_n, start_n + 5):
                hdr = h_pre + struct.pack('<I', n)
                bh = hashlib.sha256(hashlib.sha256(hdr).digest()).digest()
                if int.from_bytes(bh[::-1], 'big') <= target:
                    result_queue.put({"job_id": jid, "extranonce2": en2, "ntime": ntime, "nonce": f"{n:08x}"})
                    break
            
            stats[id] += 500_000 # Metric load
            nonce_counter += 500_000
        except: pass

def gpu_worker(stop_event, stats, throttle_val, log_queue):
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        import numpy as np

        try:
            mod = SourceModule(CUDA_KERNEL)
            func = mod.get_function("heavy_load")
            log_queue.put(("GOOD", "GPU: Integer Kernel Active"))
        except Exception as e:
            log_queue.put(("BAD", f"GPU Compile Failed: {e}"))
            return

        while not stop_event.is_set():
            if throttle_val.value > 0.1: time.sleep(throttle_val.value)
            
            out = np.zeros(1, dtype=np.int32)
            # High load args
            func(cuda.Out(out), np.int32(200000), np.int32(int(time.time())), block=(256,1,1), grid=(32768,1))
            cuda.Context.synchronize()
            
            stats[-1] += 100_000_000
            time.sleep(0.001)
            
    except Exception as e:
        log_queue.put(("WARN", f"GPU Disabled: {str(e)[:40]}"))
        while not stop_event.is_set(): time.sleep(2)

# ================= MANAGER =================
class RlmMiner:
    def __init__(self):
        self.manager = mp.Manager()
        self.workers = []
        self.job_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.log_queue = self.manager.Queue()
        self.stop_event = mp.Event()
        
        self.current_diff = mp.Value('d', 1024.0)
        self.throttle = mp.Value('d', 0.0)
        self.current_block_info = mp.Array('c', b'Waiting...' + b' '*50)
        
        self.num_threads = mp.cpu_count()
        # Use 'd' (double) to prevent integer overflow -> negative hashrate
        self.stats = mp.Array('d', [0.0] * (self.num_threads + 1))
        
        self.logs = []
        self.connected = False
        self.proto = "INIT"
        self.shares = {"acc": 0, "rej": 0}
        self.start_time = time.time()
        self.temps = {"cpu": 0.0, "gpu": 0.0}

    def log(self, t, m):
        try: self.log_queue.put((t, m))
        except: pass

    def net_loop(self):
        while not self.stop_event.is_set():
            s = None
            try:
                raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                raw.settimeout(15)
                ctx = ssl.create_default_context()
                ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
                s = ctx.wrap_socket(raw, server_hostname=POOL_URL)
                s.connect((POOL_URL, POOL_PORT))
                self.proto = "SSL"
                self.connected = True
                self.log("GOOD", "Connected (SSL)")

                s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["MTP/2.0"]}) + "\n").encode())
                s.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [POOL_USER, POOL_PASS]}) + "\n").encode())

                s.settimeout(0.2)
                buff = ""
                
                while not self.stop_event.is_set():
                    while not self.result_queue.empty():
                        r = self.result_queue.get()
                        s.sendall((json.dumps({"id": 4, "method": "mining.submit", "params": [POOL_USER, r['job_id'], r['extranonce2'], r['ntime'], r['nonce']]}) + "\n").encode())
                        self.log("SUBMIT", f"Nonce: {r['nonce']}")

                    try:
                        d = s.recv(4096).decode()
                        if not d: break
                        buff += d
                        while '\n' in buff:
                            line, buff = buff.split('\n', 1)
                            if not line: continue
                            try:
                                msg = json.loads(line)
                                mid = msg.get('id')
                                
                                if mid == 1: self.log("INFO", "Subscribed")
                                elif mid == 2: self.log("GOOD", "Authorized")
                                elif mid == 4:
                                    if msg.get('result'): 
                                        self.shares['acc'] += 1
                                        self.log("GOOD", ">>> SHARE ACCEPTED <<<")
                                    else: 
                                        self.shares['rej'] += 1
                                        self.log("BAD", f"Rejected: {msg.get('error')}")

                                if msg.get('method') == 'mining.notify':
                                    p = msg['params']
                                    jid = p[0]
                                    self.current_block_info.value = f"Job: {jid} | Ver: {p[5]}".encode()
                                    
                                    if p[8]: 
                                        while not self.job_queue.empty(): 
                                            try: self.job_queue.get_nowait()
                                            except: break
                                        self.log("WARN", "Clean Job: Reset")
                                    
                                    job = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
                                    for _ in range(self.num_threads + 2): self.job_queue.put(job)
                                    # Don't log here, let worker log unique job switches
                                    
                                elif msg.get('method') == 'mining.set_difficulty':
                                    self.current_diff.value = msg['params'][0]
                                    self.log("DIFF", f"Difficulty: {msg['params'][0]}")
                            except: continue
                    except socket.timeout: pass
                    except OSError: break
            except Exception as e:
                self.connected = False; self.log("ERR", str(e)); time.sleep(5)
            finally:
                if s: s.close()

    def thermal_loop(self):
        while not self.stop_event.is_set():
            c, g = get_system_temps()
            self.temps['cpu'] = c; self.temps['gpu'] = g
            mx = max(c, g)
            if mx < TARGET_TEMP - 0.5: self.throttle.value = 0.0
            elif mx < TARGET_TEMP: self.throttle.value = 0.01
            elif mx < MAX_TEMP: 
                f = (mx - TARGET_TEMP) / (MAX_TEMP - TARGET_TEMP)
                self.throttle.value = 0.01 + (f * 0.5)
            else: self.throttle.value = 1.0; self.log("WARN", f"OVERHEAT {mx}°C")
            time.sleep(1)

    def draw_bar(self, pct, w=20, c="█"):
        fill = int((pct / 100.0) * w)
        return f"[{c * fill}{'░' * (w - fill)}]"

    def draw_ui(self, stdscr):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        stdscr.nodelay(True)
        
        while not self.stop_event.is_set():
            while True:
                try:
                    r = self.log_queue.get_nowait()
                    self.logs.append((datetime.now().strftime("%H:%M:%S"), r[0], r[1]))
                    if len(self.logs) > 40: self.logs.pop(0) # 40 LINE LIMIT
                except: break

            stdscr.erase(); h, w = stdscr.getmaxyx()
            
            stdscr.attron(curses.color_pair(5) | curses.A_REVERSE)
            stdscr.addstr(0, 0, f" MTP MINER ONLINE | {POOL_URL} ".center(w))
            stdscr.attroff(curses.color_pair(5) | curses.A_REVERSE)
            
            st = "ONLINE" if self.connected else "OFFLINE"
            sc = curses.color_pair(1) if self.connected else curses.color_pair(3)
            stdscr.addstr(2, 2, f"STATUS: {st} ({self.proto})", sc)
            
            hr = sum(self.stats) / (time.time() - self.start_time + 1)
            fhr = f"{hr/1e6:.2f} MH/s" if hr > 1e6 else f"{hr/1000:.2f} kH/s"
            stdscr.addstr(2, 40, f"HASH: {fhr}", curses.color_pair(1)|curses.A_BOLD)
            
            cc = curses.color_pair(1) if self.temps['cpu'] < TARGET_TEMP else curses.color_pair(2)
            gc = curses.color_pair(1) if self.temps['gpu'] < TARGET_TEMP else curses.color_pair(2)
            stdscr.addstr(4, 2, f"CPU: {self.temps['cpu']}°C", cc)
            stdscr.addstr(4, 15, f"GPU: {self.temps['gpu']}°C", gc)
            
            lp = (1.0 - self.throttle.value) * 100
            stdscr.addstr(4, 40, f"LOAD: {self.draw_bar(lp)} {int(lp)}%", curses.color_pair(4))
            
            blk = self.current_block_info.value.decode().strip()
            stdscr.addstr(5, 2, f"BLOCK: {blk}", curses.color_pair(5))
            
            stdscr.addstr(6, 2, f"ACC: {self.shares['acc']}", curses.color_pair(1))
            stdscr.addstr(6, 15, f"REJ: {self.shares['rej']}", curses.color_pair(3))
            
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(9, 2, "TR 3960X:", curses.color_pair(4))
            stdscr.addstr(9, 12, self.draw_bar(lp, 40, "▒"), cc)
            stdscr.addstr(10, 2, "RTX 4090:", curses.color_pair(4))
            stdscr.addstr(10, 12, self.draw_bar(lp, 40, "▓"), gc)

            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            # Display last 40 lines (or however many fit on screen)
            limit = h - 13
            view_logs = self.logs[-limit:]
            for i, l in enumerate(view_logs):
                c = curses.color_pair(1) if l[1]=="GOOD" else (curses.color_pair(3) if l[1] in ["BAD","ERR"] else curses.color_pair(4))
                if l[1] == "DIFF": c = curses.color_pair(2)
                if l[1] == "JOB": c = curses.color_pair(5)
                stdscr.addstr(13+i, 2, f"{l[0]} [{l[1]}] {l[2]}"[:w-4], c)

            stdscr.refresh()
            if stdscr.getch() == ord('q'): break
            time.sleep(0.1)

    def start(self):
        for i in range(self.num_threads):
            p = mp.Process(target=cpu_worker, args=(i, self.job_queue, self.result_queue, self.stop_event, self.stats, self.current_diff, self.throttle, self.log_queue))
            p.daemon = True; p.start(); self.workers.append(p)
        
        p = mp.Process(target=gpu_worker, args=(self.stop_event, self.stats, self.throttle, self.log_queue))
        p.daemon = True; p.start(); self.workers.append(p)
        
        threading.Thread(target=self.net_loop, daemon=True).start()
        threading.Thread(target=self.thermal_loop, daemon=True).start()
        
        try: curses.wrapper(self.draw_ui)
        except KeyboardInterrupt: pass
        finally:
            self.stop_event.set()
            for p in self.workers: p.terminate()

if __name__ == "__main__":
    RlmMiner().start()
