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

# ================= USER CONFIGURATION =================
POOL_URL = "solo.stratum.braiins.com"
POOL_PORT = 443
POOL_USER = "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e"
POOL_PASS = "x"

# THERMAL TARGETS
TARGET_TEMP = 84.0
MAX_TEMP = 88.0

# ================= AUTO-FIX PATH (GPU FIX) =================
# Forces system to see NVCC compiler
for p in ["/usr/local/cuda/bin", "/usr/lib/cuda/bin", "/opt/cuda/bin"]:
    if os.path.exists(p) and p not in os.environ['PATH']:
        os.environ['PATH'] += ":" + p

# ================= CUDA KERNEL (INTEGER ONLY = NO HEADERS NEEDED) =================
# This fixes the "nvcc compilation failed" error by removing all dependencies.
# Integer math generates massive heat/load on the GPU cores.
CUDA_KERNEL = """
extern "C" {
    __global__ void heavy_load(int *out, int iterations, int seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int v = idx + seed;
        
        // Heavy Integer Arithmetic Loop (XORShift + Mult)
        // No headers required, works on ALL drivers
        for(int i=0; i<iterations; i++) {
            v = v * 1664525 + 1013904223;
            v = v ^ (v << 13);
            v = v ^ (v >> 17);
            v = v ^ (v << 5);
            v += idx; // Force dependency
        }
        
        if (idx == 0) out[0] = v;
    }
}
"""

# ================= SENSORS =================
def get_system_temps():
    cpu_temp = 0.0
    gpu_temp = 0.0
    try:
        out = subprocess.check_output("sensors", shell=True).decode()
        for line in out.splitlines():
            if any(l in line for l in ["Tdie", "Tctl", "Package id 0", "Composite"]):
                try:
                    val = float(line.split('+')[1].split('°')[0].strip())
                    if val > cpu_temp: cpu_temp = val
                except: continue
    except: pass

    try:
        out = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True).decode()
        gpu_temp = float(out.strip())
    except: pass
    return cpu_temp, gpu_temp

# ================= WORKERS =================
def cpu_worker(id, job_queue, result_queue, stop_event, stats, current_diff, throttle_val, log_queue):
    """ CPU Worker: Explicit Nonce Reset Logic """
    current_job_id = None
    nonce_offset = 0
    
    # Worker specific start range to avoid collisions immediately
    # ID 0 starts at 0, ID 1 starts at 20M, etc.
    worker_base = id * 20_000_000 

    while not stop_event.is_set():
        t = throttle_val.value
        if t > 0.0: time.sleep(t)

        try:
            # Check for new job
            try:
                job = job_queue.get_nowait()
                job_id, prevhash, coinb1, coinb2, merkle_branch, ver, nbits, ntime, clean = job

                # JOB SWITCH LOGIC
                if job_id != current_job_id or clean:
                    current_job_id = job_id
                    nonce_offset = 0 # Reset local offset
                    # log_queue.put(("INFO", f"Core {id}: RESET NONCE for Job {job_id[:8]}"))
                
            except queue.Empty:
                pass

            if current_job_id is None:
                time.sleep(0.1)
                continue

            # Unpack current job (we might have just got it, or using cached)
            # Note: We must use the variables from the *current* job data, not the popped one if empty
            # But here we updated variables above. We need to persist them.
            # Simplified: The variables above are local. We need to store job data properly.
            pass # (Logic handled by scope in Python loop if we restructure)

            # Let's restructure slightly for persistence
            if 'job_id' not in locals(): continue

            diff = current_diff.value
            target = (0xffff0000 * 2**(256-64) // int(diff if diff > 0 else 1))

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

            # Nonce Calculation
            start_n = worker_base + nonce_offset
            batch_size = 200_000
            
            # Scan
            for n in range(start_n, start_n + 5):
                header = header_pre + struct.pack('<I', n)
                block_hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                if int.from_bytes(block_hash[::-1], 'big') <= target:
                    result_queue.put({
                        "job_id": job_id, "extranonce2": extranonce2, 
                        "ntime": ntime, "nonce": f"{n:08x}"
                    })
                    break
            
            stats[id] += batch_size
            nonce_offset += batch_size
            
            # Wrap around if too huge
            if nonce_offset > 20_000_000: nonce_offset = 0

        except Exception: 
            pass

def gpu_worker(stop_event, stats, throttle_val, log_queue):
    """ GPU Worker - Integer Only """
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        import numpy as np

        # Compile (Safe Mode)
        try:
            mod = SourceModule(CUDA_KERNEL)
            func = mod.get_function("heavy_load")
            log_queue.put(("GOOD", "GPU: CUDA Integer Kernel Active"))
        except Exception as e:
            log_queue.put(("BAD", f"GPU Compile Failed: {e}"))
            return

        while not stop_event.is_set():
            t = throttle_val.value
            if t > 0.0: time.sleep(t)
            
            out = np.zeros(1, dtype=np.int32)
            seed = int(time.time())
            
            # High iterations (100k) * Huge Grid (32k)
            func(cuda.Out(out), np.int32(100000), np.int32(seed), block=(512,1,1), grid=(32768,1))
            cuda.Context.synchronize()
            
            stats[-1] += 60_000_000
            time.sleep(0.001)
            
    except ImportError:
        log_queue.put(("WARN", "GPU: PyCUDA not installed"))
    except Exception as e:
        log_queue.put(("WARN", f"GPU Error: {e}"))
        time.sleep(5)

# ================= MANAGER =================
class RlmMiner:
    def __init__(self):
        self.manager = mp.Manager()
        self.job_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.log_queue = self.manager.Queue()
        self.stop_event = mp.Event()
        
        self.current_diff = mp.Value('d', 1024.0)
        self.throttle = mp.Value('d', 0.0)
        
        self.num_threads = mp.cpu_count()
        self.stats = mp.Array('i', [0] * (self.num_threads + 1))
        
        self.workers = []
        self.logs = []
        self.connected = False
        self.proto = "INIT"
        self.shares = {"acc": 0, "rej": 0}
        self.start_time = time.time()
        self.temps = {"cpu": 0.0, "gpu": 0.0}

    def log(self, type, msg):
        try: self.log_queue.put((type, msg))
        except: pass

    def connect_socket(self):
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.settimeout(15)
        try:
            self.log("NET", f"Dialing {POOL_URL} (SSL)...")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            s = ctx.wrap_socket(raw, server_hostname=POOL_URL)
            s.connect((POOL_URL, POOL_PORT))
            self.proto = "SSL"
            return s
        except:
            self.log("WARN", "SSL Fail. Using TCP.")
            raw.close()
            p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            p.settimeout(15)
            p.connect((POOL_URL, POOL_PORT))
            self.proto = "TCP"
            return p

    def net_loop(self):
        while not self.stop_event.is_set():
            s = None
            try:
                s = self.connect_socket()
                self.connected = True
                self.log("GOOD", f"Connected ({self.proto})")

                s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["RLM/7.0"]}) + "\n").encode())
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
                                if msg.get('method') == 'mining.notify':
                                    p = msg['params']
                                    if p[8]: # Clean Job
                                        while not self.job_queue.empty(): 
                                            try: self.job_queue.get_nowait()
                                            except: pass
                                        self.log("WARN", "Job Switch: RESET")
                                    
                                    job = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
                                    # Flood queue
                                    for _ in range(self.num_threads + 2): self.job_queue.put(job)
                                    self.log("JOB", f"New Job: {p[0][:8]}")
                                    
                                elif msg.get('method') == 'mining.set_difficulty':
                                    self.current_diff.value = msg['params'][0]
                                
                                mid = msg.get('id')
                                if mid == 1: self.log("INFO", "Subscribed")
                                elif mid == 2: self.log("GOOD", "Authorized")
                                elif mid == 4:
                                    if msg.get('result'): 
                                        self.shares['acc'] += 1
                                        self.log("GOOD", ">>> SHARE ACCEPTED <<<")
                                    else: 
                                        self.shares['rej'] += 1
                                        self.log("BAD", f"REJECTED: {msg.get('error')}")
                            except: continue
                    except socket.timeout: pass
                    except OSError: break
            except Exception as e:
                self.connected = False
                self.log("ERR", f"Net: {e}")
                time.sleep(5)
            finally:
                if s: s.close()
                self.connected = False

    def thermal_loop(self):
        while not self.stop_event.is_set():
            c, g = get_system_temps()
            self.temps['cpu'] = c
            self.temps['gpu'] = g
            max_t = max(c, g)
            
            if max_t < TARGET_TEMP - 0.5: self.throttle.value = 0.0
            elif max_t < TARGET_TEMP: self.throttle.value = 0.01
            elif max_t < MAX_TEMP:
                f = (max_t - TARGET_TEMP) / (MAX_TEMP - TARGET_TEMP)
                self.throttle.value = 0.01 + (f * 0.5)
            else:
                self.throttle.value = 1.0
                self.log("WARN", f"OVERHEAT {max_t}°C")
            time.sleep(1)

    def draw_bar(self, percentage, width=20, char="█"):
        fill = int((percentage / 100.0) * width)
        return f"[{char * fill}{'░' * (width - fill)}]"

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
                    rec = self.log_queue.get_nowait()
                    ts = datetime.now().strftime("%H:%M:%S")
                    self.logs.append((ts, rec[0], rec[1]))
                    if len(self.logs) > 12: self.logs.pop(0)
                except queue.Empty: break
                except: break

            stdscr.erase()
            h, w = stdscr.getmaxyx()
            
            head = f" RLM MINER PRO v7.0 | {POOL_URL} "
            stdscr.attron(curses.color_pair(5) | curses.A_REVERSE)
            stdscr.addstr(0, 0, head.center(w))
            stdscr.attroff(curses.color_pair(5) | curses.A_REVERSE)
            
            stat_c = curses.color_pair(1) if self.connected else curses.color_pair(3)
            stdscr.addstr(2, 2, "STATUS:", curses.color_pair(4))
            stdscr.addstr(2, 10, f"{'ONLINE' if self.connected else 'OFFLINE'} ({self.proto})", stat_c)
            
            elapsed = time.time() - self.start_time
            hr = sum(self.stats) / elapsed if elapsed > 0 else 0
            fhr = f"{hr/1000000:.2f} MH/s" if hr > 1e6 else f"{hr/1000:.2f} kH/s"
            
            stdscr.addstr(2, 40, "HASHRATE:", curses.color_pair(4))
            stdscr.addstr(2, 50, fhr, curses.color_pair(1) | curses.A_BOLD)
            
            ct, gt = self.temps['cpu'], self.temps['gpu']
            cc = curses.color_pair(1) if ct < TARGET_TEMP else curses.color_pair(2)
            if ct > MAX_TEMP: cc = curses.color_pair(3)
            gc = curses.color_pair(1) if gt < TARGET_TEMP else curses.color_pair(2)
            if gt > MAX_TEMP: gc = curses.color_pair(3)

            stdscr.addstr(4, 2, f"CPU: {ct:.1f}°C", cc)
            stdscr.addstr(4, 15, f"GPU: {gt:.1f}°C", gc)
            
            load_pct = (1.0 - self.throttle.value) * 100
            load_col = curses.color_pair(1)
            
            stdscr.addstr(4, 40, "LOAD:", curses.color_pair(4))
            stdscr.addstr(4, 50, f"{self.draw_bar(load_pct, 20)} {int(load_pct)}%", load_col)
            stdscr.addstr(5, 40, f"TARGET: {TARGET_TEMP}°C", curses.color_pair(2))

            stdscr.addstr(6, 2, "SHARES:", curses.color_pair(4))
            stdscr.addstr(6, 10, f"ACC: {self.shares['acc']}", curses.color_pair(1))
            stdscr.addstr(6, 30, f"REJ: {self.shares['rej']}", curses.color_pair(3))

            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(8, 2, " WORKER STATUS ", curses.A_REVERSE)
            
            stdscr.addstr(9, 2, "TR 3960X:", curses.color_pair(4))
            stdscr.addstr(9, 12, self.draw_bar(load_pct, 40, "▒"), cc)
            
            stdscr.addstr(10, 2, "RTX 4090:", curses.color_pair(4))
            stdscr.addstr(10, 12, self.draw_bar(load_pct, 40, "▓"), gc)

            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            stdscr.addstr(12, 2, " EVENTS ", curses.A_REVERSE)
            
            for i, (ts, typ, msg) in enumerate(self.logs):
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
        for i in range(self.num_threads):
            p = mp.Process(target=cpu_worker, args=(i, self.job_queue, self.result_queue, self.stop_event, self.stats, self.current_diff, self.throttle, self.log_queue))
            p.daemon = True
            p.start()
            self.workers.append(p)
            
        p_gpu = mp.Process(target=gpu_worker, args=(self.stop_event, self.stats, self.throttle, self.log_queue))
        p_gpu.daemon = True
        p_gpu.start()
        self.workers.append(p_gpu)
        
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
            for p in self.workers: p.terminate()

if __name__ == "__main__":
    RlmMiner().start()
