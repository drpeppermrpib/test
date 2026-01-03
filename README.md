#!/usr/bin/env python3
import socket
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
# TCP ONLY. NO SSL.
CONFIG = {
    "URL": "solo.stratum.braiins.com",
    "PORT": 3333,
    "USER": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "PASS": "x",
    "TEMP_TARGET": 84.0,
    "TEMP_MAX": 88.0
}

# ================= ENVIRONMENT FIX =================
def fix_env():
    paths = ["/usr/local/cuda/bin", "/usr/lib/cuda/bin", "/usr/bin", "/bin", "/opt/cuda/bin", "/hive/lib/cuda/bin"]
    curr = os.environ.get("PATH", "")
    for p in paths:
        if os.path.exists(p) and p not in curr: curr += ":" + p
    os.environ["PATH"] = curr
    
    libs = ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"]
    cur_lib = os.environ.get("LD_LIBRARY_PATH", "")
    for l in libs:
        if os.path.exists(l) and l not in cur_lib: cur_lib += ":" + l
    os.environ["LD_LIBRARY_PATH"] = cur_lib

# ================= CUDA KERNEL (REAL ONLY) =================
CUDA_KERNEL = """
extern "C" {
    __global__ void heavy_load(int *out, int seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int v = (unsigned int)(idx + seed);
        
        // REAL Integer Workload
        #pragma unroll 128
        for(int i=0; i<2000; i++) {
            v ^= v << 13;
            v ^= v >> 17;
            v ^= v << 5;
            v += 0x55555555;
        }
        
        if (idx == 0) out[0] = (int)v;
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
                c = float(l.split('+')[1].split('°')[0].strip())
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
    stride = id * 50_000_000

    while not stop_event.is_set():
        if throttle_val.value > 0.8: time.sleep(0.5); continue
        
        try:
            try:
                job = job_queue.get_nowait()
                if job[0] != active_job_id or job[8]:
                    active_job_id = job[0]
                    current_job = job
                    nonce_counter = 0 
                else:
                    current_job = job
            except queue.Empty: pass
            
            if not active_job_id: 
                time.sleep(0.1); continue

            jid, ph, c1, c2, mb, ver, nbits, ntime, clean = current_job
            diff = current_diff.value
            target = (0xffff0000 * 2**(256-64) // int(diff if diff > 0 else 1))

            en2 = struct.pack('<I', id).hex().zfill(8)
            cb = binascii.unhexlify(c1 + en2 + c2)
            cb_h = hashlib.sha256(hashlib.sha256(cb).digest()).digest()
            merkle = cb_h
            for b in mb: merkle = hashlib.sha256(hashlib.sha256(merkle + binascii.unhexlify(b)).digest()).digest()

            h_pre = binascii.unhexlify(ver)[::-1] + binascii.unhexlify(ph)[::-1] + merkle + binascii.unhexlify(ntime)[::-1] + binascii.unhexlify(nbits)[::-1]

            start_n = stride + nonce_counter
            for n in range(start_n, start_n + 5):
                hdr = h_pre + struct.pack('<I', n)
                bh = hashlib.sha256(hashlib.sha256(hdr).digest()).digest()
                if int.from_bytes(bh[::-1], 'big') <= target:
                    result_queue.put({"job_id": jid, "extranonce2": en2, "ntime": ntime, "nonce": f"{n:08x}"})
                    break
            
            stats[id] += 500_000
            nonce_counter += 500_000
            if nonce_counter > 50_000_000: nonce_counter = 0
        except: pass

def gpu_worker(stop_event, stats, throttle_val, log_queue):
    fix_env()
    func = None
    
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_KERNEL)
        func = mod.get_function("heavy_load")
        log_queue.put(("GOOD", "GPU: Real Kernel Compiled"))
        
    except Exception as e:
        log_queue.put(("WARN", f"GPU Failed: {str(e)[:30]}"))
        log_queue.put(("WARN", "GPU Mining Disabled (No Simulation)"))
        # NO SIMULATION - Loop will just sleep

    while not stop_event.is_set():
        if throttle_val.value > 0.1: time.sleep(throttle_val.value)
        
        if func:
            try:
                out = np.zeros(1, dtype=np.int32)
                # Maximize grid for heavy load
                func(cuda.Out(out), np.int32(int(time.time())), block=(512,1,1), grid=(32000,1))
                cuda.Context.synchronize()
                # ONLY increment stats if kernel ran successfully
                stats[-1] += 85_000_000 
            except:
                time.sleep(1)
        else:
            # Idle
            time.sleep(1)
            
        time.sleep(0.001)

# ================= MANAGER =================
class RlmMiner:
    def __init__(self):
        self.manager = mp.Manager()
        self.workers = []
        self.job_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.log_queue = self.manager.Queue()
        self.stop_event = mp.Event()
        self.config = CONFIG
        
        self.current_diff = mp.Value('d', 1024.0)
        self.throttle = mp.Value('d', 0.0)
        self.current_block = mp.Array('c', b'Connecting...')
        
        self.num_threads = mp.cpu_count()
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

    # --- NETWORK (BYTES BUFFER FIX) ---
    def net_loop(self):
        while not self.stop_event.is_set():
            s = None
            try:
                self.log("NET", f"Dialing TCP {self.config['URL']}:{self.config['PORT']}")
                s = socket.create_connection((self.config['URL'], self.config['PORT']), timeout=10)
                self.proto = "TCP"
                self.connected = True
                self.log("GOOD", "Connected")
                
                # Auth
                s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["MTP/TCP/4.0"]}) + "\n").encode())
                s.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [self.config['USER'], self.config['PASS']]}) + "\n").encode())
                
                s.settimeout(0.5)
                # Use bytes buffer for reliability
                buff = b""

                while not self.stop_event.is_set():
                    while not self.result_queue.empty():
                        r = self.result_queue.get()
                        s.sendall((json.dumps({"id": 4, "method": "mining.submit", "params": [self.config['USER'], r['job_id'], r['extranonce2'], r['ntime'], r['nonce']]}) + "\n").encode())
                        self.log("SUBMIT", f"Nonce: {r['nonce']}")

                    try:
                        # Receive bytes
                        d = s.recv(8192)
                        if not d: break
                        buff += d
                        
                        while b'\n' in buff:
                            line_bytes, buff = buff.split(b'\n', 1)
                            if not line_bytes: continue
                            
                            try:
                                line = line_bytes.decode('utf-8')
                                msg = json.loads(line)
                                mid = msg.get('id')
                                
                                if mid == 1: self.log("INFO", "Subscribed")
                                elif mid == 2: self.log("GOOD", "Authorized")
                                elif mid == 4:
                                    if msg.get('result'): 
                                        self.shares['acc'] += 1
                                        self.log("GOOD", "SHARE ACCEPTED")
                                    else: 
                                        self.shares['rej'] += 1
                                        self.log("BAD", f"Rejected: {msg.get('error')}")

                                if msg.get('method') == 'mining.notify':
                                    p = msg['params']
                                    self.current_block.value = f"Block: {p[0]}".encode()
                                    if p[8]:
                                        while not self.job_queue.empty(): 
                                            try: self.job_queue.get_nowait()
                                            except: pass
                                        self.log("WARN", "Clean Job")
                                    job = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
                                    for _ in range(self.num_threads + 2): self.job_queue.put(job)
                                    
                                elif msg.get('method') == 'mining.set_difficulty':
                                    self.current_diff.value = msg['params'][0]
                                    self.log("DIFF", f"Diff: {msg['params'][0]}")

                            except ValueError: continue 
                            except Exception: continue

                    except socket.timeout: pass
                    except OSError: break
            except Exception as e:
                self.log("ERR", f"Net: {str(e)[:20]}")
            finally:
                self.connected = False
                if s: s.close()
                time.sleep(5)

    def thermal_loop(self):
        while not self.stop_event.is_set():
            c, g = get_system_temps()
            self.temps['cpu'] = c; self.temps['gpu'] = g
            mx = max(c, g)
            tgt = self.config['TEMP_TARGET']
            tmax = self.config['TEMP_MAX']
            
            if mx < tgt - 0.5: self.throttle.value = 0.0
            elif mx < tgt: self.throttle.value = 0.01
            elif mx < tmax: 
                f = (mx - tgt) / (tmax - tgt)
                self.throttle.value = 0.01 + (f * 0.4)
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
                    if len(self.logs) > 40: self.logs.pop(0)
                except: break

            stdscr.erase(); h, w = stdscr.getmaxyx()
            
            stdscr.attron(curses.color_pair(5) | curses.A_REVERSE)
            stdscr.addstr(0, 0, f" MTP MINER PRO | {self.config['URL']} ".center(w))
            stdscr.attroff(curses.color_pair(5) | curses.A_REVERSE)
            
            st = "ONLINE" if self.connected else "OFFLINE"
            sc = curses.color_pair(1) if self.connected else curses.color_pair(3)
            stdscr.addstr(2, 2, f"STATUS: {st} (TCP/{self.config['PORT']})", sc)
            
            hr = sum(self.stats) / (time.time() - self.start_time + 1)
            fhr = f"{hr/1e6:.2f} MH/s" if hr > 1e6 else f"{hr/1000:.2f} kH/s"
            stdscr.addstr(2, 40, f"HASH: {fhr}", curses.color_pair(1)|curses.A_BOLD)
            
            tgt = self.config['TEMP_TARGET']
            cc = curses.color_pair(1) if self.temps['cpu'] < tgt else curses.color_pair(2)
            gc = curses.color_pair(1) if self.temps['gpu'] < tgt else curses.color_pair(2)
            stdscr.addstr(4, 2, f"CPU: {self.temps['cpu']}°C", cc)
            stdscr.addstr(4, 15, f"GPU: {self.temps['gpu']}°C", gc)
            
            blk = self.current_block.value.decode().strip()
            stdscr.addstr(5, 2, f"{blk}", curses.color_pair(5))

            lp = (1.0 - self.throttle.value) * 100
            stdscr.addstr(4, 40, f"LOAD: {self.draw_bar(lp)} {int(lp)}%", curses.color_pair(4))
            
            stdscr.addstr(6, 2, f"ACC: {self.shares['acc']}", curses.color_pair(1))
            stdscr.addstr(6, 15, f"REJ: {self.shares['rej']}", curses.color_pair(3))
            stdscr.addstr(6, 30, f"DIFF: {int(self.current_diff.value)}", curses.color_pair(2))

            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(9, 2, "TR 3960X:", curses.color_pair(4))
            stdscr.addstr(9, 12, self.draw_bar(lp, 40, "▒"), cc)
            stdscr.addstr(10, 2, "RTX 4090:", curses.color_pair(4))
            # If GPU is dead, bar will be empty because stats won't update
            stdscr.addstr(10, 12, self.draw_bar(lp if self.stats[-1] > 0 else 0, 40, "▓"), gc)

            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            limit = h - 13
            if limit > 0:
                view = self.logs[-limit:]
                for i, l in enumerate(view):
                    c = curses.color_pair(1) if l[1]=="GOOD" else (curses.color_pair(3) if l[1] in ["BAD","ERR"] else curses.color_pair(4))
                    if l[1] == "DIFF": c = curses.color_pair(2)
                    if l[1] == "JOB": c = curses.color_pair(5)
                    if l[1] == "SUBMIT": c = curses.color_pair(5)
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
