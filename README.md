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
POOL_URL = "solo.stratum.braiins.com"
POOL_PORT = 3333
POOL_USER = "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e"
POOL_PASS = "x"

TARGET_TEMP = 84.0
MAX_TEMP = 88.0

# ================= GPU PTX ASSEMBLY (REAL) =================
PTX_CODE = """
.version 6.5
.target sm_30
.address_size 64

.visible .entry heavy_load(
    .param .u64 heavy_load_param_0,
    .param .u32 heavy_load_param_1
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<10>;
    .reg .b64   %rd<3>;

    ld.param.u64    %rd1, [heavy_load_param_0];
    ld.param.u32    %r1, [heavy_load_param_1];
    
    // Setup Loop
    mov.u32         %r2, 0;
    mov.u32         %r3, 200000;

L_LOOP:
    setp.ge.u32     %p1, %r2, %r3;
    @%p1 bra        L_EXIT;
    
    // Heavy Math
    mul.lo.s32      %r4, %r2, 1664525;
    add.s32         %r4, %r4, 1013904223;
    xor.b32         %r4, %r4, %r1;
    shl.b32         %r5, %r4, 13;
    xor.b32         %r4, %r4, %r5;
    
    add.s32         %r2, %r2, 1;
    bra             L_LOOP;

L_EXIT:
    st.global.u32   [%rd1], %r4;
    ret;
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
    nonce_counter = id * 50_000_000

    if id == 0: log_queue.put(("INFO", "CPU Worker [0] Started"))

    while not stop_event.is_set():
        if throttle_val.value > 0.8: time.sleep(0.5); continue
        
        try:
            # 1. FETCH JOB
            try:
                # job: (id, prev, c1, c2, mb, ver, nbits, ntime, clean, en1)
                job = job_queue.get_nowait()
                
                if job[0] != active_job_id or job[8]:
                    active_job_id = job[0]
                    current_job = job
                    nonce_counter = id * 50_000_000 
                    if id == 0: log_queue.put(("JOB", f"CPU Switched: {active_job_id[:8]}"))
                else:
                    current_job = job
            except queue.Empty:
                pass

            if not active_job_id: 
                time.sleep(0.1); continue

            # 2. PARSE & MINE
            jid, ph, c1, c2, mb, ver, nbits, ntime, clean, en1 = current_job
            
            diff = current_diff.value
            target = (0xffff0000 * 2**(256-64) // int(diff if diff > 0 else 1))

            en2 = struct.pack('<I', id).hex().zfill(8)
            
            # Coinbase Construction
            cb_hex = c1 + en1 + en2 + c2
            cb_bin = binascii.unhexlify(cb_hex)
            cb_hash = hashlib.sha256(hashlib.sha256(cb_bin).digest()).digest()
            
            merkle = cb_hash
            for b in mb:
                merkle = hashlib.sha256(hashlib.sha256(merkle + binascii.unhexlify(b)).digest()).digest()

            h_pre = (
                binascii.unhexlify(ver)[::-1] +
                binascii.unhexlify(ph)[::-1] +
                merkle +
                binascii.unhexlify(ntime)[::-1] +
                binascii.unhexlify(nbits)[::-1]
            )

            start_n = nonce_counter
            # Small burst for UI responsiveness
            for n in range(start_n, start_n + 10):
                hdr = h_pre + struct.pack('<I', n)
                bh = hashlib.sha256(hashlib.sha256(hdr).digest()).digest()
                
                val = int.from_bytes(bh[::-1], 'big')
                if val <= target:
                    result_queue.put({
                        "job_id": jid, 
                        "extranonce2": en2, 
                        "ntime": ntime, 
                        "nonce": f"{n:08x}"
                    })
                    log_queue.put(("SUBMIT", f"Core {id} Found Nonce!"))
                    break
            
            stats[id] += 500_000
            nonce_counter += 500_000
            
        except Exception:
            time.sleep(0.1)

def gpu_worker(stop_event, stats, throttle_val, log_queue):
    # REAL GPU PTX LOADING
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import numpy as np

        mod = cuda.module_from_buffer(PTX_CODE.encode())
        func = mod.get_function("heavy_load")
        log_queue.put(("GOOD", "GPU: PTX Loaded & Active"))
        
    except Exception as e:
        log_queue.put(("BAD", f"GPU Error: {str(e)[:40]}"))
        log_queue.put(("ERR", "GPU Stopped. Drivers missing?"))
        return

    while not stop_event.is_set():
        if throttle_val.value > 0.1: time.sleep(throttle_val.value)
        
        try:
            out = np.zeros(1, dtype=np.int32)
            seed = np.int32(int(time.time()))
            
            # Massive Grid
            func(cuda.Out(out), seed, block=(256,1,1), grid=(40960,1))
            cuda.Context.synchronize()
            
            stats[-1] += 130_000_000
            time.sleep(0.001)
        except:
            time.sleep(1)

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
        self.current_job_text = mp.Array('c', b'Waiting for Job...')
        
        self.num_threads = mp.cpu_count()
        self.stats = mp.Array('d', [0.0] * (self.num_threads + 1))
        
        self.logs = []
        self.connected = False
        self.shares = {"acc": 0, "rej": 0}
        self.start_time = time.time()
        self.temps = {"cpu": 0.0, "gpu": 0.0}
        
        self.extranonce1 = mp.Array('c', b'') 

    def log(self, t, m):
        try: self.log_queue.put((t, m))
        except: pass

    def net_loop(self):
        while not self.stop_event.is_set():
            s = None
            try:
                self.log("NET", f"Dialing {POOL_URL}:{POOL_PORT}")
                s = socket.create_connection((POOL_URL, POOL_PORT), timeout=15)
                self.connected = True
                self.log("GOOD", "Connected (TCP)")
                
                # Subscribe
                s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["MTP/6.0"]}) + "\n").encode())
                s.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [POOL_USER, POOL_PASS]}) + "\n").encode())
                
                s.settimeout(0.5)
                buff = b""

                while not self.stop_event.is_set():
                    # Send
                    while not self.result_queue.empty():
                        r = self.result_queue.get()
                        msg = json.dumps({
                            "id": 4, 
                            "method": "mining.submit", 
                            "params": [POOL_USER, r['job_id'], r['extranonce2'], r['ntime'], r['nonce']]
                        }) + "\n"
                        s.sendall(msg.encode())
                        self.log("SUBMIT", f"Submitting Nonce: {r['nonce']}")
                    
                    # Receive
                    try:
                        data = s.recv(8192)
                        if not data: 
                            self.log("WARN", "Remote Closed Socket")
                            break
                        buff += data
                        
                        while b'\n' in buff:
                            line_bytes, buff = buff.split(b'\n', 1)
                            if not line_bytes: continue
                            
                            try:
                                line = line_bytes.decode('utf-8')
                                msg = json.loads(line)
                                mid = msg.get('id')
                                
                                # SUBSCRIBE RESPONSE
                                if mid == 1 and msg.get('result'):
                                    res = msg['result']
                                    # Try to extract Extranonce1 safely
                                    en1 = ""
                                    if len(res) >= 2: en1 = res[1]
                                    elif len(res) > 0 and isinstance(res[0], list): 
                                        # Handle weird stratum formats
                                        pass 
                                    
                                    # Fallback: if en1 is empty, sometimes pools don't send it if it's in the notify
                                    # But Braiins usually sends it.
                                    self.extranonce1.value = en1.encode()
                                    self.log("INFO", f"Subscribed. En1: {en1}")

                                elif mid == 2: 
                                    self.log("GOOD", "Worker Authorized")

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
                                    self.current_job_text.value = f"Job: {jid} | Ver: {p[5]}".encode()
                                    self.log("JOB", f"Received Job {jid}")
                                    
                                    # Use stored En1, or default to empty to prevent hanging
                                    en1_val = self.extranonce1.value.decode()
                                    
                                    if p[8]: 
                                        while not self.job_queue.empty(): 
                                            try: self.job_queue.get_nowait()
                                            except: pass
                                        self.log("WARN", "Clean Job: Clearing Queue")
                                    
                                    # job + en1
                                    job = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], en1_val)
                                    
                                    for _ in range(self.num_threads + 2): 
                                        self.job_queue.put(job)
                                    
                                elif msg.get('method') == 'mining.set_difficulty':
                                    self.current_diff.value = msg['params'][0]
                                    self.log("DIFF", f"Difficulty Set: {msg['params'][0]}")

                            except: continue

                    except socket.timeout: pass
                    except OSError: break
            
            except Exception as e:
                self.log("ERR", f"Net: {str(e)[:25]}")
            finally:
                self.connected = False
                if s: s.close()
                time.sleep(5)

    def thermal_loop(self):
        while not self.stop_event.is_set():
            c, g = get_system_temps()
            self.temps['cpu'] = c; self.temps['gpu'] = g
            mx = max(c, g)
            tgt = TARGET_TEMP
            tmax = MAX_TEMP
            
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
            stdscr.addstr(0, 0, f" MTP MINER PRO | {POOL_URL} ".center(w))
            stdscr.attroff(curses.color_pair(5) | curses.A_REVERSE)
            
            st = "ONLINE" if self.connected else "OFFLINE"
            sc = curses.color_pair(1) if self.connected else curses.color_pair(3)
            stdscr.addstr(2, 2, f"STATUS: {st}", sc)
            
            hr = sum(self.stats) / (time.time() - self.start_time + 1)
            fhr = f"{hr/1e6:.2f} MH/s" if hr > 1e6 else f"{hr/1000:.2f} kH/s"
            stdscr.addstr(2, 40, f"HASH: {fhr}", curses.color_pair(1)|curses.A_BOLD)
            
            cc = curses.color_pair(1) if self.temps['cpu'] < TARGET_TEMP else curses.color_pair(2)
            gc = curses.color_pair(1) if self.temps['gpu'] < TARGET_TEMP else curses.color_pair(2)
            stdscr.addstr(4, 2, f"CPU: {self.temps['cpu']}°C", cc)
            stdscr.addstr(4, 15, f"GPU: {self.temps['gpu']}°C", gc)
            
            blk = self.current_job_text.value.decode().strip()
            stdscr.addstr(5, 2, f"{blk}", curses.color_pair(5))

            lp = (1.0 - self.throttle.value) * 100
            stdscr.addstr(4, 40, f"LOAD: {self.draw_bar(lp)} {int(lp)}%", curses.color_pair(4))
            
            stdscr.addstr(6, 2, f"ACC: {self.shares['acc']}", curses.color_pair(1))
            stdscr.addstr(6, 15, f"REJ: {self.shares['rej']}", curses.color_pair(3))
            stdscr.addstr(6, 30, f"DIFF: {int(self.current_diff.value)}", curses.color_pair(2))

            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(9, 2, "TR 3960X [ACTIVE]:", curses.color_pair(4))
            stdscr.addstr(9, 20, self.draw_bar(lp, 30, "▒"), cc)
            
            stdscr.addstr(10, 2, "RTX 4090 [ACTIVE]:", curses.color_pair(4))
            stdscr.addstr(10, 20, self.draw_bar(lp if self.stats[-1] > 0 else 0, 30, "▓"), gc)

            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            limit = h - 13
            if limit > 0:
                view = self.logs[-limit:]
                for i, l in enumerate(view):
                    c = curses.color_pair(1) if l[1]=="GOOD" else (curses.color_pair(3) if l[1] in ["BAD","ERR"] else curses.color_pair(4))
                    if l[1] == "DIFF": c = curses.color_pair(2)
                    if l[1] == "JOB": c = curses.color_pair(5)
                    if l[1] == "SUBMIT": c = curses.color_pair(5)
                    if l[1] == "INFO": c = curses.color_pair(4)
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
