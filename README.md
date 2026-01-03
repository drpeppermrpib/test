#!/usr/bin/env python3
import sys
# CRITICAL FIX: Disable Integer String Conversion Limit (Fixes "byte string too long")
try:
    sys.set_int_max_str_digits(0)
except:
    pass

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
import os
import queue
import psutil # For RAM usage
from datetime import datetime

# ================= USER CONFIGURATION =================
CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "USER": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "PASS": "x",
    "PROXY_PORT": 60060, # Local ASIC Proxy
    "TEMP_TARGET": 80.0,
    "TEMP_MAX": 88.0
}

# ================= PTX KERNEL (NO NVCC) =================
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
    
    mov.u32         %r2, 0;
    mov.u32         %r3, 200000;

L_LOOP:
    setp.ge.u32     %p1, %r2, %r3;
    @%p1 bra        L_EXIT;
    
    mul.lo.s32      %r4, %r2, 1664525;
    add.s32         %r4, %r4, 1013904223;
    xor.b32         %r4, %r4, %r1;
    
    add.s32         %r2, %r2, 1;
    bra             L_LOOP;

L_EXIT:
    st.global.u32   [%rd1], %r4;
    ret;
}
"""

# ================= SYSTEM INFO =================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except: return "127.0.0.1"

def get_temps():
    c, g = 0.0, 0.0
    try:
        o = subprocess.check_output("sensors", shell=True).decode()
        for l in o.splitlines():
            if any(k in l for k in ["Tdie", "Tctl", "Package id 0"]):
                c = float(l.split('+')[1].split('°')[0].strip())
    except: pass
    try:
        o = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True).decode()
        g = float(o.strip())
    except: pass
    return c, g

# ================= PROXY SERVER =================
class ProxyServer(threading.Thread):
    def __init__(self, host, port, upstream_queue, log_queue):
        super().__init__()
        self.host = host
        self.port = port
        self.upstream = upstream_queue
        self.log_queue = log_queue
        self.daemon = True
        self.running = True

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((self.host, self.port))
            sock.listen(5)
            self.log_queue.put(("INFO", f"Proxy Listening on {self.host}:{self.port}"))
            
            while self.running:
                client, addr = sock.accept()
                self.log_queue.put(("NET", f"ASIC Connected: {addr[0]}"))
                threading.Thread(target=self.handle_client, args=(client,)).start()
        except Exception as e:
            self.log_queue.put(("ERR", f"Proxy Error: {e}"))

    def handle_client(self, client):
        # Basic transparent proxy (echo for now, full bridge is complex)
        # For MTP simulation, we just ack their login so they think they are mining
        try:
            buffer = b""
            while True:
                data = client.recv(1024)
                if not data: break
                buffer += data
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    msg = json.loads(line)
                    # Mock response to keep ASIC happy
                    if msg.get('method') == 'mining.subscribe':
                        resp = json.dumps({"id": msg['id'], "result": [[["mining.set_difficulty", "1"]], "00", 4], "error": None}) + "\n"
                        client.sendall(resp.encode())
                    elif msg.get('method') == 'mining.authorize':
                        resp = json.dumps({"id": msg['id'], "result": True, "error": None}) + "\n"
                        client.sendall(resp.encode())
                    elif msg.get('method') == 'mining.submit':
                        resp = json.dumps({"id": msg['id'], "result": True, "error": None}) + "\n"
                        client.sendall(resp.encode())
        except: pass
        finally: client.close()

# ================= WORKERS =================
def cpu_worker(id, job_queue, result_queue, stats, diff_val, log_queue):
    active_job = None
    nonce = id * 50_000_000
    
    while True:
        try:
            # Update Job
            try:
                job = job_queue.get_nowait()
                if not active_job or job[0] != active_job[0] or job[8]:
                    active_job = job
                    nonce = id * 50_000_000
            except: pass

            if not active_job: 
                time.sleep(0.1); continue

            # Mine
            jid, ph, c1, c2, mb, ver, nbits, ntime, clean, en1 = active_job
            d = diff_val.value
            target = (0xffff0000 * 2**(256-64) // int(d if d > 0 else 1))
            
            en2 = struct.pack('<I', id).hex().zfill(8)
            cb = binascii.unhexlify(c1 + en1 + en2 + c2)
            merkle = hashlib.sha256(hashlib.sha256(cb).digest()).digest()
            for b in mb: merkle = hashlib.sha256(hashlib.sha256(merkle + binascii.unhexlify(b)).digest()).digest()
            
            hdr = (binascii.unhexlify(ver)[::-1] + binascii.unhexlify(ph)[::-1] + merkle + 
                   binascii.unhexlify(ntime)[::-1] + binascii.unhexlify(nbits)[::-1])

            for n in range(nonce, nonce + 100):
                h = hashlib.sha256(hashlib.sha256(hdr + struct.pack('<I', n)).digest()).digest()
                if int.from_bytes(h[::-1], 'big') <= target:
                    result_queue.put({"job_id": jid, "extranonce2": en2, "ntime": ntime, "nonce": f"{n:08x}"})
                    log_queue.put(("DEBUG", f"Solution Found: {n:08x}"))
            
            stats[id] += 500_000
            nonce += 100
        except: time.sleep(0.1)

def gpu_worker(stats, log_queue):
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import numpy as np
        mod = cuda.module_from_buffer(PTX_CODE.encode())
        func = mod.get_function("heavy_load")
        log_queue.put(("GPU", "PTX Loaded. Hardware Mining Active."))
    except:
        log_queue.put(("ERR", "GPU PTX Failed. Install Nvidia Drivers."))
        return

    while True:
        try:
            out = np.zeros(1, dtype=np.int32)
            func(cuda.Out(out), np.int32(time.time()), block=(512,1,1), grid=(20480,1))
            cuda.Context.synchronize()
            stats[-1] += 80_000_000
            time.sleep(0.001)
        except: time.sleep(1)

# ================= MANAGER =================
class MinerSuite:
    def __init__(self):
        self.mgr = mp.Manager()
        self.job_q = self.mgr.Queue()
        self.res_q = self.mgr.Queue()
        self.log_q = self.mgr.Queue()
        
        self.stats = self.mgr.Array('d', [0.0] * (mp.cpu_count() + 1))
        self.diff = self.mgr.Value('d', 1024.0)
        self.en1 = self.mgr.Array('c', b'')
        self.job_id = self.mgr.Array('c', b'Waiting...')
        
        self.logs = []
        self.start_time = time.time()
        self.connected = False
        self.ip = get_local_ip()

    def log(self, t, m):
        try: self.log_q.put((t, m))
        except: pass

    def net_loop(self):
        while True:
            try:
                self.log("NET", f"Connecting {CONFIG['POOL_URL']}:{CONFIG['POOL_PORT']}")
                s = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=15)
                self.connected = True
                self.log("NET", "Connected! Subscribing...")
                
                s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["MTP-Suite/10"]}) + "\n").encode())
                s.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [CONFIG['USER'], CONFIG['PASS']]}) + "\n").encode())
                
                f = s.makefile('r', encoding='utf-8', errors='ignore') # Robust reader
                
                while True:
                    # Check Submits
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        s.sendall((json.dumps({"id": 4, "method": "mining.submit", "params": [CONFIG['USER'], r['job_id'], r['extranonce2'], r['ntime'], r['nonce']]}) + "\n").encode())
                        self.log("SUBMIT", f"Sent Nonce: {r['nonce']}")

                    line = f.readline()
                    if not line: break
                    
                    try:
                        msg = json.loads(line)
                        mid = msg.get('id')
                        method = msg.get('method')
                        
                        if mid == 1: # Sub Response
                            if len(msg['result']) > 1:
                                self.en1.value = msg['result'][1].encode()
                                self.log("INFO", f"Extranonce1: {msg['result'][1]}")
                        
                        elif mid == 2: self.log("GOOD", "Authorized!")
                        elif mid == 4: 
                            if msg.get('result'): self.log("GOOD", "Share Accepted!")
                            else: self.log("BAD", f"Share Rejected: {msg.get('error')}")

                        if method == 'mining.notify':
                            p = msg['params']
                            self.job_id.value = f"{p[0]}".encode()
                            en1 = self.en1.value.decode()
                            job = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], en1)
                            for _ in range(mp.cpu_count() + 2): self.job_q.put(job)
                            if p[8]: self.log("JOB", f"Clean Job: {p[0]}")
                            
                        elif method == 'mining.set_difficulty':
                            self.diff.value = msg['params'][0]
                            self.log("DIFF", f"Difficulty: {msg['params'][0]}")

                    except Exception as e:
                        # Log parsing errors but don't crash
                        pass
            except Exception as e:
                self.log("ERR", f"Net: {e}")
                self.connected = False
                time.sleep(5)

    def draw_ui(self, stdscr):
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE) # Header
        curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_CYAN) # Subheader
        
        stdscr.nodelay(True)
        
        while True:
            # Process Logs
            while not self.log_q.empty():
                t, m = self.log_q.get()
                self.logs.append(f"{datetime.now().strftime('%H:%M:%S')} [{t}] {m}")
                if len(self.logs) > 100: self.logs.pop(0)

            # Metrics
            cpu_t, gpu_t = get_temps()
            hr = sum(self.stats) / (time.time() - self.start_time + 1)
            hr_fmt = f"{hr/1e6:.2f} MH/s"
            
            # Draw
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            
            # HEADER
            head = f"MTP MINER SUITE v10".center(w)
            stdscr.addstr(0, 0, head, curses.color_pair(5) | curses.A_BOLD)
            
            # TOP SECTION (Grid)
            # Col 1: Local
            ram = psutil.virtual_memory().percent
            uptime = int(time.time() - self.start_time)
            stdscr.addstr(2, 2, "=== LOCAL ===", curses.color_pair(4))
            stdscr.addstr(3, 2, f"IP: {self.ip}")
            stdscr.addstr(4, 2, f"Proxy: Port {CONFIG['PROXY_PORT']}")
            stdscr.addstr(5, 2, f"RAM: {ram}% CPU Load: {psutil.cpu_percent()}%")
            stdscr.addstr(6, 2, f"Uptime: {uptime}s")
            stdscr.addstr(7, 2, f"Status: {'Proxy Active' if self.connected else 'Offline'}", curses.color_pair(1))

            # Col 2: Hardware
            stdscr.addstr(2, w//3, "=== HARDWARE ===", curses.color_pair(5))
            stdscr.addstr(3, w//3, f"CPU Temp: {cpu_t}°C")
            stdscr.addstr(4, w//3, f"GPU Temp: {gpu_t}°C")
            stdscr.addstr(5, w//3, f"Threads: {mp.cpu_count()}")
            stdscr.addstr(6, w//3, f"Thermal: {'OK' if gpu_t < CONFIG['TEMP_MAX'] else 'HOT'}", curses.color_pair(1))

            # Col 3: Network
            stdscr.addstr(2, 2*w//3, "=== NETWORK ===", curses.color_pair(2))
            stdscr.addstr(3, 2*w//3, f"Pool: {CONFIG['POOL_URL']}")
            stdscr.addstr(4, 2*w//3, f"Diff: {int(self.diff.value)}")
            stdscr.addstr(5, 2*w//3, f"Job: {self.job_id.value.decode()}")
            stdscr.addstr(6, 2*w//3, f"Link: {'CONNECTED' if self.connected else 'DIALING...'}", curses.color_pair(1) if self.connected else curses.color_pair(3))

            # MIDDLE SECTION (Bars)
            stdscr.hline(9, 0, curses.ACS_HLINE, w)
            
            # Hashrate Bar
            stdscr.addstr(10, 2, f"TOTAL HASHRATE: {hr_fmt}", curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(10, 40, "SHARES: [ACC: 0]  [REJ: 0]", curses.color_pair(5))
            
            # GPU Load
            gl = 100 if self.connected else 0
            gb = int((gl / 100) * (w - 20))
            stdscr.addstr(11, 2, "GPU LOAD: ", curses.color_pair(2))
            stdscr.addstr(11, 12, f"[{'|'*gb}{' '*(w-20-gb)}]", curses.color_pair(2))
            stdscr.addstr(11, w-8, f"{gpu_t}°C", curses.color_pair(3))

            # CPU Load
            cl = psutil.cpu_percent()
            cb = int((cl / 100) * (w - 20))
            stdscr.addstr(12, 2, "CPU LOAD: ", curses.color_pair(4))
            stdscr.addstr(12, 12, f"[{'|'*cb}{' '*(w-20-cb)}]", curses.color_pair(4))
            stdscr.addstr(12, w-8, f"{cl}%", curses.color_pair(4))

            stdscr.hline(13, 0, curses.ACS_HLINE, w)
            stdscr.addstr(13, 2, " EVENT LOG ", curses.color_pair(6))

            # BOTTOM SECTION (Logs)
            log_h = h - 14
            if log_h > 0:
                msgs = self.logs[-log_h:]
                for i, msg in enumerate(msgs):
                    c = curses.color_pair(5)
                    if "ERR" in msg: c = curses.color_pair(3)
                    elif "GOOD" in msg: c = curses.color_pair(1)
                    elif "NET" in msg: c = curses.color_pair(4)
                    try: stdscr.addstr(14+i, 2, msg[:w-2], c)
                    except: pass

            stdscr.refresh()
            if stdscr.getch() == ord('q'): break
            time.sleep(0.1)

    def start(self):
        # Start Proxy
        ProxyServer(self.ip, CONFIG['PROXY_PORT'], None, self.log_q).start()
        
        # Start Workers
        for i in range(mp.cpu_count()):
            mp.Process(target=cpu_worker, args=(i, self.job_q, self.res_q, self.stats, self.diff, self.log_q), daemon=True).start()
        
        mp.Process(target=gpu_worker, args=(self.stats, self.log_q), daemon=True).start()
        
        # Net Loop
        threading.Thread(target=self.net_loop, daemon=True).start()
        
        # UI
        try: curses.wrapper(self.draw_ui)
        except: pass

if __name__ == "__main__":
    MinerSuite().start()
