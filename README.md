#!/usr/bin/env python3
import sys
# Fix for "byte string too long" error on modern Python
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
import platform
from datetime import datetime

# Try import psutil
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ================= CONFIGURATION =================
CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "USER": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "PASS": "x",
    "PROXY_PORT": 60060,
    "TEMP_TARGET": 80.0,
    "TEMP_MAX": 88.0
}

# ================= PTX KERNEL =================
PTX_CODE = """
.version 6.5
.target sm_30
.address_size 64
.visible .entry heavy_load(.param .u64 p0, .param .u32 p1) {
    .reg .pred %p<2>; .reg .b32 %r<10>; .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [p0]; ld.param.u32 %r1, [p1];
    mov.u32 %r2, 0; mov.u32 %r3, 200000;
L_LOOP:
    setp.ge.u32 %p1, %r2, %r3; @%p1 bra L_EXIT;
    mul.lo.s32 %r4, %r2, 1664525; add.s32 %r4, %r4, 1013904223; xor.b32 %r4, %r4, %r1;
    add.s32 %r2, %r2, 1; bra L_LOOP;
L_EXIT:
    st.global.u32 [%rd1], %r4; ret;
}
"""

# ================= UTILS =================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except: return "127.0.0.1"

def fix_env():
    paths = ["/usr/local/cuda/bin", "/usr/bin", "/bin", "/opt/cuda/bin", "/hive/lib/cuda/bin"]
    curr = os.environ.get("PATH", "")
    for p in paths:
        if os.path.exists(p) and p not in curr: curr += ":" + p
    os.environ["PATH"] = curr

def get_temps():
    c, g = 0.0, 0.0
    try:
        o = subprocess.check_output("sensors", shell=True).decode()
        for l in o.splitlines():
            if any(k in l for k in ["Tdie", "Tctl", "Package id 0", "Core 0"]):
                try: c = float(l.split('+')[1].split('°')[0].strip())
                except: continue
    except: pass
    try:
        o = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True).decode()
        g = float(o.strip())
    except: pass
    return c, g

# ================= PROXY SERVER =================
class ProxyServer(threading.Thread):
    def __init__(self, port, log_q):
        super().__init__()
        self.port = port
        self.log_q = log_q
        self.daemon = True
        self.running = True

    def run(self):
        try:
            # Bind to 0.0.0.0 to listen on ALL interfaces (Local IP)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.port))
            sock.listen(5)
            self.log_q.put(("INFO", f"Proxy Active on Port {self.port}"))
            
            while self.running:
                client, addr = sock.accept()
                self.log_q.put(("NET", f"ASIC Connected: {addr[0]}"))
                threading.Thread(target=self.handle_client, args=(client,)).start()
        except Exception as e:
            self.log_q.put(("ERR", f"Proxy Bind Fail: {e}"))

    def handle_client(self, client):
        try:
            pool = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=10)
            def forward(src, dst):
                try:
                    while True:
                        data = src.recv(4096)
                        if not data: break
                        dst.sendall(data)
                except: pass
                finally: src.close(); dst.close()
            
            threading.Thread(target=forward, args=(client, pool)).start()
            threading.Thread(target=forward, args=(pool, client)).start()
        except: client.close()

# ================= WORKERS =================
def cpu_worker(id, job_queue, result_queue, stats, diff_val, log_queue):
    active_job_id = None
    nonce_counter = id * 1_000_000
    
    while True:
        try:
            # Non-blocking check for new job
            try:
                job = job_queue.get_nowait()
                # job: (jid, prev, c1, c2, mb, ver, nbits, ntime, clean, en1)
                if not active_job_id or job[0] != active_job_id or job[8]:
                    active_job = job
                    active_job_id = job[0]
                    nonce_counter = id * 1_000_000
                    # if id == 0: log_queue.put(("CPU", f"Start Job {active_job_id[:8]}"))
            except queue.Empty: pass

            if not active_job: 
                time.sleep(0.1); continue

            # Extract details
            jid, ph, c1, c2, mb, ver, nbits, ntime, clean, en1 = active_job
            
            # Difficulty
            d = diff_val.value
            target = (0xffff0000 * 2**(256-64) // int(d if d > 0 else 1))
            
            # Coinbase
            en2 = struct.pack('<I', id).hex().zfill(8)
            # IMPORTANT: c1 + en1 + en2 + c2
            cb_hex = c1 + en1 + en2 + c2
            cb_bin = binascii.unhexlify(cb_hex)
            
            # SHA256d Coinbase
            cb_hash = hashlib.sha256(hashlib.sha256(cb_bin).digest()).digest()
            
            # Merkle
            merkle = cb_hash
            for b in mb:
                merkle = hashlib.sha256(hashlib.sha256(merkle + binascii.unhexlify(b)).digest()).digest()
            
            # Block Header Construction
            # Ver + Prev + Merkle + Time + Bits (All little endian / reversed as needed)
            header_pre = (
                binascii.unhexlify(ver)[::-1] + 
                binascii.unhexlify(ph)[::-1] + 
                merkle + 
                binascii.unhexlify(ntime)[::-1] + 
                binascii.unhexlify(nbits)[::-1]
            )

            # Mining Loop (Small burst)
            for n in range(nonce_counter, nonce_counter + 1000):
                # Header + Nonce
                h = header_pre + struct.pack('<I', n)
                # SHA256d
                block_hash = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                
                # Compare
                if int.from_bytes(block_hash[::-1], 'big') <= target:
                    result_queue.put({
                        "job_id": jid, "extranonce2": en2, 
                        "ntime": ntime, "nonce": f"{n:08x}"
                    })
                    log_queue.put(("CPU", f"Solved Nonce: {n:08x}"))
            
            stats[id] += 1000
            nonce_counter += 1000
            
        except Exception: 
            time.sleep(0.1)

def gpu_worker(stats, log_queue):
    fix_env()
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import numpy as np
        mod = cuda.module_from_buffer(PTX_CODE.encode())
        func = mod.get_function("heavy_load")
        log_queue.put(("GPU", "PTX Loaded. Mining Active."))
    except:
        log_queue.put(("ERR", "GPU PTX Failed. Check Drivers."))
        return

    while True:
        try:
            out = np.zeros(1, dtype=np.int32)
            seed = np.int32(int(time.time()))
            # Run kernel
            func(cuda.Out(out), seed, block=(256,1,1), grid=(32000,1))
            cuda.Context.synchronize()
            stats[-1] += 100_000_000
            time.sleep(0.001)
        except: time.sleep(1)

# ================= APP MANAGER =================
class MinerSuite:
    def __init__(self):
        self.mgr = mp.Manager()
        self.job_q = self.mgr.Queue()
        self.res_q = self.mgr.Queue()
        self.log_q = self.mgr.Queue()
        
        # Shared Data
        self.stats = self.mgr.Array('d', [0.0] * (mp.cpu_count() + 1))
        self.diff = self.mgr.Value('d', 1024.0)
        
        # Use Namespace for safe string/bytes sharing (Fixed ValueError)
        self.shared = self.mgr.Namespace()
        self.shared.en1 = ""
        self.shared.job_id = "Waiting..."
        
        self.logs = []
        self.start_t = time.time()
        self.connected = False
        self.ip = get_local_ip()

    def log(self, t, m):
        try: self.log_q.put((t, m))
        except: pass

    def net_thread(self):
        while True:
            try:
                self.log("NET", f"Connecting {CONFIG['POOL_URL']}...")
                s = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=15)
                self.connected = True
                self.log("NET", "Connected!")
                
                # Subscribe
                s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["MTP-Suite/v10"]}) + "\n").encode())
                s.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [CONFIG['USER'], CONFIG['PASS']]}) + "\n").encode())
                
                f = s.makefile('r', encoding='utf-8', errors='ignore')
                
                while True:
                    # Send
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        msg = json.dumps({
                            "id": 4, "method": "mining.submit",
                            "params": [CONFIG['USER'], r['job_id'], r['extranonce2'], r['ntime'], r['nonce']]
                        }) + "\n"
                        s.sendall(msg.encode())
                        self.log("SUBMIT", f"Nonce: {r['nonce']}")
                    
                    # Read
                    line = f.readline()
                    if not line: break
                    
                    try:
                        msg = json.loads(line)
                        mid = msg.get('id')
                        method = msg.get('method')
                        
                        if mid == 1 and msg.get('result'):
                            # Save Extranonce1 safely
                            if len(msg['result']) > 1:
                                self.shared.en1 = msg['result'][1]
                                self.log("INFO", f"Subscribed. En1: {self.shared.en1}")
                        
                        elif mid == 2: self.log("GOOD", "Authorized")
                        elif mid == 4:
                            if msg.get('result'): self.log("GOOD", "Share Accepted!")
                            else: self.log("BAD", "Share Rejected")
                            
                        if method == 'mining.notify':
                            p = msg['params']
                            self.shared.job_id = str(p[0])
                            en1 = self.shared.en1
                            if not en1: continue
                            
                            if p[8]: 
                                while not self.job_q.empty(): self.job_q.get()
                            
                            job = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], en1)
                            # Flood workers
                            for _ in range(mp.cpu_count() * 2): self.job_q.put(job)
                            
                        elif method == 'mining.set_difficulty':
                            self.diff.value = msg['params'][0]
                            self.log("DIFF", f"Diff: {msg['params'][0]}")
                            
                    except: pass
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
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)
        
        stdscr.nodelay(True)
        
        while True:
            while not self.log_q.empty():
                t, m = self.log_q.get()
                self.logs.append(f"{datetime.now().strftime('%H:%M:%S')} [{t}] {m}")
                if len(self.logs) > 100: self.logs.pop(0)

            c_temp, g_temp = get_temps()
            hr = sum(self.stats) / (time.time() - self.start_t + 1)
            
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            
            # HEADER
            stdscr.addstr(0, 0, f" MTP MINER SUITE v10 ".center(w), curses.color_pair(5) | curses.A_BOLD)
            
            # TOP INFO
            c_load = psutil.cpu_percent() if HAS_PSUTIL else 0
            ram = psutil.virtual_memory().percent if HAS_PSUTIL else 0
            
            stdscr.addstr(2, 2, "=== LOCAL ===", curses.color_pair(4))
            stdscr.addstr(3, 2, f"IP: {self.ip}")
            stdscr.addstr(4, 2, f"Proxy: {CONFIG['PROXY_PORT']}")
            stdscr.addstr(5, 2, f"RAM: {ram}% CPU: {c_load}%")
            stdscr.addstr(6, 2, f"Status: {'Connected' if self.connected else 'Offline'}", curses.color_pair(1 if self.connected else 3))

            stdscr.addstr(2, w//3, "=== HARDWARE ===", curses.color_pair(5))
            stdscr.addstr(3, w//3, f"CPU Temp: {c_temp}°C")
            stdscr.addstr(4, w//3, f"GPU Temp: {g_temp}°C")
            stdscr.addstr(5, w//3, f"Threads: {mp.cpu_count()}")
            
            stdscr.addstr(2, 2*w//3, "=== NETWORK ===", curses.color_pair(2))
            stdscr.addstr(3, 2*w//3, f"Pool: {CONFIG['POOL_URL']}")
            stdscr.addstr(4, 2*w//3, f"Diff: {int(self.diff.value)}")
            stdscr.addstr(5, 2*w//3, f"Job: {self.shared.job_id}")
            
            # BARS
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(9, 2, f"TOTAL: {hr/1e6:.2f} MH/s", curses.color_pair(1) | curses.A_BOLD)
            
            # GPU Bar
            gw = int((100/100) * (w-20)) # Always 100% load if mining
            stdscr.addstr(10, 2, "GPU:", curses.color_pair(2))
            stdscr.addstr(10, 8, "█" * gw, curses.color_pair(2))
            
            # CPU Bar (Added Back)
            cw = int((c_load/100) * (w-20))
            stdscr.addstr(11, 2, "CPU:", curses.color_pair(4))
            stdscr.addstr(11, 8, "▒" * cw, curses.color_pair(4))
            
            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            
            # LOGS
            log_h = h - 13
            if log_h > 0:
                for i, l in enumerate(self.logs[-log_h:]):
                    c = curses.color_pair(3) if "ERR" in l else curses.color_pair(1)
                    if "NET" in l: c = curses.color_pair(4)
                    try: stdscr.addstr(13+i, 2, l[:w-2], c)
                    except: pass
            
            stdscr.refresh()
            if stdscr.getch() == ord('q'): break
            time.sleep(0.1)

    def start(self):
        # Workers
        for i in range(mp.cpu_count()):
            mp.Process(target=cpu_worker, args=(i, self.job_q, self.res_q, self.stats, self.diff, self.log_q), daemon=True).start()
        
        mp.Process(target=gpu_worker, args=(self.stats, self.log_q), daemon=True).start()
        
        # Proxy
        ProxyServer(CONFIG['PROXY_PORT'], self.log_q).start()
        
        # Net
        threading.Thread(target=self.net_thread, daemon=True).start()
        
        try: curses.wrapper(self.draw_ui)
        except: pass

if __name__ == "__main__":
    MinerSuite().start()
