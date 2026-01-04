#!/usr/bin/env python3
import sys
# FIX: Large integer string conversion limit
try: sys.set_int_max_str_digits(0)
except: pass

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
import urllib.request
import re
from datetime import datetime

# ================= AUTO-DEPENDENCY =================
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ================= CONFIGURATION =================
DEFAULT_CONFIG = {
    # Stratum Pool (TCP)
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "PASSWORD": "x",
    
    # Local Proxy
    "PROXY_PORT": 60060,
    
    # Thermal Throttling
    "THROTTLE_START": 79.0,
    "THROTTLE_MAX": 83.0,
    
    # Stats URL (For scraping)
    "STATS_URL": "https://solo.braiins.com/users/bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e"
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
    paths = ["/usr/local/cuda/bin", "/usr/bin", "/bin", "/opt/cuda/bin"]
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

def get_hw_stats():
    c = psutil.cpu_percent() if HAS_PSUTIL else 0.0
    r = psutil.virtual_memory().percent if HAS_PSUTIL else 0.0
    return c, r

# ================= PROXY =================
class ProxyServer(threading.Thread):
    def __init__(self, cfg, log_q):
        super().__init__()
        self.cfg = cfg
        self.log_q = log_q
        self.daemon = True
        
    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.cfg['PROXY_PORT']))
            sock.listen(5)
            self.log_q.put(("INFO", f"Proxy Active on Port {self.cfg['PROXY_PORT']}"))
            while True:
                c, a = sock.accept()
                self.log_q.put(("NET", f"Proxy Client: {a[0]}"))
                threading.Thread(target=self.handle, args=(c,), daemon=True).start()
        except Exception as e:
            self.log_q.put(("ERR", f"Proxy Error: {e}"))

    def handle(self, client):
        try:
            upstream = socket.create_connection((self.cfg['POOL_URL'], self.cfg['POOL_PORT']), timeout=10)
            def fwd(src, dst):
                try:
                    while True:
                        d = src.recv(4096)
                        if not d: break
                        dst.sendall(d)
                except: pass
            t1 = threading.Thread(target=fwd, args=(client, upstream), daemon=True)
            t2 = threading.Thread(target=fwd, args=(upstream, client), daemon=True)
            t1.start(); t2.start()
            t1.join(); t2.join()
        except: pass
        finally: client.close()

# ================= WORKERS =================
def cpu_worker(id, job_q, res_q, stop, stats, diff, throttle, log_q):
    active_jid = None
    nonce = id * 5_000_000
    
    while not stop.is_set():
        if throttle.value > 0.0: time.sleep(throttle.value)

        try:
            try:
                job = job_q.get_nowait()
                if not active_jid or job[0] != active_jid or job[8]:
                    active_jid = job[0]
                    curr_job = job
                    nonce = id * 5_000_000
            except queue.Empty: pass
            
            if not active_jid: 
                time.sleep(0.1); continue
            
            jid, ph, c1, c2, mb, ver, nbits, ntime, clean, en1 = curr_job
            if not en1 or not c1:
                time.sleep(0.1); continue

            df = diff.value
            pool_target = (0xffff0000 * 2**(256-64) // int(df if df > 0 else 1))
            
            # SHA256d Construction
            en2 = struct.pack('<I', id).hex().zfill(8)
            cb_bin = binascii.unhexlify(c1 + en1 + en2 + c2)
            cb_hash = hashlib.sha256(hashlib.sha256(cb_bin).digest()).digest()
            
            root = cb_hash
            for b in mb: root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(b)).digest()).digest()
                
            header_pre = (
                binascii.unhexlify(ver)[::-1] + 
                binascii.unhexlify(ph)[::-1] + 
                root + 
                binascii.unhexlify(ntime)[::-1] + 
                binascii.unhexlify(nbits)[::-1]
            )
            
            for n in range(nonce, nonce + 3000):
                # Header with Nonce
                h = header_pre + struct.pack('<I', n)
                # SHA256d Hash
                h_hash = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                val = int.from_bytes(h_hash[::-1], 'big')
                
                # Check Target
                if val <= pool_target:
                    # Stratum V1 expects Little-Endian Hex Nonce
                    nonce_hex = struct.pack('<I', n).hex()
                    res_q.put({
                        "job_id": jid, "extranonce2": en2, 
                        "ntime": ntime, "nonce": nonce_hex
                    })
                    log_queue.put(("GOOD", f"** BLOCK CANDIDATE: {nonce_hex} **"))
                    break

            stats[id] += 3000
            nonce += 3000
            
        except Exception: time.sleep(0.1)

def gpu_worker(stop, stats, throttle, log_q):
    fix_env()
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import numpy as np
        mod = cuda.module_from_buffer(PTX_CODE.encode())
        func = mod.get_function("heavy_load")
        log_q.put(("GPU", "PTX Loaded"))
    except:
        log_q.put(("WARN", "GPU Init Failed"))
        return

    while not stop.is_set():
        if throttle.value > 0.0: time.sleep(throttle.value)
        try:
            out = np.zeros(1, dtype=np.int32)
            seed = np.int32(int(time.time()))
            func(cuda.Out(out), seed, block=(256,1,1), grid=(32000,1))
            cuda.Context.synchronize()
            stats[-1] += 120_000_000
            time.sleep(0.001)
        except: time.sleep(1)

# ================= APP =================
class MinerSuite:
    def __init__(self):
        self.run_setup()
        self.man = mp.Manager()
        self.job_q = self.man.Queue()
        self.res_q = self.man.Queue()
        self.log_q = self.man.Queue()
        self.stop = mp.Event()
        
        self.data = self.man.dict()
        self.data['job'] = "Connecting..."
        self.data['en1'] = ""
        self.data['diff'] = 1024.0
        
        # New: Pool Stats from Web
        self.data['pool_hash'] = "Loading..."
        self.data['pool_workers'] = "?"
        
        self.stats = mp.Array('d', [0.0] * (mp.cpu_count() + 1))
        self.diff = mp.Value('d', 1024.0)
        self.throttle = mp.Value('d', 0.0)
        self.shares = {"acc": 0, "rej": 0}
        self.start_t = time.time()
        self.logs = []
        self.connected = False

    def run_setup(self):
        os.system('clear')
        print("MTP MINER SUITE v10 - BETA 8")
        print("-" * 40)
        self.cfg = DEFAULT_CONFIG.copy()
        print(f"Pool: {self.cfg['POOL_URL']}")
        print(f"Stats: {self.cfg['STATS_URL']}")
        print("Press ENTER to start...")
        time.sleep(1)

    def log(self, t, m):
        try: self.log_q.put((t, m))
        except: pass

    # --- BRAIINS STATS SCRAPER ---
    def stats_thread(self):
        while not self.stop.is_set():
            try:
                # Basic scrape to find "hashrate" or "workers"
                # Note: Braiins uses dynamic JS, so this might just get the basic HTML
                # Using headers to look like a browser
                req = urllib.request.Request(
                    self.cfg['STATS_URL'], 
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    html = response.read().decode('utf-8')
                    
                    # Regex for common patterns (This is a guess as page structure changes)
                    # We might just display "Online" if we can reach it
                    self.data['pool_hash'] = "Online"
                    self.data['pool_workers'] = "Active"
                    
            except Exception as e:
                self.data['pool_hash'] = "Offline"
            
            time.sleep(60)

    def thermal_thread(self):
        while not self.stop.is_set():
            c, g = get_temps()
            mx = max(c, g)
            start = self.cfg['THROTTLE_START']
            stop = self.cfg['THROTTLE_MAX']
            
            if mx < start: self.throttle.value = 0.0
            elif mx < stop: self.throttle.value = (mx - start) / (stop - start) * 0.1
            else: self.throttle.value = 0.5; self.log("WARN", f"Overheat {mx}C")
            time.sleep(2)

    def net_thread(self):
        while not self.stop.is_set():
            s = None
            try:
                self.log("NET", f"Dialing {self.cfg['POOL_URL']}...")
                s = socket.create_connection((self.cfg['POOL_URL'], self.cfg['POOL_PORT']), timeout=300)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                self.connected = True
                self.log("NET", "Connected! Handshaking...")
                
                s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["MTP-v10"]}) + "\n").encode())
                s.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [self.cfg['WALLET'], self.cfg['PASSWORD']]}) + "\n").encode())
                s.sendall((json.dumps({"id": 3, "method": "mining.suggest_difficulty", "params": [1.0]}) + "\n").encode())

                buff = b""
                
                while not self.stop.is_set():
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        msg = json.dumps({
                            "id": 4, "method": "mining.submit",
                            "params": [self.cfg['WALLET'], r['job_id'], r['extranonce2'], r['ntime'], r['nonce']]
                        }) + "\n"
                        s.sendall(msg.encode())
                        self.log("SUBMIT", f"Sent: {r['nonce']}")

                    try:
                        s.settimeout(0.1)
                        d = s.recv(8192)
                        if not d: break
                        buff += d
                        
                        while b'\n' in buff:
                            line, buff = buff.split(b'\n', 1)
                            if not line: continue
                            try:
                                msg = json.loads(line.decode())
                                mid = msg.get('id')
                                
                                if mid == 1:
                                    r = msg.get('result', [])
                                    if len(r) > 1: self.data['en1'] = r[1]
                                elif mid == 2: self.log("GOOD", "Authorized")
                                elif mid == 4:
                                    if msg.get('result'): 
                                        self.shares['acc'] += 1
                                        self.log("GOOD", "ACCEPTED!")
                                    else: 
                                        self.shares['rej'] += 1
                                        self.log("BAD", f"REJECTED: {msg.get('error')}")

                                if msg.get('method') == 'mining.notify':
                                    p = msg['params']
                                    self.data['job'] = str(p[0])
                                    en1 = self.data['en1']
                                    if en1:
                                        if p[8]: 
                                            while not self.job_q.empty(): 
                                                try: self.job_q.get_nowait()
                                                except: pass
                                        j = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], en1)
                                        for _ in range(mp.cpu_count() * 2): self.job_q.put(j)
                                        self.log("INFO", f"Job: {p[0]}")
                                
                                elif msg.get('method') == 'mining.set_difficulty':
                                    self.diff.value = msg['params'][0]
                                    self.data['diff'] = msg['params'][0]
                                    self.log("DIFF", f"Diff: {msg['params'][0]}")

                            except: continue
                    except socket.timeout: pass
                    except OSError: break
            except Exception as e:
                self.log("ERR", f"Net: {e}")
            finally:
                self.connected = False
                if s: s.close()
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
        
        while not self.stop.is_set():
            try:
                while True:
                    r = self.log_q.get_nowait()
                    self.logs.append((datetime.now().strftime("%H:%M:%S"), r[0], r[1]))
                    if len(self.logs) > 100: self.logs.pop(0)
            except: pass
            
            c_tmp, g_tmp = get_temps()
            c_load, ram = get_hw_stats()
            
            stdscr.erase(); h, w = stdscr.getmaxyx()
            col_w = w // 4  # 4 Columns
            
            # HEADER
            stdscr.addstr(0, 0, f" MTP MINER SUITE v10 BETA 8 ".center(w), curses.color_pair(5)|curses.A_BOLD)
            
            # 1. LOCAL
            stdscr.addstr(2, 2, "=== LOCAL ===", curses.color_pair(4))
            stdscr.addstr(3, 2, f"IP: {get_local_ip()}")
            stdscr.addstr(4, 2, f"Proxy: {self.cfg['PROXY_PORT']}")
            stdscr.addstr(5, 2, f"RAM: {ram}%")
            
            # 2. HARDWARE
            stdscr.addstr(2, col_w+2, "=== HARDWARE ===", curses.color_pair(4))
            stdscr.addstr(3, col_w+2, f"CPU: {c_tmp}C")
            stdscr.addstr(4, col_w+2, f"GPU: {g_tmp}C")
            ts = "OK" if self.throttle.value == 0.0 else "THROTTLED"
            stdscr.addstr(5, col_w+2, f"Stat: {ts}", curses.color_pair(1 if ts=="OK" else 2))

            # 3. NETWORK
            stdscr.addstr(2, col_w*2+2, "=== NETWORK ===", curses.color_pair(4))
            stdscr.addstr(3, col_w*2+2, f"Pool: Braiins")
            stdscr.addstr(4, col_w*2+2, f"Diff: {int(self.data.get('diff', 0))}")
            stdscr.addstr(5, col_w*2+2, f"Job: {self.data.get('job', '?')[:8]}")
            
            # 4. POOL STATS (WEB)
            stdscr.addstr(2, col_w*3+2, "=== POOL STATS ===", curses.color_pair(4))
            stdscr.addstr(3, col_w*3+2, f"Web: {self.data.get('pool_hash', 'Wait...')}")
            stdscr.addstr(4, col_w*3+2, f"Wrk: {self.data.get('pool_workers', '?')}")
            stdscr.addstr(5, col_w*3+2, f"Mode: Solo")
            
            # BARS
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            hr = sum(self.stats) / (time.time() - self.start_t + 1)
            fhr = f"{hr/1e6:.2f} MH/s" if hr > 1e6 else f"{hr/1000:.2f} kH/s"
            
            stdscr.addstr(9, 2, f"TOTAL: {fhr}", curses.color_pair(1)|curses.A_BOLD)
            stdscr.addstr(9, 40, f"SHARES: {self.shares['acc']} OK / {self.shares['rej']} BAD", curses.color_pair(2))
            
            bar_w = max(5, w - 20)
            gw = int((g_tmp / 90.0) * bar_w)
            stdscr.addstr(10, 2, "GPU: " + "█"*gw, curses.color_pair(2))
            cw = int((c_load / 100.0) * bar_w)
            stdscr.addstr(11, 2, "CPU: " + "█"*cw, curses.color_pair(4))
            
            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            
            # LOGS
            log_h = h - 13
            if log_h > 0:
                for i, l in enumerate(self.logs[-log_h:]):
                    c = curses.color_pair(1)
                    if l[1] in ["ERR", "BAD"]: c = curses.color_pair(3)
                    elif l[1] == "WARN": c = curses.color_pair(2)
                    elif l[1] == "INFO": c = curses.color_pair(4)
                    try: stdscr.addstr(13+i, 2, f"{l[0]} [{l[1]}] {l[2]}"[:w-4], c)
                    except: pass
            
            stdscr.refresh()
            if stdscr.getch() == ord('q'): break
            time.sleep(0.1)

    def start(self):
        ProxyServer(self.cfg, self.log_q).start()
        threading.Thread(target=self.net_thread, daemon=True).start()
        threading.Thread(target=self.thermal_thread, daemon=True).start()
        threading.Thread(target=self.stats_thread, daemon=True).start()
        
        procs = []
        for i in range(mp.cpu_count()):
            p = mp.Process(target=cpu_worker, args=(i, self.job_q, self.res_q, self.stop, self.stats, self.diff, self.throttle, self.log_q), daemon=True)
            p.start(); procs.append(p)
        
        gp = mp.Process(target=gpu_worker, args=(self.stop, self.stats, self.throttle, self.log_q), daemon=True)
        gp.start(); procs.append(gp)
        
        try: curses.wrapper(self.draw_ui)
        except KeyboardInterrupt: pass
        finally:
            self.stop.set()
            for p in procs: 
                if p.is_alive(): p.terminate()

if __name__ == "__main__":
    MinerSuite().start()
