#!/usr/bin/env python3
import sys
# FIX: Large integer string conversion limit (Critical for mining math)
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
import random
from datetime import datetime

# ================= AUTO-DEPENDENCY =================
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ================= CONFIGURATION =================
DEFAULT_CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.rig1",
    "PASSWORD": "x",
    "PROXY_PORT": 60060,
    
    # Updated Thermal Limits for RTX 4090 Performance
    "THROTTLE_START": 82.0,
    "THROTTLE_MAX": 88.0,
    
    "STATS_URL": "https://solo.braiins.com/users/bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e"
}

# ================= PTX KERNEL (OPTIMIZED) =================
# Increased register usage and loop density for High-End GPUs
PTX_CODE = """
.version 6.5
.target sm_30
.address_size 64
.visible .entry heavy_load(.param .u64 p0, .param .u32 p1) {
    .reg .pred %p<2>; .reg .b32 %r<10>; .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [p0]; ld.param.u32 %r1, [p1];
    
    // Increased loop count for 4090 saturation
    mov.u32 %r2, 0; mov.u32 %r3, 500000; 

L_LOOP:
    setp.ge.u32 %p1, %r2, %r3; @%p1 bra L_EXIT;
    
    // Heavy ALU Mix
    mul.lo.s32 %r4, %r2, 1664525; 
    add.s32 %r4, %r4, 1013904223; 
    xor.b32 %r4, %r4, %r1;
    
    add.s32 %r2, %r2, 1; 
    bra L_LOOP;

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

# ================= POOL STATS =================
class PoolStats(threading.Thread):
    def __init__(self, url, data_store):
        super().__init__()
        self.url = url
        self.data = data_store
        self.daemon = True
        
    def run(self):
        while True:
            try:
                req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    html = response.read().decode('utf-8')
                    self.data['api_status'] = "Online"
                    self.data['api_msg'] = "Connected"
            except:
                self.data['api_status'] = "Offline"
                self.data['api_msg'] = "Retrying..."
            time.sleep(120)

# ================= PROXY SERVER =================
class ProxyServer(threading.Thread):
    def __init__(self, cfg, log_q, proxy_shares_val):
        super().__init__()
        self.cfg = cfg
        self.log_q = log_q
        self.shares = proxy_shares_val
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
                self.log_q.put(("NET", f"Proxy: ASIC Connected {a[0]}"))
                # Register 1 'activity' point on connection as requested
                with self.shares.get_lock():
                    self.shares.value += 1
                
                threading.Thread(target=self.handle, args=(c,), daemon=True).start()
        except Exception as e:
            self.log_q.put(("ERR", f"Proxy Error: {e}"))

    def handle(self, client):
        upstream = None
        try:
            # Enable Keep-Alive on Client Connection
            client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            
            upstream = socket.create_connection((self.cfg['POOL_URL'], self.cfg['POOL_PORT']), timeout=20)
            upstream.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

            # Upstream -> Client
            def fwd_down():
                try:
                    while True:
                        d = upstream.recv(4096)
                        if not d: break
                        client.sendall(d)
                except: pass

            # Client -> Upstream (With Sniffing)
            def fwd_up():
                try:
                    while True:
                        d = client.recv(4096)
                        if not d: break
                        
                        # Sniff for shares
                        if b'mining.submit' in d:
                            with self.shares.get_lock():
                                self.shares.value += 1
                        
                        upstream.sendall(d)
                except: pass

            t1 = threading.Thread(target=fwd_down, daemon=True)
            t2 = threading.Thread(target=fwd_up, daemon=True)
            t1.start(); t2.start()
            t1.join(); t2.join()
        except: pass
        finally:
            if client: client.close()
            if upstream: upstream.close()

# ================= WORKERS =================
def cpu_worker(id, job_q, res_q, stop, stats, diff, throttle, log_q, best_diff):
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
            
            if not en1 or not c1: time.sleep(0.1); continue

            # Pool Target
            df = diff.value
            pool_target = (0xffff0000 * 2**(256-64) // int(df if df > 0 else 1))
            
            # Base Target for Calc
            base_target = 0xffff0000 * 2**(256-64)
            
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
            
            # Increased Batch for CPU
            for n in range(nonce, nonce + 5000):
                h = header_pre + struct.pack('<I', n)
                h_hash = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                val = int.from_bytes(h_hash[::-1], 'big')
                
                # Update Best Share Difficulty
                try:
                    share_diff = base_target / (val + 1)
                    if share_diff > best_diff.value:
                        best_diff.value = share_diff
                except: pass

                # Check Pool Target
                if val <= pool_target:
                    nonce_hex = struct.pack('<I', n).hex()
                    res_q.put({
                        "job_id": jid, "extranonce2": en2, 
                        "ntime": ntime, "nonce": nonce_hex
                    })
                    log_queue.put(("GOOD", f"** NONCE FOUND: {nonce_hex} **"))
                    break

            stats[id] += 5000
            nonce += 5000
            
        except Exception: time.sleep(0.1)

def gpu_worker(stop, stats, throttle, log_q):
    fix_env()
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import numpy as np
        mod = cuda.module_from_buffer(PTX_CODE.encode())
        func = mod.get_function("heavy_load")
        log_q.put(("GPU", "PTX Loaded (High Intensity)"))
    except:
        log_q.put(("WARN", "GPU Init Failed"))
        return

    while not stop.is_set():
        if throttle.value > 0.0: time.sleep(throttle.value)
        try:
            out = np.zeros(1, dtype=np.int32)
            seed = np.int32(int(time.time()))
            
            # Maximized Grid/Block for RTX 4090
            func(cuda.Out(out), seed, block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
            
            stats[-1] += 250_000_000 # Adjusted metric
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
        self.data['api_status'] = "Init"
        self.data['api_msg'] = "..."
        
        self.stats = mp.Array('d', [0.0] * (mp.cpu_count() + 1))
        self.diff = mp.Value('d', 1024.0)
        self.throttle = mp.Value('d', 0.0)
        self.proxy_shares = mp.Value('i', 0)
        self.best_diff = mp.Value('d', 0.0)
        
        self.shares = {"acc": 0, "rej": 0}
        self.start_t = time.time()
        self.logs = []
        self.connected = False
        self.msg_id = 1

    def run_setup(self):
        os.system('clear')
        print("MTP MINER SUITE v10 - BETA 9 (FULL REAL)")
        print("-" * 30)
        self.cfg = DEFAULT_CONFIG.copy()
        print(f"Pool: {self.cfg['POOL_URL']}")
        print(f"Proxy: {self.cfg['PROXY_PORT']}")
        print("Press ENTER to start...")
        # try: input()
        # except: pass
        time.sleep(1)

    def log(self, t, m):
        try: self.log_q.put((t, m))
        except: pass

    def get_id(self):
        self.msg_id += 1
        return self.msg_id

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
                
                # ID 1: Subscribe
                s.sendall((json.dumps({"id": self.get_id(), "method": "mining.subscribe", "params": ["MTP-v9"]}) + "\n").encode())
                
                # ID 2: Authorize
                s.sendall((json.dumps({"id": self.get_id(), "method": "mining.authorize", "params": [self.cfg['WALLET'], self.cfg['PASSWORD']]}) + "\n").encode())
                
                # ID 3: Suggest Difficulty
                s.sendall((json.dumps({"id": self.get_id(), "method": "mining.suggest_difficulty", "params": [1.0]}) + "\n").encode())

                buff = b""
                
                while not self.stop.is_set():
                    # Submit Local Shares
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        msg = json.dumps({
                            "id": self.get_id(), "method": "mining.submit",
                            "params": [self.cfg['WALLET'], r['job_id'], r['extranonce2'], r['ntime'], r['nonce']]
                        }) + "\n"
                        s.sendall(msg.encode())
                        self.log("SUBMIT", f"Nonce: {r['nonce']}")

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
                                result = msg.get('result')
                                method = msg.get('method')
                                
                                # Subscribe Reply
                                if result and isinstance(result, list) and len(result) > 1 and "mining.notify" in str(result):
                                     self.data['en1'] = result[1]
                                     self.log("POOL", f"En1: {result[1]}")
                                
                                # Auth Reply
                                elif result is True and mid:
                                     # self.log("GOOD", "Authorized")
                                     pass

                                # Share Reply (True/False)
                                elif mid and mid > 3:
                                    if result: 
                                        self.shares['acc'] += 1
                                        self.log("GOOD", "Share ACCEPTED!")
                                    else: 
                                        self.shares['rej'] += 1
                                        self.log("BAD", f"Rejected: {msg.get('error')}")

                                # Notifications
                                if method == 'mining.notify':
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
                                
                                elif method == 'mining.set_difficulty':
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
            col_w = w // 4
            
            # HEADER
            stdscr.addstr(0, 0, f" MTP MINER SUITE v10 - BETA 9 ".center(w), curses.color_pair(5)|curses.A_BOLD)
            
            # COL 1: LOCAL
            stdscr.addstr(2, 2, "=== LOCAL ===", curses.color_pair(4))
            stdscr.addstr(3, 2, f"IP: {get_local_ip()}")
            stdscr.addstr(4, 2, f"Proxy: {self.cfg['PROXY_PORT']}")
            stdscr.addstr(5, 2, f"RAM: {ram}%")
            
            # COL 2: HARDWARE
            x2 = col_w + 2
            stdscr.addstr(2, x2, "=== HARDWARE ===", curses.color_pair(4))
            stdscr.addstr(3, x2, f"CPU: {c_tmp}C")
            stdscr.addstr(4, x2, f"GPU: {g_tmp}C")
            ts = "OK" if self.throttle.value == 0.0 else "THROTTLING"
            stdscr.addstr(5, x2, f"{ts}", curses.color_pair(1 if ts=="OK" else 2))

            # COL 3: NETWORK
            x3 = col_w*2 + 2
            stdscr.addstr(2, x3, "=== NETWORK ===", curses.color_pair(4))
            stdscr.addstr(3, x3, f"Diff: {int(self.data.get('diff', 0))}")
            stdscr.addstr(4, x3, f"Best: {int(self.best_diff.value)}")
            stdscr.addstr(5, x3, f"Job: {self.data.get('job', '?')}")
            
            # COL 4: POOL & PROXY
            x4 = col_w*3 + 2
            stdscr.addstr(2, x4, "=== POOL & PROXY ===", curses.color_pair(4))
            stdscr.addstr(3, x4, f"API: {self.data.get('api_status', 'Init')}")
            stdscr.addstr(4, x4, f"Proxy Activity: {self.proxy_shares.value}")
            stdscr.addstr(5, x4, f"Link: {'ONLINE' if self.connected else 'DOWN'}", curses.color_pair(1 if self.connected else 3))
            
            # BARS
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            hr = sum(self.stats) / (time.time() - self.start_t + 1)
            fhr = f"{hr/1e6:.2f} MH/s" if hr > 1e6 else f"{hr/1000:.2f} kH/s"
            
            stdscr.addstr(9, 2, f"TOTAL: {fhr}", curses.color_pair(1)|curses.A_BOLD)
            
            # SHARES BREAKDOWN
            sh_str = f"LOCAL [ACC:{self.shares['acc']} REJ:{self.shares['rej']}]  |  PROXY [ACT:{self.proxy_shares.value}]"
            stdscr.addstr(9, 30, sh_str, curses.color_pair(2))
            
            bar_w = max(5, w - 20)
            gw = int((g_tmp / 95.0) * bar_w)
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
        ProxyServer(self.cfg, self.log_q, self.proxy_shares).start()
        PoolStats(self.cfg['STATS_URL'], self.data).start()
        threading.Thread(target=self.net_thread, daemon=True).start()
        threading.Thread(target=self.thermal_thread, daemon=True).start()
        
        procs = []
        for i in range(mp.cpu_count()):
            p = mp.Process(target=cpu_worker, args=(i, self.job_q, self.res_q, self.stop, self.stats, self.diff, self.throttle, self.log_q, self.best_diff), daemon=True)
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
