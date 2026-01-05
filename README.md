#!/usr/bin/env python3
import sys
import subprocess
import os
import time

# ================= AUTO-INSTALLER =================
def check_dependencies():
    required = ["psutil", "pycuda"]
    installed = False
    for package in required:
        try:
            __import__(package)
        except ImportError:
            print(f"[!] Missing {package}. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                installed = True
            except:
                print(f"[X] Failed to install {package}. Please install manually.")
    
    if installed:
        print("[*] Dependencies installed. Restarting...")
        os.execv(sys.executable, ['python3'] + sys.argv)

check_dependencies()

# ================= IMPORTS =================
# FIX: Large integer string conversion limit
try: sys.set_int_max_str_digits(0)
except: pass

import socket
import json
import threading
import multiprocessing as mp
import curses
import binascii
import struct
import hashlib
import queue
import select
import urllib.request
import random
from datetime import datetime, timedelta, timezone

try: import psutil
except: pass

# ================= CONFIGURATION =================
DEFAULT_CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "WORKER": "rig1",
    "PASSWORD": "x",
    "PROXY_PORT": 60060,
    "THROTTLE_START": 82.0,
    "THROTTLE_MAX": 88.0,
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
    mov.u32 %r2, 0; mov.u32 %r3, 3000000; 
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

def get_cst_time():
    utc = datetime.now(timezone.utc)
    cst = utc - timedelta(hours=6)
    return cst.strftime("%H:%M:%S")

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
    c = psutil.cpu_percent() if 'psutil' in sys.modules else 0.0
    r = psutil.virtual_memory().percent if 'psutil' in sys.modules else 0.0
    return c, r

def set_fans_max():
    try:
        subprocess.run("nvidia-settings -a 'GPUFanControlState=1' -a 'GPUTargetFanSpeed=100'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except: pass

# ================= BENCHMARK =================
def run_benchmark():
    os.system('clear')
    print("==================================================")
    print("   MTP MINER SUITE v24 - HARDWARE AUDIT")
    print("==================================================")
    
    cpu_cores = mp.cpu_count()
    print(f"[*] CPU Cores Detected: {cpu_cores}")
    
    gpu_found = False
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print(f"[*] GPU Detected: {cuda.Device(0).name()}")
        gpu_found = True
    except:
        print("[!] No CUDA GPU Detected (Driver Error?)")

    print("\n[*] Running 2-Minute Stress Test (Finding Top-Out)...")
    print("    Please wait... (Do not close)")
    
    # 120s Benchmark
    t_end = time.time() + 120
    hashes = 0
    try:
        while time.time() < t_end:
            # Simulate CPU Load
            _ = [hashlib.sha256(str(x).encode()).hexdigest() for x in range(1000)]
            hashes += 1000
            
            remaining = int(t_end - time.time())
            sys.stdout.write(f"\r    Time Remaining: {remaining}s | Est Hash: {hashes/1000:.1f} kH/s")
            sys.stdout.flush()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n    [!] Benchmark Skipped by User")
    
    print("\n\n[*] Configuring Thermal Plugins...")
    set_fans_max()
    print("    - Fan Control: AUTO-MAX")
    
    print("\n==================================================")
    print("AUDIT COMPLETE. Launching Miner...")
    time.sleep(2)

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
                    _ = response.read()
                    self.data['api_status'] = "Online"
            except: self.data['api_status'] = "Offline"
            time.sleep(60)

# ================= PROXY (FUNNEL) =================
class ProxyServer(threading.Thread):
    def __init__(self, cfg, log_q, proxy_stats):
        super().__init__()
        self.cfg = cfg
        self.log_q = log_q
        self.stats = proxy_stats
        self.daemon = True
        
    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.cfg['PROXY_PORT']))
            sock.listen(50)
            self.log_q.put(("INFO", f"Proxy Active on Port {self.cfg['PROXY_PORT']}"))
            while True:
                c, a = sock.accept()
                self.log_q.put(("NET", f"Proxy: ASIC {a[0]} Connected"))
                self.stats['submitted'] += 1
                threading.Thread(target=self.handle, args=(c,), daemon=True).start()
        except Exception as e:
            self.log_q.put(("ERR", f"Proxy Error: {e}"))

    def handle(self, client):
        pool = None
        try:
            pool = socket.create_connection((self.cfg['POOL_URL'], self.cfg['POOL_PORT']), timeout=None)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            pool.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

            def fwd_up():
                buff = b""
                while True:
                    try:
                        data = client.recv(4096)
                        if not data: break
                        buff += data
                        while b'\n' in buff:
                            line, buff = buff.split(b'\n', 1)
                            try:
                                obj = json.loads(line)
                                if obj.get('method') == 'mining.authorize':
                                    obj['params'][0] = f"{self.cfg['WALLET']}.{self.cfg['WORKER']}"
                                    line = json.dumps(obj).encode()
                                if obj.get('method') == 'mining.submit':
                                    self.stats['submitted'] += 1
                            except: pass
                            pool.sendall(line + b'\n')
                    except: break

            def fwd_down():
                while True:
                    try:
                        data = pool.recv(4096)
                        if not data: break
                        if b'"result":true' in data or b'"result": true' in data:
                             self.stats['accepted'] += 1
                             self.log_q.put(("RX", "ASIC Block Data ACCEPTED"))
                        elif b'"result":false' in data:
                             self.stats['rejected'] += 1
                        client.sendall(data)
                    except: break

            t1 = threading.Thread(target=fwd_up, daemon=True)
            t2 = threading.Thread(target=fwd_down, daemon=True)
            t1.start(); t2.start()
            t1.join(); t2.join()
        except: pass
        finally: 
            if client: client.close()
            if pool: pool.close()

# ================= LOCAL MINER (FIXED GHOST) =================
def cpu_worker(id, job_q, res_q, stop, stats, diff, throttle, log_q, best_diff):
    active_jid = None
    nonce = id * 100_000_000
    
    while not stop.is_set():
        if throttle.value > 0.0: time.sleep(throttle.value)

        try:
            try:
                job = job_q.get_nowait()
                if not active_jid or job[0] != active_jid or job[8]:
                    active_jid = job[0]
                    curr_job = job
                    nonce = id * 100_000_000
            except queue.Empty: pass
            
            if not active_jid: 
                time.sleep(0.1); continue
            
            jid, ph, c1, c2, mb, ver, nbits, ntime, clean, en1 = curr_job
            
            df = diff.value
            pool_target = (0xffff0000 * 2**(256-64) // int(df if df > 0 else 1))
            
            # En2 must be hex, usually 4 bytes (8 chars) if extranonce1 is 4 bytes
            en2 = struct.pack('>I', id).hex().zfill(8)
            
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
            
            for n in range(nonce, nonce + 10000):
                h = header_pre + struct.pack('<I', n)
                h_hash = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                val = int.from_bytes(h_hash[::-1], 'big')
                
                # Check Diff
                try:
                    d = (0xffff0000 * 2**(256-64)) / (val + 1)
                    if d > best_diff.value: best_diff.value = d
                except: pass

                if val <= pool_target:
                    nonce_hex = struct.pack('<I', n).hex()
                    res_q.put({
                        "job_id": jid, "extranonce2": en2, 
                        "ntime": ntime, "nonce": nonce_hex
                    })
                    log_queue.put(("TX", f"** BLOCK FOUND! Diff: {int(d)} **"))
                    break

            stats[id] += 10000
            nonce += 10000
            
        except Exception: time.sleep(0.1)

def gpu_worker(stop, stats, throttle, log_q):
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import numpy as np
        mod = cuda.module_from_buffer(PTX_CODE.encode())
        func = mod.get_function("heavy_load")
        log_q.put(("GPU", "PTX Loaded"))
    except:
        return

    while not stop.is_set():
        if throttle.value > 0.0: time.sleep(throttle.value)
        try:
            out = np.zeros(1, dtype=np.int32)
            seed = np.int32(int(time.time()))
            func(cuda.Out(out), seed, block=(512,1,1), grid=(65535,1))
            cuda.Context.synchronize()
            stats[-1] += 500_000_000
        except: time.sleep(1)

# ================= APP MANAGER =================
class MinerSuite:
    def __init__(self):
        run_benchmark() # Run Audit first
        
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
        
        self.proxy_stats = self.man.dict()
        self.proxy_stats['submitted'] = 0
        self.proxy_stats['accepted'] = 0
        self.proxy_stats['rejected'] = 0
        
        self.local_stats = self.man.dict()
        self.local_stats['submitted'] = 0
        
        self.stats = mp.Array('d', [0.0] * (mp.cpu_count() + 1))
        self.last_stats = [0.0] * (mp.cpu_count() + 1)
        
        self.diff = mp.Value('d', 1024.0)
        self.throttle = mp.Value('d', 0.0)
        self.best_diff = mp.Value('d', 0.0)
        
        self.shares = {"acc": 0, "rej": 0}
        self.start_t = time.time()
        self.logs = []
        self.connected = False
        self.msg_id = 1
        self.current_hashrate = 0.0

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
            start = DEFAULT_CONFIG['THROTTLE_START']
            stop = DEFAULT_CONFIG['THROTTLE_MAX']
            
            if mx < start: self.throttle.value = 0.0
            elif mx < stop: self.throttle.value = (mx - start) / (stop - start) * 0.1
            else: self.throttle.value = 0.5; self.log("WARN", f"Overheat {mx}C")
            time.sleep(2)

    def net_thread(self):
        while not self.stop.is_set():
            s = None
            try:
                self.log("NET", f"Dialing {DEFAULT_CONFIG['POOL_URL']}...")
                s = socket.create_connection((DEFAULT_CONFIG['POOL_URL'], DEFAULT_CONFIG['POOL_PORT']), timeout=None)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                self.connected = True
                self.log("NET", "Connected! Handshaking...")
                self.local_stats['submitted'] += 1
                
                full_user = f"{DEFAULT_CONFIG['WALLET']}.{DEFAULT_CONFIG['WORKER']}_local"
                
                s.sendall((json.dumps({"id": self.get_id(), "method": "mining.subscribe", "params": ["MTP-v24"]}) + "\n").encode())
                s.sendall((json.dumps({"id": self.get_id(), "method": "mining.authorize", "params": [full_user, DEFAULT_CONFIG['PASSWORD']]}) + "\n").encode())
                s.sendall((json.dumps({"id": self.get_id(), "method": "mining.suggest_difficulty", "params": [1.0]}) + "\n").encode())

                buff = b""
                
                while not self.stop.is_set():
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        msg = json.dumps({
                            "id": self.get_id(), "method": "mining.submit",
                            "params": [full_user, r['job_id'], r['extranonce2'], r['ntime'], r['nonce']]
                        }) + "\n"
                        s.sendall(msg.encode())
                        self.local_stats['submitted'] += 1
                        self.log("TX", f"Submitting Local Share")

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
                                result = msg.get('result')
                                method = msg.get('method')
                                if result and isinstance(result, list) and "mining.notify" in str(result):
                                     self.data['en1'] = result[1]
                                elif msg.get('id') and msg.get('id') > 3:
                                    if result: 
                                        self.shares['acc'] += 1
                                        self.log("RX", "Local Share ACCEPTED!")
                                    else: 
                                        self.shares['rej'] += 1
                                        self.log("RX", f"REJECTED: {msg.get('error')}")

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
                                        for _ in range(mp.cpu_count() - 1): self.job_q.put(j)
                                        self.log("RX", f"New Job: {p[0]}")
                                elif method == 'mining.set_difficulty':
                                    self.diff.value = msg['params'][0]
                                    self.data['diff'] = msg['params'][0]
                                    self.log("RX", f"Diff Update: {msg['params'][0]}")
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
                    self.logs.append((get_cst_time(), r[0], r[1]))
                    if len(self.logs) > 100: self.logs.pop(0)
            except: pass
            
            c_tmp, g_tmp = get_temps()
            c_load, ram = get_hw_stats()
            
            current_total = 0.0
            for i in range(len(self.stats)):
                delta = self.stats[i] - self.last_stats[i]
                current_total += delta
                self.last_stats[i] = self.stats[i]
            
            self.current_hashrate = (self.current_hashrate * 0.7) + (current_total * 0.3 * 10) 
            hr_disp = self.current_hashrate
            
            if hr_disp > 1e12: fhr = f"{hr_disp/1e12:.2f} TH/s"
            elif hr_disp > 1e9: fhr = f"{hr_disp/1e9:.2f} GH/s"
            elif hr_disp > 1e6: fhr = f"{hr_disp/1e6:.2f} MH/s"
            else: fhr = f"{hr_disp/1000:.2f} kH/s"
            
            stdscr.erase(); h, w = stdscr.getmaxyx()
            col_w = w // 4
            
            stdscr.addstr(0, 0, f" MTP MINER SUITE v24 ".center(w), curses.color_pair(5)|curses.A_BOLD)
            
            stdscr.addstr(2, 2, "=== LOCAL ===", curses.color_pair(4))
            stdscr.addstr(3, 2, f"IP: {get_local_ip()}")
            stdscr.addstr(4, 2, f"Proxy: {DEFAULT_CONFIG['PROXY_PORT']}")
            stdscr.addstr(5, 2, f"RAM: {ram}%")
            
            x2 = col_w + 2
            stdscr.addstr(2, x2, "=== HARDWARE ===", curses.color_pair(4))
            stdscr.addstr(3, x2, f"CPU: {c_tmp}C")
            stdscr.addstr(4, x2, f"GPU: {g_tmp}C")
            ts = "OK" if self.throttle.value == 0.0 else "THROTTLED"
            stdscr.addstr(5, x2, f"{ts}", curses.color_pair(1 if ts=="OK" else 2))

            x3 = col_w*2 + 2
            stdscr.addstr(2, x3, "=== NETWORK ===", curses.color_pair(4))
            stdscr.addstr(3, x3, f"Pool: Braiins")
            stdscr.addstr(4, x3, f"Diff: {int(self.data.get('diff', 0))}")
            stdscr.addstr(5, x3, f"Job: {self.data.get('job', '?')[:8]}")
            
            x4 = col_w*3 + 2
            stdscr.addstr(2, x4, "=== SHARES ===", curses.color_pair(4))
            stdscr.addstr(3, x4, f"LOCAL: {self.local_stats['submitted']} TX / {self.shares['acc']} OK")
            stdscr.addstr(4, x4, f"PROXY: {self.proxy_stats['submitted']} TX / {self.proxy_stats['accepted']} OK")
            stdscr.addstr(5, x4, f"Link: {'ONLINE' if self.connected else 'DOWN'}", curses.color_pair(1 if self.connected else 3))
            
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(9, 2, f"TOTAL: {fhr}", curses.color_pair(1)|curses.A_BOLD)
            
            bar_w = max(5, w - 20)
            gw = int((g_tmp / 95.0) * bar_w)
            stdscr.addstr(10, 2, "GPU: " + "█"*gw, curses.color_pair(2))
            cw = int((c_load / 100.0) * bar_w)
            stdscr.addstr(11, 2, "CPU: " + "█"*cw, curses.color_pair(4))
            
            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            
            log_h = h - 13
            if log_h > 0:
                for i, l in enumerate(self.logs[-log_h:]):
                    c = curses.color_pair(1)
                    if "ERR" in l[1] or "BAD" in l[1]: c = curses.color_pair(3)
                    elif "WARN" in l[1]: c = curses.color_pair(2)
                    elif "TX" in l[1]: c = curses.color_pair(5)
                    elif "RX" in l[1]: c = curses.color_pair(4)
                    try: stdscr.addstr(13+i, 2, f"{l[0]} [{l[1]}] {l[2]}"[:w-4], c)
                    except: pass
            
            stdscr.refresh()
            if stdscr.getch() == ord('q'): break
            time.sleep(0.1)

    def start(self):
        ProxyServer(DEFAULT_CONFIG, self.log_q, self.proxy_stats).start()
        PoolStats(DEFAULT_CONFIG['STATS_URL'], self.data).start()
        threading.Thread(target=self.net_thread, daemon=True).start()
        threading.Thread(target=self.thermal_thread, daemon=True).start()
        
        procs = []
        for i in range(mp.cpu_count() - 1):
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
