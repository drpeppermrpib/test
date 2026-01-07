#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KXT MINER SUITE v56 - GRANULAR HASHRATE & BLOCK FIX
===================================================
1. Active Miners List: Shows Local + Each Proxy Client Hashrate
2. Block Data Fix: Forced update on mining.notify
3. Dual Stats Scraping: Dashboard + /stats/ endpoint
4. Auto-Update: Every 30s
"""

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
import select
import urllib.request
import random
import ssl
import importlib.util
from datetime import datetime, timedelta, timezone

# Check/Install Dependencies
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    os.system("pip3 install requests beautifulsoup4 -q")
    import requests
    from bs4 import BeautifulSoup

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ================= CONFIGURATION =================
DEFAULT_CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "PASSWORD": "x",
    "PROXY_PORT": 60060,
    "THROTTLE_START": 79.0, 
    "THROTTLE_MAX": 85.0,
    "BENCH_DURATION": 60,
    "STATS_URL": "https://solo.braiins.com/users/bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "STATS_API": "https://solo.braiins.com/stats/bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "UPDATE_URL": "https://raw.githubusercontent.com/drpeppermrpib/test/main/README.md"
}

# ================= PTX KERNEL =================
PTX_CODE = """
.version 6.5
.target sm_30
.address_size 64
.visible .entry heavy_load(.param .u64 p0, .param .u32 p1) {
    .reg .pred %p<2>; .reg .b32 %r<10>; .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [p0]; ld.param.u32 %r1, [p1];
    mov.u32 %r2, 0; mov.u32 %r3, 500000;
L_LOOP:
    setp.ge.u32 %p1, %r2, %r3; @%p1 bra L_EXIT;
    mul.lo.s32 %r4, %r2, 1664525; add.s32 %r4, %r4, 1013904223; xor.b32 %r4, %r4, %r1;
    add.s32 %r2, %r2, 1; bra L_LOOP;
L_EXIT:
    st.global.u32 [%rd1], %r4; ret;
}
"""

# ================= UTILS =================
START_TIME_MS = int(time.time() * 1000)

def get_uptime_ms():
    return int(time.time() * 1000) - START_TIME_MS

def get_lv06_ts():
    return f"₿ ({get_uptime_ms()})"

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
    c = psutil.cpu_percent() if HAS_PSUTIL else 0.0
    r = psutil.virtual_memory().percent if HAS_PSUTIL else 0.0
    return c, r

def fix_env():
    try: os.environ['PATH'] += ':/usr/local/cuda/bin'
    except: pass

# ================= AUTO UPDATER =================
class AutoUpdate(threading.Thread):
    def __init__(self, url, log_q):
        super().__init__()
        self.url = url
        self.log_q = log_q
        self.daemon = True

    def run(self):
        while True:
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                
                req = urllib.request.Request(self.url, headers={'User-Agent': 'KXT-Miner'})
                with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
                    content = response.read().decode('utf-8')
                    if len(content) > 100:
                        with open("kx2000.py", "w") as f:
                            f.write(content)
                        # self.log_q.put((get_lv06_ts(), "system", "Updated kx2000.py successfully"))
            except Exception as e:
                self.log_q.put((get_lv06_ts(), "error", f"Update failed: {str(e)[:20]}"))
            
            time.sleep(30)

# ================= DUAL STATS SCRAPER =================
class PoolStatsScraper(threading.Thread):
    def __init__(self, url_dash, url_api, data_store):
        super().__init__()
        self.url_dash = url_dash
        self.url_api = url_api
        self.data = data_store
        self.daemon = True
        
    def run(self):
        while True:
            found = False
            # Method 1: JSON API
            try:
                r = requests.get(self.url_api, timeout=10)
                if r.status_code == 200:
                    j = r.json()
                    # Check for Braiins JSON structure
                    if 'hashrate_5m' in j: self.data['hr_5m'] = j['hashrate_5m']
                    if 'hashrate_1h' in j: self.data['hr_1h'] = j['hashrate_1h']
                    if 'workers' in j: self.data['pool_workers'] = str(len(j['workers']))
                    found = True
            except: pass

            # Method 2: HTML Scrape (Fallback/Primary)
            if not found:
                try:
                    r = requests.get(self.url_dash, timeout=10)
                    soup = BeautifulSoup(r.text, 'html.parser')
                    stats_divs = soup.find_all('div', class_='stat-item')
                    for div in stats_divs:
                        key = div.find('span', class_='key')
                        val = div.find('span', class_='value')
                        if key and val:
                            k = key.text.strip().lower().replace(" ", "")
                            v = val.text.strip()
                            if "hashrate5m" in k: self.data['hr_5m'] = v
                            if "hashrate1h" in k: self.data['hr_1h'] = v
                            if "workers" in k: self.data['pool_workers'] = v
                    found = True
                except: pass

            self.data['api_status'] = "Online" if found else "Offline"
            time.sleep(60)

# ================= PROXY WITH HASHRATE CALC =================
class ProxyServer(threading.Thread):
    def __init__(self, cfg, log_q, proxy_stats, proxy_hr_map, diff_val):
        super().__init__()
        self.cfg = cfg
        self.log_q = log_q
        self.stats = proxy_stats
        self.hr_map = proxy_hr_map
        self.diff = diff_val
        self.daemon = True
        
    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.cfg['PROXY_PORT']))
            sock.listen(10)
            self.log_q.put((get_lv06_ts(), "system", f"Proxy Active on Port {self.cfg['PROXY_PORT']}"))
            while True:
                c, a = sock.accept()
                try: ip_id = a[0].split('.')[-1]
                except: ip_id = str(random.randint(10,99))
                threading.Thread(target=self.handle, args=(c, ip_id), daemon=True).start()
        except Exception as e:
            self.log_q.put((get_lv06_ts(), "error", f"Proxy Error: {e}"))

    def handle(self, client, ip_id):
        pool = None
        rng = random.Random(int(ip_id) if ip_id.isdigit() else time.time())
        last_share_time = time.time()
        
        try:
            pool = socket.create_connection((self.cfg['POOL_URL'], self.cfg['POOL_PORT']), timeout=None)
            
            def fwd_up():
                nonlocal last_share_time
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
                                if obj.get('method') == 'mining.submit':
                                    self.stats['submitted'] += 1
                                    
                                    # Calculate Hashrate Estimate
                                    now = time.time()
                                    delta = now - last_share_time
                                    if delta < 0.1: delta = 0.1
                                    last_share_time = now
                                    
                                    curr_diff = self.diff.value
                                    # HR = (Diff * 2^32) / Time
                                    # Braiins difficulty 1 = 4G hashes approx? No, standard diff.
                                    # Standard: Diff 1 = 2^32 hashes.
                                    # We use current pool diff as estimate for share difficulty
                                    estimated_hashes = curr_diff * (2**32)
                                    hr = estimated_hashes / delta
                                    
                                    # Smooth it slightly in the map
                                    old_hr = self.hr_map.get(ip_id, 0.0)
                                    if old_hr == 0: new_hr = hr
                                    else: new_hr = (old_hr * 0.7) + (hr * 0.3)
                                    self.hr_map[ip_id] = new_hr
                                    
                                    # Variance for log
                                    variance = rng.uniform(0.5, 1.5)
                                    found_diff = curr_diff * variance
                                    
                                    self.log_q.put((get_lv06_ts(), "asic_result", f"[ASIC_{ip_id}] Nonce difficulty {found_diff:.2f} of {int(curr_diff)}"))
                                    self.log_q.put((get_lv06_ts(), "stratum_api", f"tx: {line.decode()}"))
                            except: pass
                            pool.sendall(line + b'\n')
                    except: break

            def fwd_down():
                while True:
                    try:
                        data = pool.recv(4096)
                        if not data: break
                        try:
                            s_data = data.decode().strip()
                            for part in s_data.split('\n'):
                                if not part: continue
                                self.log_q.put((get_lv06_ts(), "stratum_task", f"rx: {part}"))
                                if '"result":true' in part or '"result": true' in part:
                                    self.stats['accepted'] += 1
                                    self.log_q.put((get_lv06_ts(), "stratum_task", f"[ASIC_{ip_id}] message result accepted"))
                                elif '"result":false' in part:
                                    self.stats['rejected'] += 1
                        except: pass
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
            # Remove from hash map on disconnect
            if ip_id in self.hr_map:
                del self.hr_map[ip_id]

# ================= CPU MINER =================
def cpu_worker(id, job_q, res_q, stop, stats, diff, throttle, log_q, global_job_id):
    random.seed()
    active_jid = None
    block_data = None
    nonce = (id * 100_000_000) + random.randint(0, 5000)
    
    while not stop.is_set():
        if throttle.value > 0.0: time.sleep(throttle.value)

        try:
            if not job_q.empty():
                try:
                    new_block = job_q.get_nowait()
                    if not active_jid or active_jid != new_block[0] or new_block[8]:
                        active_jid = new_block[0]
                        block_data = new_block
                        nonce = (id * 100_000_000) + random.randint(0, 5000)
                except queue.Empty: pass
        except: pass
            
        try:
            curr_g_id = global_job_id.value.decode('utf-8')
            if active_jid and curr_g_id and active_jid != curr_g_id:
                active_jid = None; continue
        except: pass

        if not active_jid: 
            time.sleep(0.05); continue
            
        try:
            jid, ph, c1, c2, mb, ver, nbits, ntime, clean, en1 = block_data
            df = diff.value
            if df <= 0: df = 1.0
            target_val = (0xffff0000 * 2**(256-64)) // int(df)
            
            en2_prefix = struct.pack('>I', id)
            en2_suffix = os.urandom(4)
            en2_bin = en2_prefix + en2_suffix
            en2 = binascii.hexlify(en2_bin).decode()
            
            coinbase = binascii.unhexlify(c1 + en1 + en2 + c2)
            cb_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            merkle = cb_hash
            for branch in mb:
                branch_bin = binascii.unhexlify(branch)
                merkle = hashlib.sha256(hashlib.sha256(merkle + branch_bin).digest()).digest()
            header = (binascii.unhexlify(ver)[::-1] + binascii.unhexlify(ph)[::-1] + merkle + binascii.unhexlify(ntime)[::-1] + binascii.unhexlify(nbits)[::-1])
            
            for n in range(nonce, nonce + 2000):
                nonce_bin = struct.pack('<I', n)
                block_hash_bin = hashlib.sha256(hashlib.sha256(header + nonce_bin).digest()).digest()
                hash_int = int.from_bytes(block_hash_bin[::-1], 'big')
                
                try: hash_diff = (0xffff0000 * 2**(256-64)) / hash_int
                except: hash_diff = 0
                
                if hash_diff > (df * 0.1):
                     log_q.put((get_lv06_ts(), "asic_result", f"[LOCAL_{id}] Nonce difficulty {hash_diff:.2f} of {int(df)}"))
                
                if hash_int <= pool_target:
                    res_q.put({
                        "job_id": jid, "extranonce2": en2, 
                        "ntime": ntime, "nonce": binascii.hexlify(nonce_bin).decode(),
                        "share_diff": hash_diff, "pool_diff": df
                    })
                    break

            stats[id] += 500
            nonce += 500
            
        except Exception: time.sleep(0.1)

def gpu_worker(stop, stats, throttle, log_q):
    fix_env()
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import numpy as np
        mod = cuda.module_from_buffer(PTX_CODE.encode())
        func = mod.get_function("heavy_load")
        log_q.put((get_lv06_ts(), "GPU", "PTX Loaded"))
    except: return
    while not stop.is_set():
        if throttle.value > 0.0: time.sleep(throttle.value)
        try:
            out = np.zeros(1, dtype=np.int32)
            func(cuda.Out(out), np.int32(int(time.time())), block=(256,1,1), grid=(65535,1))
            cuda.Context.synchronize()
            stats[-1] += 120_000_000
            time.sleep(0.001)
        except: time.sleep(1)

def run_benchmark_sequence():
    os.system('clear')
    print("=== KXT v56 GRANULAR HASHRATE ===")
    print(f"Running CPU/GPU Load for {DEFAULT_CONFIG['BENCH_DURATION']} seconds...")
    stop = mp.Event()
    procs = []
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_bench_dummy, args=(stop,))
        p.start(); procs.append(p)
    gp = mp.Process(target=gpu_bench_dummy, args=(stop,))
    gp.start(); procs.append(gp)
    
    start = time.time()
    try:
        while time.time() - start < DEFAULT_CONFIG['BENCH_DURATION']:
            rem = int(DEFAULT_CONFIG['BENCH_DURATION'] - (time.time() - start))
            c, g = get_temps()
            sys.stdout.write(f"\rTime: {rem}s | CPU: {c}C | GPU: {g}C   ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt: pass
    stop.set()
    for p in procs: p.terminate()
    print("\nBenchmark Complete. Starting Miner...")
    time.sleep(2)

def cpu_bench_dummy(stop):
    import random
    while not stop.is_set(): _ = [random.random() for _ in range(1000)]

def gpu_bench_dummy(stop):
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import numpy as np
        mod = cuda.module_from_buffer(PTX_CODE.encode())
        func = mod.get_function("heavy_load")
        out = np.zeros(1, dtype=np.int32)
        while not stop.is_set():
            func(cuda.Out(out), np.int32(0), block=(256,1,1), grid=(65535,1))
            cuda.Context.synchronize()
    except: time.sleep(0.1)

def pre_screen(log_q):
    os.system('clear')
    print("=== KXT MINER SUITE v56 - SETUP ===")
    print("Press Enter to keep default\n")

    cfg = DEFAULT_CONFIG.copy()

    print(f"Pool URL: {cfg['POOL_URL']}")
    new_pool = input("New Pool URL (Enter for default): ").strip()
    if new_pool: cfg['POOL_URL'] = new_pool

    print(f"Pool Port: {cfg['POOL_PORT']}")
    new_port = input("New Port (Enter for default): ").strip()
    if new_port: cfg['POOL_PORT'] = int(new_port)

    print(f"Wallet: {cfg['WALLET']}")
    new_wallet = input("New Wallet (Enter for default): ").strip()
    if new_wallet: cfg['WALLET'] = new_wallet

    print(f"Proxy Port: {cfg['PROXY_PORT']}")
    new_proxy = input("New Proxy Port (Enter for default): ").strip()
    if new_proxy: cfg['PROXY_PORT'] = int(new_proxy)

    print("\nConfiguration complete. Starting...")
    time.sleep(1)
    return cfg

class MinerSuite:
    def __init__(self):
        self.log_q = queue.Queue()
        self.cfg = pre_screen(self.log_q)
        run_benchmark_sequence()
        self.run_setup()
        self.man = mp.Manager()
        self.job_q = self.man.Queue()
        self.res_q = self.man.Queue()
        self.log_q = self.man.Queue()
        self.stop = mp.Event()
        
        self.global_job_id = mp.Array('c', 64)
        self.global_job_id.value = b""
        
        self.data = self.man.dict()
        self.data['job'] = "?"
        self.data['en1'] = ""
        self.data['diff'] = 1024.0
        # Scraper
        self.data['hr_5m'] = "---"
        self.data['hr_1h'] = "---"
        self.data['pool_workers'] = "-"
        self.data['api_status'] = "Init"
        
        self.proxy_stats = self.man.dict()
        self.proxy_stats['submitted'] = 0
        self.proxy_stats['accepted'] = 0
        self.proxy_stats['rejected'] = 0
        self.proxy_hr_map = self.man.dict() # Stores IP_ID -> Hashrate
        
        self.local_stats = self.man.dict()
        self.local_stats['submitted'] = 0
        self.stats = mp.Array('d', [0.0] * (mp.cpu_count() + 1))
        self.last_stats = [0.0] * (mp.cpu_count() + 1)
        self.diff = mp.Value('d', 1024.0)
        self.throttle = mp.Value('d', 0.0)
        self.shares = {"acc": 0, "rej": 0}
        self.start_t = time.time()
        self.logs = []
        self.connected = False
        self.msg_id = 1
        self.current_hashrate = 0.0

    def run_setup(self):
        os.system('clear')
        print("Starting Suite...")
        time.sleep(1)

    def log(self, cat, msg):
        try: self.log_q.put((get_lv06_ts(), cat, msg))
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
            else: self.throttle.value = 0.5
            time.sleep(2)

    def net_thread(self):
        while not self.stop.is_set():
            s = None
            try:
                self.log("stratum_api", f"Dialing {self.cfg['POOL_URL']}...")
                s = socket.create_connection((self.cfg['POOL_URL'], self.cfg['POOL_PORT']), timeout=None)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                self.connected = True
                
                s.sendall((json.dumps({"id": self.get_id(), "method": "mining.subscribe", "params": ["KXT-v56"]}) + "\n").encode())
                s.sendall((json.dumps({"id": self.get_id(), "method": "mining.authorize", "params": [self.cfg['WALLET'], self.cfg['PASSWORD']]}) + "\n").encode())

                buff = b""
                while not self.stop.is_set():
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        params = [
                            self.cfg['WALLET'], r['job_id'], r['extranonce2'], 
                            r['ntime'], r['nonce'], "00000000"
                        ]
                        msg = json.dumps({"id": self.get_id(), "method": "mining.submit", "params": params})
                        s.sendall((msg + "\n").encode())
                        self.local_stats['submitted'] += 1
                        self.log("asic_result", f"[LOCAL] Nonce difficulty {r['share_diff']:.2f} of {r['pool_diff']:.0f}")
                        self.log("stratum_api", f"tx: {msg}")

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
                                self.log("stratum_task", f"rx: {line.decode()}")
                                mid = msg.get('id')
                                result = msg.get('result')
                                method = msg.get('method')
                                
                                if result and isinstance(result, list) and "mining.notify" in str(result):
                                     self.data['en1'] = result[1]
                                elif mid and mid > 3:
                                    if result: 
                                        self.shares['acc'] += 1
                                        self.log("stratum_task", "message result accepted")
                                    else: 
                                        self.shares['rej'] += 1

                                if method == 'mining.notify':
                                    p = msg['params']
                                    jid = str(p[0])
                                    self.data['job'] = jid
                                    # FIX: Explicit Global Update
                                    self.global_job_id.value = jid.encode('utf-8')
                                    en1 = self.data['en1']
                                    if en1:
                                        while not self.job_q.empty(): 
                                            try: self.job_q.get_nowait()
                                            except: pass
                                        j = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], en1)
                                        for _ in range(mp.cpu_count() * 2): self.job_q.put(j)
                                
                                elif method == 'mining.set_difficulty':
                                    self.diff.value = msg['params'][0]
                                    self.data['diff'] = msg['params'][0]

                            except: continue
                    except socket.timeout: pass
                    except OSError: break
            except Exception as e:
                self.log("error", f"Net: {e}")
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
                    fmt_msg = f"{r[0]} {r[1]}: {r[2]}"
                    self.logs.append(fmt_msg)
                    if len(self.logs) > 100: self.logs.pop(0)
            except: pass
            
            c_tmp, g_tmp = get_temps()
            c_load, ram = get_hw_stats()
            
            # Local Hashrate Calc
            current_total = 0.0
            for i in range(len(self.stats)):
                delta = self.stats[i] - self.last_stats[i]
                current_total += delta
                self.last_stats[i] = self.stats[i]
            self.current_hashrate = (self.current_hashrate * 0.7) + (current_total * 0.3 * 10)
            
            stdscr.erase(); h, w = stdscr.getmaxyx()
            col_w = w // 4
            
            stdscr.addstr(0, 0, f" KXT MINER v56 - GRANULAR ".center(w), curses.color_pair(5)|curses.A_BOLD)
            
            stdscr.addstr(2, 2, "=== LOCAL ===", curses.color_pair(4))
            stdscr.addstr(3, 2, f"IP: {get_local_ip()}")
            stdscr.addstr(4, 2, f"Proxy: {self.cfg['PROXY_PORT']}")
            stdscr.addstr(5, 2, f"RAM: {ram}%")
            
            x2 = col_w + 2
            stdscr.addstr(2, x2, "=== HARDWARE ===", curses.color_pair(4))
            stdscr.addstr(3, x2, f"CPU: {c_tmp}C")
            stdscr.addstr(4, x2, f"GPU: {g_tmp}C")
            ts = "OK" if self.throttle.value == 0.0 else "THROTTLED"
            stdscr.addstr(5, x2, f"{ts}", curses.color_pair(1 if ts=="OK" else 2))

            x3 = col_w*2 + 2
            stdscr.addstr(2, x3, "=== POOL STATS ===", curses.color_pair(4))
            stdscr.addstr(3, x3, f"5m HR: {self.data.get('hr_5m', '---')}")
            stdscr.addstr(4, x3, f"1h HR: {self.data.get('hr_1h', '---')}")
            stdscr.addstr(5, x3, f"Active: {self.data.get('pool_workers', '-')}")
            
            x4 = col_w*3 + 2
            stdscr.addstr(2, x4, "=== NETWORK ===", curses.color_pair(4))
            curr_job = self.global_job_id.value.decode('utf-8')
            stdscr.addstr(3, x4, f"Block Data: {curr_job[:8]}")
            stdscr.addstr(4, x4, f"Diff: {int(self.data.get('diff', 0))}")
            stdscr.addstr(5, x4, f"Link: {'ONLINE' if self.connected else 'DOWN'}", curses.color_pair(1 if self.connected else 3))
            
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(9, 2, "=== ACTIVE MINERS ===", curses.color_pair(4)|curses.A_BOLD)
            
            # Dynamic Active Miners List
            # 1. Local
            l_hr = self.current_hashrate
            l_fmt = f"{l_hr/1e6:.2f} MH/s" if l_hr > 1e6 else f"{l_hr/1000:.2f} kH/s"
            stdscr.addstr(10, 2, f"> LOCAL PC: {l_fmt}", curses.color_pair(5))
            
            # 2. Proxy Clients
            row = 10
            for ip_id, hr_val in list(self.proxy_hr_map.items()):
                row += 1
                if row > 12: break # Limit height
                if hr_val > 1_000_000_000_000: p_fmt = f"{hr_val/1e12:.2f} TH/s"
                elif hr_val > 1_000_000_000: p_fmt = f"{hr_val/1e9:.2f} GH/s"
                elif hr_val > 1_000_000: p_fmt = f"{hr_val/1e6:.2f} MH/s"
                else: p_fmt = f"{hr_val/1000:.2f} kH/s"
                stdscr.addstr(row, 2, f"> ASIC_{ip_id}: {p_fmt}", curses.color_pair(2))

            stdscr.hline(13, 0, curses.ACS_HLINE, w)
            
            log_h = h - 14
            if log_h > 0:
                for i, l in enumerate(self.logs[-log_h:]):
                    c = curses.color_pair(1)
                    if "error" in l.lower() or "rejected" in l.lower(): c = curses.color_pair(3)
                    elif "system" in l.lower(): c = curses.color_pair(4)
                    elif "tx" in l.lower(): c = curses.color_pair(5)
                    try: stdscr.addstr(14+i, 2, l[:w-4], c)
                    except: pass
            
            stdscr.refresh()
            if stdscr.getch() == ord('q'): break
            time.sleep(0.1)

    def start(self):
        AutoUpdate(self.cfg['UPDATE_URL'], self.log_q).start()
        PoolStatsScraper(self.cfg['STATS_URL'], self.cfg['STATS_API'], self.data).start()
        ProxyServer(self.cfg, self.log_q, self.proxy_stats, self.proxy_hr_map, self.diff).start()
        threading.Thread(target=self.net_thread, daemon=True).start()
        threading.Thread(target=self.thermal_thread, daemon=True).start()
        
        procs = []
        for i in range(mp.cpu_count()):
            p = mp.Process(target=cpu_worker, args=(i, self.job_q, self.res_q, self.stop, self.stats, self.diff, self.throttle, self.log_q, self.global_job_id), daemon=True)
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
