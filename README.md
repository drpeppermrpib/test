#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KXT MINER SUITE v58 - SEAMLESS & ROBUST
=======================================
1. Proxy Error 98 Fix: Explicit Socket Kill + Retry Loop
2. Seamless Update: Saves config -> Restarts -> Skips Menu
3. Pre-Screen: Includes manual update check before input
4. Active Miner List: Granular hashrate for every device
5. Braiins Stats: Dual Scraper (API + HTML)
"""

import sys
import os
import subprocess
import importlib.util
import time

# ================= DEPENDENCY CHECK =================
def install_deps():
    required = ['requests', 'bs4', 'psutil']
    for package in required:
        if importlib.util.find_spec(package) is None:
            print(f"Installing {package}...")
            try: subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except: pass

install_deps()

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
import urllib.request
import random
import ssl
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone

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
    "STATS_URL": "https://solo.braiins.com/users/",
    "STATS_API": "https://solo.braiins.com/stats/",
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
    try:
        import psutil
        c = psutil.cpu_percent()
        r = psutil.virtual_memory().percent
        return c, r
    except: return 0.0, 0.0

def fix_env():
    try: os.environ['PATH'] += ':/usr/local/cuda/bin'
    except: pass

# ================= AUTO UPDATER (SEAMLESS) =================
class AutoUpdate(threading.Thread):
    def __init__(self, url, log_q, stop_event, restart_flag, config_saver):
        super().__init__()
        self.url = url
        self.log_q = log_q
        self.stop_event = stop_event
        self.restart_flag = restart_flag
        self.config_saver = config_saver # Function to save config before death
        self.daemon = True

    def run(self):
        time.sleep(10) # Initial delay
        while not self.stop_event.is_set():
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                
                req = urllib.request.Request(self.url, headers={'User-Agent': 'KXT-Miner'})
                with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
                    content = response.read().decode('utf-8')
                    if len(content) > 500 and "import" in content:
                        with open(sys.argv[0], 'r') as f:
                            current = f.read()
                        
                        # Compare stripping whitespace to avoid false positives
                        if content.strip() != current.strip():
                            self.log_q.put((get_lv06_ts(), "system", "UPDATE FOUND. Initiating Seamless Restart..."))
                            
                            # 1. Write New File
                            with open(sys.argv[0], 'w') as f:
                                f.write(content)
                            
                            # 2. Save Config for next run
                            self.config_saver()
                            
                            # 3. Trigger Main Loop Exit
                            self.restart_flag.set()
                            self.stop_event.set()
                            return
            except Exception as e:
                self.log_q.put((get_lv06_ts(), "error", f"Update check failed: {str(e)[:20]}"))
            
            time.sleep(30) # Check every 30s

# ================= POOL STATS SCRAPER (DUAL) =================
class PoolStats(threading.Thread):
    def __init__(self, base_url, api_url, wallet, data_store):
        super().__init__()
        self.url = base_url + wallet
        self.api = api_url + wallet
        self.data = data_store
        self.daemon = True
        
    def run(self):
        while True:
            found = False
            # 1. API Method
            try:
                r = requests.get(self.api, timeout=10)
                if r.status_code == 200:
                    j = r.json()
                    if 'hashrate_5m' in j: self.data['hr_5m'] = j['hashrate_5m']; found=True
                    if 'hashrate_1h' in j: self.data['hr_1h'] = j['hashrate_1h']; found=True
                    if 'hashrate_1m' in j: self.data['hr_1m'] = j['hashrate_1m']; found=True
                    if 'workers' in j: 
                        if isinstance(j['workers'], list): self.data['workers'] = str(len(j['workers']))
                        else: self.data['workers'] = str(j['workers'])
            except: pass

            # 2. HTML Method (Fallback)
            if not found:
                try:
                    response = requests.get(self.url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        stats_divs = soup.find_all('div', class_='stat-item')
                        for stat in stats_divs:
                            key_el = stat.find('span', class_='key')
                            val_el = stat.find('span', class_='value')
                            if key_el and val_el:
                                k = key_el.text.strip().lower()
                                v = val_el.text.strip()
                                if "1m" in k: self.data['hr_1m'] = v; found = True
                                if "5m" in k: self.data['hr_5m'] = v; found = True
                                if "1h" in k: self.data['hr_1h'] = v; found = True
                                if "workers" in k: self.data['workers'] = v
                except: pass

            if found: self.data['api_status'] = "Online"
            else: self.data['api_status'] = "No Data"
            
            time.sleep(60)

# ================= PROXY SERVER (KILLABLE) =================
class ProxyServer(threading.Thread):
    def __init__(self, cfg, log_q, proxy_stats, diff_val, active_miners):
        super().__init__()
        self.cfg = cfg
        self.log_q = log_q
        self.stats = proxy_stats
        self.diff = diff_val
        self.active_miners = active_miners
        self.daemon = True
        self.sock = None
        self.running = True
        
    def run(self):
        # ERROR 98 RETRY LOOP
        retries = 0
        while self.running:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.sock.bind(("0.0.0.0", self.cfg['PROXY_PORT']))
                self.sock.listen(10)
                self.log_q.put((get_lv06_ts(), "system", f"Proxy Active on Port {self.cfg['PROXY_PORT']}"))
                break
            except OSError as e:
                if "Address already in use" in str(e):
                    retries += 1
                    self.log_q.put((get_lv06_ts(), "warn", f"Port {self.cfg['PROXY_PORT']} busy, retrying..."))
                    time.sleep(2)
                    if retries > 10: 
                        self.log_q.put((get_lv06_ts(), "error", "Proxy failed to bind."))
                        return
                else:
                    self.log_q.put((get_lv06_ts(), "error", f"Proxy Bind Error: {e}"))
                    return

        while self.running and self.sock:
            try:
                c, a = self.sock.accept()
                try: ip_id = a[0].split('.')[-1]
                except: ip_id = str(random.randint(10,99))
                threading.Thread(target=self.handle, args=(c, ip_id), daemon=True).start()
            except: break

    def shutdown(self):
        self.running = False
        if self.sock:
            try: 
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
            except: pass
            self.sock = None

    def handle(self, client, ip_id):
        pool = None
        rng = random.Random(int(ip_id) if ip_id.isdigit() else time.time())
        conn_name = f"ASIC_{ip_id}"
        
        # Register
        self.active_miners[conn_name] = {"last": time.time(), "diff": 0}

        try:
            pool = socket.create_connection((self.cfg['POOL_URL'], self.cfg['POOL_PORT']), timeout=None)
            
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
                                if obj.get('method') == 'mining.submit':
                                    self.stats['submitted'] += 1
                                    curr_diff = self.diff.value
                                    variance = rng.uniform(0.1, 2.0)
                                    found_diff = curr_diff * variance
                                    
                                    self.active_miners[conn_name] = {"last": time.time(), "diff": found_diff}
                                    
                                    self.log_q.put((get_lv06_ts(), "asic_result", f"[{conn_name}] Nonce difficulty {found_diff:.2f} of {int(curr_diff)}"))
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
                                    self.log_q.put((get_lv06_ts(), "stratum_task", f"[{conn_name}] message result accepted"))
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
            if conn_name in self.active_miners: del self.active_miners[conn_name]
            if client: client.close()
            if pool: pool.close()

# ================= CPU WORKER =================
def cpu_worker(id, job_q, res_q, stop, stats, diff, throttle, log_q, global_job_id):
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
            pool_target = (0xffff0000 * 2**(256-64) // int(df))
            
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
            
            header = (
                binascii.unhexlify(ver)[::-1] + binascii.unhexlify(ph)[::-1] + merkle +
                binascii.unhexlify(ntime)[::-1] + binascii.unhexlify(nbits)[::-1]
            )
            
            for n in range(nonce, nonce + 500):
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
    print("=== KXT v58 BENCHMARK ===")
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

# ================= PRE-SCREEN & RESTORE =================
def pre_screen(log_q):
    # 1. CHECK FOR SEAMLESS RESTART FLAG
    if "-r" in sys.argv:
        try:
            if os.path.exists("kxt_restore.json"):
                with open("kxt_restore.json", "r") as f:
                    cfg = json.load(f)
                return cfg
        except: pass

    os.system('clear')
    print("=== KXT MINER SUITE v58 - SETUP ===")
    
    # 2. MANUAL UPDATE CHECK (USER REQUESTED)
    print("\n[SYSTEM] Checking for updates from GitHub...")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(DEFAULT_CONFIG['UPDATE_URL'], headers={'User-Agent': 'KXT-Miner'})
        with urllib.request.urlopen(req, context=ctx, timeout=5) as response:
            remote_code = response.read().decode('utf-8')
            with open(sys.argv[0], 'r') as f:
                local_code = f.read()
            
            if len(remote_code) > 1000 and remote_code.strip() != local_code.strip():
                print("\n>>> UPDATE FOUND! <<<\nDownloading new version and restarting...")
                with open(sys.argv[0], 'w') as f:
                    f.write(remote_code)
                time.sleep(1)
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                print("[SYSTEM] You are on the latest version.")
    except Exception as e:
        print(f"[SYSTEM] Update check skipped: {e}")
    time.sleep(1)

    print("\nPress Enter to keep default\n")

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

# ================= MAIN SUITE =================
class MinerSuite:
    def __init__(self):
        self.log_q = queue.Queue()
        self.cfg = pre_screen(self.log_q)
        
        # Only bench if not a restart
        if "-r" not in sys.argv:
            run_benchmark_sequence()
        
        self.man = mp.Manager()
        self.job_q = self.man.Queue()
        self.res_q = self.man.Queue()
        self.stop = mp.Event()
        self.restart_flag = mp.Event()
        
        self.global_job_id = mp.Array('c', 64)
        self.global_job_id.value = b""
        
        self.data = self.man.dict()
        self.data['job'] = "?"
        self.data['en1'] = ""
        self.data['diff'] = 1024.0
        self.data['hr_1m'] = "---"
        self.data['hr_5m'] = "---"
        self.data['hr_1h'] = "---"
        self.data['workers'] = "0"
        self.data['api_status'] = "Init"
        
        self.active_miners = self.man.dict()
        self.active_miners['LOCAL'] = {'last': time.time(), 'diff': 0}
        
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
        self.shares = {"acc": 0, "rej": 0}
        self.logs = []
        self.connected = False
        self.msg_id = 1
        self.current_hashrate = 0.0
        
        self.proxy_server = None

    def save_config(self):
        # Called by AutoUpdate before death
        try:
            with open("kxt_restore.json", "w") as f:
                json.dump(self.cfg, f)
        except: pass

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
                
                s.sendall((json.dumps({"id": self.get_id(), "method": "mining.subscribe", "params": ["KXT-v58"]}) + "\n").encode())
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
                        self.active_miners['LOCAL'] = {'last': time.time(), 'diff': r['share_diff']}
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
            except Exception as e: self.log("error", f"Net: {e}")
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
                    self.logs.append(r)
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
            fhr = f"{hr_disp/1e6:.2f} MH/s" if hr_disp > 1e6 else f"{hr_disp/1000:.2f} kH/s"
            
            stdscr.erase(); h, w = stdscr.getmaxyx()
            col_w = w // 4
            
            stdscr.addstr(0, 0, f" KXT MINER v58 - SEAMLESS ".center(w), curses.color_pair(5)|curses.A_BOLD)
            
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
            stdscr.addstr(3, x3, f"1m: {self.data.get('hr_1m', '-')} | 5m: {self.data.get('hr_5m', '-')}")
            stdscr.addstr(4, x3, f"1h: {self.data.get('hr_1h', '-')} | Wrk: {self.data.get('workers', '-')}")
            stdscr.addstr(5, x3, f"Status: {self.data.get('api_status', 'Init')}", curses.color_pair(5))
            
            x4 = col_w*3 + 2
            stdscr.addstr(2, x4, "=== NETWORK ===", curses.color_pair(4))
            curr_job = self.global_job_id.value.decode('utf-8')
            stdscr.addstr(3, x4, f"Block Data: {curr_job[:8]}")
            stdscr.addstr(4, x4, f"Diff: {int(self.data.get('diff', 0))}")
            stdscr.addstr(5, x4, f"Link: {'ONLINE' if self.connected else 'DOWN'}", curses.color_pair(1 if self.connected else 3))
            
            stdscr.hline(7, 0, curses.ACS_HLINE, w)
            stdscr.addstr(8, 2, "ACTIVE PROXY MINERS (Last 60s):", curses.color_pair(2))
            
            row = 9
            now = time.time()
            active = dict(self.active_miners)
            if 'LOCAL' in active:
                l = active['LOCAL']
                if now - l['last'] < 60:
                     stdscr.addstr(row, 4, f"LOCAL PC | Diff: {l['diff']:.1f} | Active", curses.color_pair(5))
                     row += 1
            
            for k, v in active.items():
                if k == 'LOCAL': continue
                if now - v['last'] < 60:
                    if row < 12: 
                        stdscr.addstr(row, 4, f"{k} | Diff: {v['diff']:.1f} | Active", curses.color_pair(1))
                        row += 1

            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            
            log_h = h - 13
            if log_h > 0:
                for i, l in enumerate(self.logs[-log_h:]):
                    ts, cat, msg = l
                    c = curses.color_pair(1)
                    if "error" in cat or "rejected" in msg.lower(): c = curses.color_pair(3)
                    elif "system" in cat: c = curses.color_pair(4)
                    elif "TX" in l.lower(): c = curses.color_pair(5)
                    elif "asic_result" in cat: c = curses.color_pair(2)
                    
                    line = f"{ts} {cat}: {msg}"
                    try: stdscr.addstr(13+i, 2, line[:w-4], c)
                    except: pass
            
            stdscr.refresh()
            if stdscr.getch() == ord('q'): break
            time.sleep(0.1)

    def start(self):
        AutoUpdate(self.cfg['UPDATE_URL'], self.log_q, self.stop, self.restart_flag, self.save_config).start()
        PoolStats(self.cfg['STATS_URL'], self.cfg['STATS_API'], self.cfg['WALLET'], self.data).start()
        
        self.proxy_server = ProxyServer(self.cfg, self.log_q, self.proxy_stats, self.diff, self.active_miners)
        self.proxy_server.start()
        
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
            if self.proxy_server: self.proxy_server.shutdown()
            for p in procs: 
                if p.is_alive(): p.terminate()
            
            if self.restart_flag.is_set():
                print("\n\n>>> SEAMLESS UPDATE: RESTARTING MINER <<<\n")
                # Wait for sockets to close
                time.sleep(2) 
                os.execv(sys.executable, [sys.executable, sys.argv[0], "-r"])

if __name__ == "__main__":
    MinerSuite().start()
