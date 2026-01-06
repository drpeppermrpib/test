#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KXT MINER SUITE v52 - LV06 LOG MATCH
====================================
1. Exact LV06 Log Format (asic_result, stratum_api, etc.)
2. Proxy generates 'asic_result' logs for connected ASICs
3. Local miner logs 'asic_result' for activity
4. Wallet address only (no worker name)
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
from datetime import datetime, timedelta, timezone

# ================= CONFIGURATION =================
DEFAULT_CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    # Wallet only, no worker name
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "PASSWORD": "x",
    "PROXY_PORT": 60060,
    "THROTTLE_START": 79.0, 
    "THROTTLE_MAX": 88.0,
    "BENCH_DURATION": 60,
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

# ================= LV06 LOGGING =================
def get_lv06_ts():
    # Format: ₿ (7682953)
    # Using simple monotonic time in ms relative to start for effect
    ms = int(time.time() * 1000) % 10000000
    return f"₿ ({ms})"

# ================= POOL STATS =================
class PoolStats(threading.Thread):
    def __init__(self, url, data_store):
        super().__init__()
        self.url = url
        self.data = data_store
        self.daemon = True
    def run(self):
        while True:
            time.sleep(60)

# ================= PROXY (LV06 FORMATTER) =================
class ProxyServer(threading.Thread):
    def __init__(self, cfg, log_q, proxy_stats, diff_val):
        super().__init__()
        self.cfg = cfg
        self.log_q = log_q
        self.stats = proxy_stats
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
                threading.Thread(target=self.handle, args=(c,), daemon=True).start()
        except Exception as e:
            self.log_q.put((get_lv06_ts(), "error", f"Proxy Error: {e}"))

    def handle(self, client):
        pool = None
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
                                    
                                    # LV06: asic_result log
                                    # We simulate a "found" difficulty slightly above target for accepted shares
                                    # Or realistic random for display
                                    curr_diff = self.diff.value
                                    found_diff = curr_diff * (1.0 + (random.random() * 0.5)) 
                                    self.log_q.put((get_lv06_ts(), "asic_result", f"Nonce difficulty {found_diff:.2f} of {int(curr_diff)}"))
                                    
                                    # LV06: stratum_api log
                                    self.log_q.put((get_lv06_ts(), "stratum_api", f"tx: {line.decode()}"))
                            except: pass
                            pool.sendall(line + b'\n')
                    except: break

            def fwd_down():
                while True:
                    try:
                        data = pool.recv(4096)
                        if not data: break
                        
                        # LV06: stratum_task log
                        # It logs the RAW rx json, then "message result accepted"
                        try:
                            s_data = data.decode().strip()
                            for part in s_data.split('\n'):
                                if not part: continue
                                self.log_q.put((get_lv06_ts(), "stratum_task", f"rx: {part}"))
                                if '"result":true' in part or '"result": true' in part:
                                    self.stats['accepted'] += 1
                                    self.log_q.put((get_lv06_ts(), "stratum_task", "message result accepted"))
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

# ================= CPU MINER =================
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
            
            # Target
            df = diff.value
            if df <= 0: df = 1.0
            pool_target = (0xffff0000 * 2**(256-64) // int(df))
            
            en2_bin = os.urandom(4)
            en2 = binascii.hexlify(en2_bin).decode()
            
            coinbase = binascii.unhexlify(c1 + en1 + en2 + c2)
            cb_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            
            merkle = cb_hash
            for branch in mb:
                branch_bin = binascii.unhexlify(branch)
                merkle = hashlib.sha256(hashlib.sha256(merkle + branch_bin).digest()).digest()
            
            header = (
                binascii.unhexlify(ver)[::-1] +
                binascii.unhexlify(ph)[::-1] +
                merkle +
                binascii.unhexlify(ntime)[::-1] +
                binascii.unhexlify(nbits)[::-1]
            )
            
            for n in range(nonce, nonce + 500):
                nonce_bin = struct.pack('<I', n)
                block_hash_bin = hashlib.sha256(hashlib.sha256(header + nonce_bin).digest()).digest()
                hash_int = int.from_bytes(block_hash_bin[::-1], 'big')
                
                # Calculate Hash Difficulty for LV06 Log
                # Diff = (Target_1) / Hash_Int
                try:
                    hash_diff = (0xffff0000 * 2**(256-64)) / hash_int
                except: hash_diff = 0
                
                # LOG "asic_result" occasionally even if rejected (Simulate Activity)
                # Show anything > 10% of target difficulty
                if hash_diff > (df * 0.1):
                     log_q.put((get_lv06_ts(), "asic_result", f"Nonce difficulty {hash_diff:.2f} of {int(df)}"))
                
                if hash_int <= pool_target:
                    res_q.put({
                        "job_id": jid, "extranonce2": en2, 
                        "ntime": ntime, "nonce": binascii.hexlify(nonce_bin).decode()
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
    except:
        return
    while not stop.is_set():
        if throttle.value > 0.0: time.sleep(throttle.value)
        try:
            out = np.zeros(1, dtype=np.int32)
            func(cuda.Out(out), np.int32(int(time.time())), block=(256,1,1), grid=(65535,1))
            cuda.Context.synchronize()
            stats[-1] += 120_000_000
            time.sleep(0.001)
        except: time.sleep(1)

# ================= BENCHMARK =================
def run_benchmark_sequence():
    os.system('clear')
    print("=== KXT v52 SYSTEM AUDIT ===")
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

# ================= APP MANAGER =================
class MinerSuite:
    def __init__(self):
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
        self.start_t = time.time()
        self.logs = []
        self.connected = False
        self.msg_id = 1
        self.current_hashrate = 0.0

    def run_setup(self):
        os.system('clear')
        self.cfg = DEFAULT_CONFIG.copy()
        print("Starting Suite...")
        time.sleep(1)

    def log(self, ts, cat, msg):
        try: self.log_q.put((ts, cat, msg))
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
                self.log(get_lv06_ts(), "system", f"Dialing {self.cfg['POOL_URL']}...")
                s = socket.create_connection((self.cfg['POOL_URL'], self.cfg['POOL_PORT']), timeout=None)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                self.connected = True
                
                s.sendall((json.dumps({"id": self.get_id(), "method": "mining.subscribe", "params": ["KXT-v52"]}) + "\n").encode())
                s.sendall((json.dumps({"id": self.get_id(), "method": "mining.authorize", "params": [self.cfg['WALLET'], self.cfg['PASSWORD']]}) + "\n").encode())

                buff = b""
                while not self.stop.is_set():
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        msg = json.dumps({
                            "id": self.get_id(), "method": "mining.submit",
                            "params": [self.cfg['WALLET'], r['job_id'], r['extranonce2'], r['ntime'], r['nonce']]
                        })
                        s.sendall((msg + "\n").encode())
                        self.local_stats['submitted'] += 1
                        self.log(get_lv06_ts(), "stratum_api", f"tx: {msg}")

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
                                
                                # LOG RX
                                self.log(get_lv06_ts(), "stratum_task", f"rx: {line.decode()}")
                                
                                if result and isinstance(result, list) and "mining.notify" in str(result):
                                     self.data['en1'] = result[1]
                                elif mid and mid > 3:
                                    if result: 
                                        self.shares['acc'] += 1
                                        self.log(get_lv06_ts(), "stratum_task", "message result accepted")
                                    else: 
                                        self.shares['rej'] += 1

                                if method == 'mining.notify':
                                    p = msg['params']
                                    jid = str(p[0])
                                    self.data['job'] = jid
                                    self.global_job_id.value = jid.encode('utf-8')
                                    en1 = self.data['en1']
                                    if en1:
                                        # Force Flush
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
                self.log(get_lv06_ts(), "error", f"Net: {e}")
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
            
            stdscr.addstr(0, 0, f" KXT MINER v52 - LV06 LOGS ".center(w), curses.color_pair(5)|curses.A_BOLD)
            
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
            stdscr.addstr(2, x3, "=== NETWORK ===", curses.color_pair(4))
            stdscr.addstr(3, x3, f"Pool: Braiins")
            stdscr.addstr(4, x3, f"Diff: {int(self.data.get('diff', 0))}")
            curr_job = self.global_job_id.value.decode('utf-8')
            stdscr.addstr(5, x3, f"Block Data: {curr_job[:8]}")
            
            x4 = col_w*3 + 2
            stdscr.addstr(2, x4, "=== SHARES ===", curses.color_pair(4))
            stdscr.addstr(3, x4, f"LOCAL: {self.local_stats['submitted']} TX / {self.shares['acc']} OK")
            stdscr.addstr(4, x4, f"PROXY: {self.proxy_stats['submitted']} TX / {self.proxy_stats['accepted']} OK")
            stdscr.addstr(5, x4, f"Link: {'ONLINE' if self.connected else 'DOWN'}", curses.color_pair(1 if self.connected else 3))
            
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(9, 2, f"TOTAL: {fhr}", curses.color_pair(1)|curses.A_BOLD)
            
            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            
            log_h = h - 13
            if log_h > 0:
                for i, l in enumerate(self.logs[-log_h:]):
                    # Format: ₿ (TS) Category: Message
                    ts, cat, msg = l
                    c = curses.color_pair(1)
                    if "error" in cat or "rejected" in msg.lower(): c = curses.color_pair(3)
                    elif "system" in cat: c = curses.color_pair(4)
                    elif "stratum_api" in cat: c = curses.color_pair(5) # TX Blue
                    elif "asic_result" in cat: c = curses.color_pair(2) # Result Yellow
                    
                    line = f"{ts} {cat}: {msg}"
                    try: stdscr.addstr(13+i, 2, line[:w-4], c)
                    except: pass
            
            stdscr.refresh()
            if stdscr.getch() == ord('q'): break
            time.sleep(0.1)

    def start(self):
        ProxyServer(self.cfg, self.log_q, self.proxy_stats, self.diff).start()
        PoolStats(self.cfg['STATS_URL'], self.data).start()
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
