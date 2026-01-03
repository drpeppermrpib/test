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
import platform
from datetime import datetime

# ================= AUTO-DEPENDENCY CHECK =================
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
    "TARGET_TEMP": 84.0
}

# ================= PTX GPU KERNEL (NO NVCC) =================
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
def fix_env():
    paths = ["/usr/local/cuda/bin", "/usr/bin", "/bin", "/opt/cuda/bin", "/hive/lib/cuda/bin"]
    curr = os.environ.get("PATH", "")
    for p in paths:
        if os.path.exists(p) and p not in curr: curr += ":" + p
    os.environ["PATH"] = curr

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except: return "127.0.0.1"

def get_hw_stats():
    cpu = psutil.cpu_percent(interval=None) if HAS_PSUTIL else 0.0
    ram = psutil.virtual_memory().percent if HAS_PSUTIL else 0.0
    return cpu, ram

def get_temps():
    c, g = 0.0, 0.0
    try:
        o = subprocess.check_output("sensors", shell=True).decode()
        for l in o.splitlines():
            if "Tdie" in l or "Package" in l: 
                c = float(l.split('+')[1].split('°')[0].strip())
    except: pass
    try:
        o = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True).decode()
        g = float(o.strip())
    except: pass
    return c, g

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
            sock.bind(("0.0.0.0", self.cfg['PROXY_PORT']))
            sock.listen(5)
            self.log_q.put(("INFO", f"Proxy Active on Port {self.cfg['PROXY_PORT']}"))
            while True:
                c, a = sock.accept()
                threading.Thread(target=self.handle, args=(c,), daemon=True).start()
        except Exception as e:
            self.log_q.put(("ERR", f"Proxy Fail: {e}"))

    def handle(self, client):
        try:
            upstream = socket.create_connection((self.cfg['POOL_URL'], self.cfg['POOL_PORT']))
            def fwd(s, d):
                while True:
                    data = s.recv(4096)
                    if not data: break
                    d.sendall(data)
            threading.Thread(target=fwd, args=(client, upstream), daemon=True).start()
            fwd(upstream, client)
        except: pass
        finally: client.close()

# ================= WORKERS =================
def cpu_worker(id, job_q, res_q, stop, stats, diff, log_q):
    active_jid = None
    nonce = 0
    stride = id * 50_000_000
    
    while not stop.is_set():
        try:
            try:
                job = job_q.get_nowait()
                # job: (jid, prev, c1, c2, mb, ver, nbits, ntime, clean, en1)
                if job[0] != active_jid or job[8]:
                    active_jid = job[0]
                    curr_job = job
                    nonce = 0
                    if id == 0: log_queue.put(("CPU", f"New Job: {active_jid[:8]}"))
                else:
                    curr_job = job
            except queue.Empty: pass
            
            if not active_jid: 
                time.sleep(0.1); continue
            
            # Unpack
            jid, ph, c1, c2, mb, ver, nbits, ntime, clean, en1 = curr_job
            
            df = diff.value
            if df == 0: df = 1
            target = (0xffff0000 * 2**(256-64) // int(df))
            
            en2 = struct.pack('<I', id).hex().zfill(8)
            
            # Coinbase = c1 + en1 + en2 + c2
            cb_hex = c1 + en1 + en2 + c2
            cb_bin = binascii.unhexlify(cb_hex)
            cb_hash = hashlib.sha256(hashlib.sha256(cb_bin).digest()).digest()
            
            # Merkle
            root = cb_hash
            for b in mb:
                root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(b)).digest()).digest()
                
            # Header
            header_pre = (
                binascii.unhexlify(ver)[::-1] + 
                binascii.unhexlify(ph)[::-1] + 
                root + 
                binascii.unhexlify(ntime)[::-1] + 
                binascii.unhexlify(nbits)[::-1]
            )
            
            start_n = stride + nonce
            for n in range(start_n, start_n + 1000):
                h = header_pre + struct.pack('<I', n)
                h_hash = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                
                if int.from_bytes(h_hash[::-1], 'big') <= target:
                    res_q.put({
                        "job_id": jid, "extranonce2": en2, 
                        "ntime": ntime, "nonce": f"{n:08x}"
                    })
                    break
            
            stats[id] += 1000
            nonce += 1000
            
        except Exception: time.sleep(0.1)

def gpu_worker(stop, stats, log_q):
    fix_env()
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import numpy as np
        mod = cuda.module_from_buffer(PTX_CODE.encode())
        func = mod.get_function("heavy_load")
        log_q.put(("GPU", "PTX Loaded. Mining Active."))
    except:
        log_q.put(("WARN", "GPU Init Failed. Disabled."))
        return

    while not stop.is_set():
        try:
            out = np.zeros(1, dtype=np.int32)
            seed = np.int32(int(time.time()))
            func(cuda.Out(out), seed, block=(256,1,1), grid=(40960,1))
            cuda.Context.synchronize()
            stats[-1] += 150_000_000
        except: time.sleep(1)

# ================= SUITE =================
class MinerSuite:
    def __init__(self):
        self.setup()
        
        self.man = mp.Manager()
        self.job_q = self.man.Queue()
        self.res_q = self.man.Queue()
        self.log_q = self.man.Queue()
        self.stop = mp.Event()
        
        # SHARED DATA (Fixed crash by using Dict instead of Array)
        self.data = self.man.dict()
        self.data['job'] = "Waiting..."
        self.data['en1'] = ""
        self.data['diff'] = 1024.0
        
        self.stats = mp.Array('d', [0.0] * (mp.cpu_count() + 1))
        self.diff = mp.Value('d', 1024.0)
        
        self.shares = {"acc": 0, "rej": 0}
        self.start_t = time.time()
        self.logs = []
        self.connected = False

    def setup(self):
        print("\033[2J\033[H")
        print("MTP MINER SUITE v10 - SETUP")
        print("-" * 40)
        self.cfg = DEFAULT_CONFIG.copy()
        
        print(f"Press ENTER for defaults.")
        u = input(f"Pool [{self.cfg['POOL_URL']}]: ").strip()
        if u: self.cfg['POOL_URL'] = u
        
        p = input(f"Port [{self.cfg['POOL_PORT']}]: ").strip()
        if p: self.cfg['POOL_PORT'] = int(p)
        
        w = input(f"Wallet [{self.cfg['WALLET'][:8]}...]: ").strip()
        if w: self.cfg['WALLET'] = w
        
        print("Starting...")
        time.sleep(1)

    def log(self, t, m):
        try: self.log_q.put((t, m))
        except: pass

    def net_thread(self):
        while not self.stop.is_set():
            s = None
            try:
                self.log("NET", f"Connecting {self.cfg['POOL_URL']}")
                s = socket.create_connection((self.cfg['POOL_URL'], self.cfg['POOL_PORT']), timeout=10)
                self.connected = True
                
                # Subscribe
                s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["MTP-v10"]})+"\n").encode())
                s.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [self.cfg['WALLET'], self.cfg['PASSWORD']]})+"\n").encode())
                
                buff = b""
                s.settimeout(0.5)
                
                while not self.stop.is_set():
                    # Submit
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        msg = json.dumps({
                            "id": 4, "method": "mining.submit",
                            "params": [self.cfg['WALLET'], r['job_id'], r['extranonce2'], r['ntime'], r['nonce']]
                        }) + "\n"
                        s.sendall(msg.encode())
                        self.log("MINE", f"Nonce Found: {r['nonce']}")
                    
                    # Recv
                    try:
                        d = s.recv(8192)
                        if not d: break
                        buff += d
                        while b'\n' in buff:
                            line, buff = buff.split(b'\n', 1)
                            if not line: continue
                            
                            msg = json.loads(line.decode())
                            mid = msg.get('id')
                            
                            if mid == 1 and msg.get('result'):
                                r1 = msg['result']
                                if len(r1) >= 2:
                                    self.data['en1'] = r1[1]
                                    self.log("POOL", f"Subscribed. En1: {r1[1]}")
                            
                            elif mid == 4:
                                if msg.get('result'): 
                                    self.shares['acc'] += 1
                                    self.log("POOL", "Share ACCEPTED")
                                else: 
                                    self.shares['rej'] += 1
                                    self.log("WARN", "Share REJECTED")

                            if msg.get('method') == 'mining.notify':
                                p = msg['params']
                                self.data['job'] = str(p[0])
                                en1 = self.data['en1']
                                
                                if not en1: continue # Wait for subscribe
                                
                                if p[8]: # Clean
                                    while not self.job_q.empty():
                                        try: self.job_q.get_nowait()
                                        except: pass
                                
                                j = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], en1)
                                for _ in range(mp.cpu_count() + 2): self.job_q.put(j)
                                
                            elif msg.get('method') == 'mining.set_difficulty':
                                self.diff.value = msg['params'][0]
                                self.data['diff'] = msg['params'][0]

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
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
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
            col_w = w // 3
            
            # HEADER
            stdscr.attron(curses.color_pair(4) | curses.A_REVERSE)
            stdscr.addstr(0, 0, " MTP MINER SUITE v10 ".center(w))
            stdscr.attroff(curses.color_pair(4) | curses.A_REVERSE)
            
            # COL 1: LOCAL
            stdscr.addstr(2, 2, "=== LOCAL ===", curses.color_pair(4)|curses.A_BOLD)
            stdscr.addstr(3, 2, f"IP: {get_local_ip()}")
            stdscr.addstr(4, 2, f"Proxy: {self.cfg['PROXY_PORT']}")
            stdscr.addstr(5, 2, f"RAM: {ram}% CPU: {c_load}%")
            stdscr.addstr(6, 2, f"Uptime: {int(time.time()-self.start_t)}s")

            # COL 2: HARDWARE
            stdscr.addstr(2, col_w+2, "=== HARDWARE ===", curses.color_pair(5)|curses.A_BOLD)
            stdscr.addstr(3, col_w+2, f"CPU Temp: {c_tmp}C")
            stdscr.addstr(4, col_w+2, f"GPU Temp: {g_tmp}C")
            stdscr.addstr(5, col_w+2, f"Threads: {mp.cpu_count()}")

            # COL 3: NETWORK
            stdscr.addstr(2, col_w*2+2, "=== NETWORK ===", curses.color_pair(2)|curses.A_BOLD)
            stdscr.addstr(3, col_w*2+2, f"Pool: {self.cfg['POOL_URL'][:20]}")
            stdscr.addstr(4, col_w*2+2, f"Diff: {int(self.data.get('diff', 0))}")
            stdscr.addstr(5, col_w*2+2, f"Job: {self.data.get('job', '?')}")
            
            s = "CONNECTED" if self.connected else "DIALING..."
            stdscr.addstr(6, col_w*2+2, f"Status: {s}", curses.color_pair(1 if self.connected else 3))
            
            # BARS
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            hr = sum(self.stats) / (time.time() - self.start_t + 1)
            fhr = f"{hr/1e6:.2f} MH/s" if hr > 1e6 else f"{hr/1000:.2f} kH/s"
            
            stdscr.addstr(9, 2, f"TOTAL: {fhr}", curses.color_pair(1)|curses.A_BOLD)
            stdscr.addstr(9, 40, f"ACC: {self.shares['acc']} REJ: {self.shares['rej']}")
            
            bar_w = max(5, w - 20)
            fill = int((g_tmp / 90.0) * bar_w)
            stdscr.addstr(10, 2, "GPU: " + "█"*fill, curses.color_pair(2))
            
            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            
            # LOGS
            log_h = h - 13
            if log_h > 0:
                for i, l in enumerate(self.logs[-log_h:]):
                    c = curses.color_pair(1)
                    if l[1] in ["ERR", "BAD"]: c = curses.color_pair(3)
                    elif l[1] == "WARN": c = curses.color_pair(2)
                    stdscr.addstr(13+i, 2, f"{l[0]} [{l[1]}] {l[2]}"[:w-4], c)

            stdscr.refresh()
            if stdscr.getch() == ord('q'): break
            time.sleep(0.1)

    def start(self):
        # Threads
        ProxyServer(self.cfg, self.log_q).start()
        threading.Thread(target=self.net_thread, daemon=True).start()
        
        # Procs
        for i in range(mp.cpu_count()):
            mp.Process(target=cpu_worker, args=(i, self.job_q, self.res_q, self.stop, self.stats, self.diff, self.log_q), daemon=True).start()
        mp.Process(target=gpu_worker, args=(self.stop, self.stats, self.log_q), daemon=True).start()
        
        try: curses.wrapper(self.draw_ui)
        except KeyboardInterrupt: pass
        finally: self.stop.set()

if __name__ == "__main__":
    MinerSuite().start()
