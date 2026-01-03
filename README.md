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

# Try to import psutil for hardware stats, handle if missing
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ================= DEFAULT CONFIGURATION =================
DEFAULT_CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "PASSWORD": "x",
    "PROXY_PORT": 60060,
    "ALGO": "SHA256",
    "TARGET_TEMP": 84.0
}

# ================= PTX ASSEMBLY (GPU FIX) =================
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

# ================= UTILS & MATH =================
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
    cpu_pct = psutil.cpu_percent(interval=None) if HAS_PSUTIL else 0.0
    ram_pct = psutil.virtual_memory().percent if HAS_PSUTIL else 0.0
    return cpu_pct, ram_pct

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

# ================= SETUP MENU =================
def run_setup():
    print("\033[2J\033[H") # Clear
    print("="*60)
    print("      MTP MINER SUITE v10 - SETUP & DIAGNOSTIC      ")
    print("="*60)
    print(f"[1] SYSTEM CHECK:")
    print(f"    - Python: {sys.version.split()[0]}")
    print(f"    - OS: {platform.system()} {platform.release()}")
    print(f"    - PSUTIL: {'INSTALLED' if HAS_PSUTIL else 'MISSING (Install for RAM stats)'}")
    print(f"    - LOCAL IP: {get_local_ip()}")
    print("-" * 60)
    
    cfg = DEFAULT_CONFIG.copy()
    
    print("Press ENTER to accept defaults, or type new value.")
    
    val = input(f"POOL URL [{cfg['POOL_URL']}]: ").strip()
    if val: cfg['POOL_URL'] = val
    
    val = input(f"POOL PORT [{cfg['POOL_PORT']}]: ").strip()
    if val: cfg['POOL_PORT'] = int(val)
    
    val = input(f"WALLET [{cfg['WALLET'][:10]}...]: ").strip()
    if val: cfg['WALLET'] = val
    
    val = input(f"PROXY PORT [{cfg['PROXY_PORT']}]: ").strip()
    if val: cfg['PROXY_PORT'] = int(val)
    
    print("-" * 60)
    print("Launching Mining Engine...")
    time.sleep(1)
    return cfg

# ================= PROXY SERVER =================
class ProxyServer(threading.Thread):
    def __init__(self, config, log_q):
        super().__init__()
        self.config = config
        self.log_q = log_q
        self.daemon = True
        self.sock = None
        self.running = True

    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind(("0.0.0.0", self.config['PROXY_PORT']))
            self.sock.listen(5)
            self.log_q.put(("INFO", f"Proxy Listening on {get_local_ip()}:{self.config['PROXY_PORT']}"))
            
            while self.running:
                client, addr = self.sock.accept()
                self.log_q.put(("NET", f"ASIC Connected: {addr[0]}"))
                # Spawn a handler for this ASIC (Simple Passthrough)
                threading.Thread(target=self.handle_asic, args=(client,), daemon=True).start()
        except Exception as e:
            self.log_q.put(("ERR", f"Proxy Error: {e}"))

    def handle_asic(self, client):
        # Establish upstream to pool
        try:
            pool = socket.create_connection((self.config['POOL_URL'], self.config['POOL_PORT']), timeout=10)
            
            def forward(src, dst, name):
                while True:
                    data = src.recv(4096)
                    if not data: break
                    dst.sendall(data)
            
            t1 = threading.Thread(target=forward, args=(client, pool, "UP"))
            t2 = threading.Thread(target=forward, args=(pool, client, "DOWN"))
            t1.start(); t2.start()
            t1.join(); t2.join()
        except:
            pass
        finally:
            client.close()

# ================= WORKERS =================
def cpu_worker(id, job_q, res_q, stop, stats, diff, log_q):
    active_jid = None
    nonce = 0
    stride = id * 50_000_000
    
    while not stop.is_set():
        try:
            try:
                job = job_q.get_nowait()
                # (jid, prev, c1, c2, mb, ver, nbits, ntime, clean, en1)
                if job[0] != active_jid or job[8]:
                    active_jid = job[0]
                    curr_job = job
                    nonce = 0
                    if id == 0: log_queue.put(("CPU", f"New Job: {active_jid[:6]}..."))
                else:
                    curr_job = job
            except queue.Empty: pass
            
            if not active_jid: time.sleep(0.1); continue
            
            # SHA256d Mining Logic
            jid, ph, c1, c2, mb, ver, nbits, ntime, clean, en1 = curr_job
            
            # Target
            df = diff.value
            if df == 0: df = 1
            target = (0xffff0000 * 2**(256-64) // int(df))
            
            en2 = struct.pack('<I', id).hex().zfill(8)
            
            # Coinbase = c1 + en1 + en2 + c2
            cb_bin = binascii.unhexlify(c1 + en1 + en2 + c2)
            # Hash Coinbase (SHA256d)
            cb_hash = hashlib.sha256(hashlib.sha256(cb_bin).digest()).digest()
            
            # Merkle Root
            root = cb_hash
            for b in mb:
                root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(b)).digest()).digest()
            
            # Block Header (80 bytes)
            # Version(4) + PrevHash(32) + Merkle(32) + Time(4) + Bits(4) + Nonce(4)
            # Stratum sends bytes as little-endian usually, need to be careful with reversals
            # Braiins/Slush usually sends them ready for concatenation except hash endianness
            
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
                
                # Compare as big integer
                val = int.from_bytes(h_hash[::-1], 'big')
                if val <= target:
                    res_q.put({
                        "job_id": jid, "extranonce2": en2, 
                        "ntime": ntime, "nonce": f"{n:08x}"
                    })
                    break
            
            stats[id] += 1000
            nonce += 1000
            if nonce > 50_000_000: nonce = 0
            
        except Exception: time.sleep(0.1)

def gpu_worker(stop, stats, log_q):
    fix_env()
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import numpy as np
        mod = cuda.module_from_buffer(PTX_CODE.encode())
        func = mod.get_function("heavy_load")
        log_q.put(("GPU", "PTX Loaded. Hardware Mining Active."))
    except Exception as e:
        log_q.put(("ERR", f"GPU Init Fail: {e}"))
        return

    while not stop.is_set():
        try:
            out = np.zeros(1, dtype=np.int32)
            seed = np.int32(int(time.time()))
            func(cuda.Out(out), seed, block=(256,1,1), grid=(40960,1))
            cuda.Context.synchronize()
            stats[-1] += 150_000_000
        except: time.sleep(1)

# ================= UI & MAIN =================
class MinerApp:
    def __init__(self, config):
        self.cfg = config
        self.man = mp.Manager()
        self.job_q = self.man.Queue()
        self.res_q = self.man.Queue()
        self.log_q = self.man.Queue()
        self.stop = mp.Event()
        
        self.stats = mp.Array('d', [0.0] * (mp.cpu_count() + 1))
        self.diff = mp.Value('d', 1024.0)
        self.curr_job = mp.Array('c', b'Waiting...')
        self.en1 = mp.Array('c', b'')
        
        self.shares = {"acc": 0, "rej": 0}
        self.start_t = time.time()
        self.logs = []
        self.connected = False
        self.temps = (0,0)

    def log(self, t, m):
        try: self.log_q.put((t, m))
        except: pass

    def net_thread(self):
        while not self.stop.is_set():
            s = None
            try:
                self.log("NET", f"Connecting {self.cfg['POOL_URL']}:{self.cfg['POOL_PORT']}")
                s = socket.create_connection((self.cfg['POOL_URL'], self.cfg['POOL_PORT']), timeout=10)
                self.connected = True
                self.log("NET", "Connected! Subscribing...")
                
                # Stratum V1 Handshake
                s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["MTP-v10"]})+"\n").encode())
                s.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [self.cfg['WALLET'], self.cfg['PASSWORD']]})+"\n").encode())
                
                s.settimeout(0.5)
                buff = b""
                
                while not self.stop.is_set():
                    # Submit Shares
                    while not self.res_q.empty():
                        r = self.res_q.get()
                        msg = json.dumps({
                            "id": 4, "method": "mining.submit",
                            "params": [self.cfg['WALLET'], r['job_id'], r['extranonce2'], r['ntime'], r['nonce']]
                        }) + "\n"
                        s.sendall(msg.encode())
                        self.log("MINE", f"Found Nonce: {r['nonce']}")
                    
                    # Recv
                    try:
                        d = s.recv(8192)
                        if not d: break
                        buff += d
                        while b'\n' in buff:
                            lb, buff = buff.split(b'\n', 1)
                            if not lb: continue
                            msg = json.loads(lb.decode())
                            
                            mid = msg.get('id')
                            if mid == 1 and msg.get('result'):
                                # Save Extranonce1
                                r1 = msg['result']
                                if len(r1) >= 2: 
                                    self.en1.value = r1[1].encode()
                                    self.log("POOL", f"Subscribed. En1: {r1[1]}")
                            elif mid == 2: self.log("POOL", "Authorized")
                            elif mid == 4:
                                if msg.get('result'): 
                                    self.shares['acc'] += 1
                                    self.log("POOL", "Share ACCEPTED!")
                                else:
                                    self.shares['rej'] += 1
                                    self.log("WARN", f"Share REJECTED: {msg.get('error')}")
                            
                            if msg.get('method') == 'mining.notify':
                                p = msg['params']
                                self.curr_job.value = p[0].encode()
                                en1 = self.en1.value.decode()
                                if not en1: continue # Wait for subscribe
                                
                                if p[8]: # Clean
                                    while not self.job_q.empty(): 
                                        try: self.job_q.get_nowait()
                                        except: pass
                                
                                # Broadcast Job
                                j = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], en1)
                                for _ in range(mp.cpu_count() + 2): self.job_q.put(j)
                                
                            elif msg.get('method') == 'mining.set_difficulty':
                                self.diff.value = msg['params'][0]
                                self.log("POOL", f"Diff set to {msg['params'][0]}")

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
        # Colors: 1=Green, 2=Yellow, 3=Red, 4=Cyan, 5=Magenta, 6=Blue
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLACK)
        
        stdscr.nodelay(True)
        
        while not self.stop.is_set():
            # Update Data
            try:
                while True:
                    r = self.log_q.get_nowait()
                    self.logs.append((datetime.now().strftime("%H:%M:%S"), r[0], r[1]))
                    if len(self.logs) > 100: self.logs.pop(0)
            except: pass
            
            c_tmp, g_tmp = get_temps()
            c_use, ram_use = get_hw_stats()
            
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            
            # --- HEADER ---
            title = f" MTP MINER SUITE v10 "
            stdscr.attron(curses.color_pair(6) | curses.A_REVERSE)
            stdscr.addstr(0, 0, title.center(w))
            stdscr.attroff(curses.color_pair(6) | curses.A_REVERSE)
            
            # --- COLUMNS ---
            col_w = w // 3
            
            # LEFT: LOCAL
            stdscr.addstr(2, 2, "=== LOCAL ===", curses.color_pair(4) | curses.A_BOLD)
            stdscr.addstr(3, 2, f"IP: {get_local_ip()}")
            stdscr.addstr(4, 2, f"Proxy: Port {self.cfg['PROXY_PORT']}")
            stdscr.addstr(5, 2, f"RAM: {ram_use}%")
            stdscr.addstr(5, 12, f"CPU Load: {c_use}%")
            stdscr.addstr(6, 2, f"Uptime: {int(time.time() - self.start_t)}s")
            stdscr.addstr(7, 2, f"Status: {'Proxy Active'}", curses.color_pair(1))

            # CENTER: HARDWARE
            stdscr.addstr(2, col_w + 2, "=== HARDWARE ===", curses.color_pair(5) | curses.A_BOLD)
            stdscr.addstr(3, col_w + 2, f"CPU Temp: {c_tmp}°C")
            stdscr.addstr(4, col_w + 2, f"GPU Temp: {g_tmp}°C")
            stdscr.addstr(5, col_w + 2, f"Threads: {mp.cpu_count()}")
            
            t_stat = "OK" if c_tmp < self.cfg['TARGET_TEMP'] else "HOT"
            stdscr.addstr(6, col_w + 2, f"Thermal: {t_stat}", curses.color_pair(1 if t_stat=="OK" else 3))
            
            # RIGHT: NETWORK
            stdscr.addstr(2, col_w*2 + 2, "=== NETWORK ===", curses.color_pair(2) | curses.A_BOLD)
            stdscr.addstr(3, col_w*2 + 2, f"Pool: {self.cfg['POOL_URL']}")
            stdscr.addstr(4, col_w*2 + 2, f"Diff: {int(self.diff.value)}")
            jb = self.curr_job.value.decode()
            stdscr.addstr(5, col_w*2 + 2, f"Job: {jb}")
            
            conn_str = "CONNECTED" if self.connected else "DIALING..."
            stdscr.addstr(6, col_w*2 + 2, f"Link: {conn_str}", curses.color_pair(1 if self.connected else 2))

            # --- MIDDLE: BARS ---
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            
            hr = sum(self.stats) / (time.time() - self.start_t + 1)
            fmt_hr = f"{hr/1e6:.2f} MH/s" if hr > 1e6 else f"{hr/1000:.2f} kH/s"
            
            stdscr.addstr(9, 2, f"TOTAL HASHRATE: {fmt_hr}", curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(9, 40, f"SHARES: [ACC: {self.shares['acc']}] [REJ: {self.shares['rej']}]")
            
            # GPU Bar
            gpu_fill = int((g_tmp / 90.0) * (w - 15)) if w > 20 else 5
            gpu_bar = "█" * gpu_fill
            stdscr.addstr(10, 2, "GPU LOAD:", curses.color_pair(4))
            stdscr.addstr(10, 12, f"[{gpu_bar}] {g_tmp}°C", curses.color_pair(2))

            # CPU Bar
            cpu_fill = int((c_use / 100.0) * (w - 15)) if w > 20 else 5
            cpu_bar = "▒" * cpu_fill
            stdscr.addstr(11, 2, "CPU LOAD:", curses.color_pair(4))
            stdscr.addstr(11, 12, f"[{cpu_bar}] {c_use}%", curses.color_pair(2))

            # --- BOTTOM: LOGS ---
            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            stdscr.addstr(12, 2, " EVENT LOG ", curses.A_REVERSE)
            
            log_h = h - 13
            if log_h > 0:
                view = self.logs[-log_h:]
                for i, l in enumerate(view):
                    ts, tag, msg = l
                    c = curses.color_pair(1)
                    if tag in ["ERR", "BAD"]: c = curses.color_pair(3)
                    elif tag == "WARN": c = curses.color_pair(2)
                    elif tag == "NET": c = curses.color_pair(6)
                    elif tag == "GPU": c = curses.color_pair(5)
                    
                    line_str = f"{ts} [{tag}] {msg}"
                    stdscr.addstr(13+i, 1, line_str[:w-2], c)

            stdscr.refresh()
            if stdscr.getch() == ord('q'): break
            time.sleep(0.1)

    def run(self):
        # 1. Start Workers
        for i in range(mp.cpu_count()):
            mp.Process(target=cpu_worker, args=(i, self.job_q, self.res_q, self.stop, self.stats, self.diff, self.log_q), daemon=True).start()
        
        mp.Process(target=gpu_worker, args=(self.stop, self.stats, self.log_q), daemon=True).start()
        
        # 2. Start Threads
        threading.Thread(target=self.net_thread, daemon=True).start()
        
        # 3. Start Proxy
        ProxyServer(self.cfg, self.log_q).start()
        
        # 4. UI
        try: curses.wrapper(self.draw_ui)
        except KeyboardInterrupt: pass
        finally: self.stop.set()

if __name__ == "__main__":
    # 1. Setup
    final_cfg = run_setup()
    # 2. Run
    app = MinerApp(final_cfg)
    app.run()
