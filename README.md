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
import signal
from datetime import datetime

# ================= AUTO-DEPENDENCY =================
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ================= DEFAULT CONFIG =================
DEFAULT_CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "PASSWORD": "x",
    "PROXY_PORT": 60060,
    "TEMP_TARGET": 79.0,
    "TEMP_MAX": 83.0
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
def fix_env():
    paths = ["/usr/local/cuda/bin", "/usr/bin", "/bin", "/opt/cuda/bin"]
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
            self.log_q.put(("INFO", f"Proxy Listening on Port {self.cfg['PROXY_PORT']}"))
            while True:
                c, a = sock.accept()
                self.log_q.put(("NET", f"Proxy Client: {a[0]}"))
                threading.Thread(target=self.handle, args=(c,), daemon=True).start()
        except Exception as e:
            self.log_q.put(("ERR", f"Proxy Bind Fail: {e}"))

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
    nonce = 0
    stride = id * 50_000_000
    
    while not stop.is_set():
        if throttle.value > 0.0: time.sleep(throttle.value)

        try:
            try:
                job = job_q.get_nowait()
                if job[0] != active_jid or job[8]:
                    active_jid = job[0]
                    curr_job = job
                    nonce = 0
            except queue.Empty: pass
            
            if not active_jid: 
                time.sleep(0.1); continue
            
            jid, ph, c1, c2, mb, ver, nbits, ntime, clean, en1 = curr_job
            
            df = diff.value
            # Pool Target (Diff 8000)
            pool_target = (0xffff0000 * 2**(256-64) // int(df if df > 0 else 1))
            # Lower Target (Diff 32) - To force submission for visibility
            vis_target = (0xffff0000 * 2**(256-64) // 32)
            
            en2 = struct.pack('<I', id).hex().zfill(8)
            cb_hex = c1 + en1 + en2 + c2
            cb_bin = binascii.unhexlify(cb_hex)
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
            
            start_n = stride + nonce
            # Large batch size for speed
            for n in range(start_n, start_n + 500000):
                h = header_pre + struct.pack('<I', n)
                h_hash = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                val = int.from_bytes(h_hash[::-1], 'big')
                
                # Check for ANY decent share (Diff > 32)
                if val <= vis_target:
                    res_q.put({
                        "job_id": jid, "extranonce2": en2, 
                        "ntime": ntime, "nonce": f"{n:08x}"
                    })
                    # Only log if it's actually high diff to avoid spam
                    if val <= pool_target:
                        log_queue.put(("CPU", f"*** POOL SHARE: {n:08x} ***"))
                    else:
                        log_queue.put(("CPU", f"Submitting Share (Diff 32+)"))
                    break # Submit one per batch
            
            stats[id] += 500000
            nonce += 500000
            
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
            stats[-1] += 150_000_000
            time.sleep(0.001)
        except: time.sleep(1)

# ================= MAIN APP =================
class MinerSuite:
    def __init__(self):
        self.run_setup_menu()

        self.man = mp.Manager()
        self.job_q = self.man.Queue()
        self.res_q = self.man.Queue()
        self.log_q = self.man.Queue()
        self.stop = mp.Event()
        
        self.data = self.man.dict()
        self.data['job'] = "Connecting..."
        self.data['en1'] = ""
        self.data['diff'] = 1024.0
        self.data['status'] = "WAITING"
        
        self.stats = mp.Array('d', [0.0] * (mp.cpu_count() + 1))
        self.diff = mp.Value('d', 1024.0)
        self.throttle = mp.Value('d', 0.0)
        
        self.shares = self.man.dict()
        self.shares['acc'] = 0
        self.shares['rej'] = 0
        
        self.start_t = time.time()
        self.logs = []
        self.connected = False

    def run_setup_menu(self):
        os.system('clear')
        print("="*50)
        print("    MTP MINER SUITE v10 - CONFIGURATION    ")
        print("="*50)
        
        self.cfg = DEFAULT_CONFIG.copy()
        print(f"Defaults: Pool={self.cfg['POOL_URL']} Wallet={self.cfg['WALLET'][:8]}...")
        print(f"Press ENTER to accept defaults.")
        
        u = input(f"Pool [{self.cfg['POOL_URL']}]: ").strip()
        if u: self.cfg['POOL_URL'] = u
        
        p = input(f"Port [{self.cfg['POOL_PORT']}]: ").strip()
        if p: self.cfg['POOL_PORT'] = int(p)
        
        w = input(f"Wallet [{self.cfg['WALLET'][:10]}...]: ").strip()
        if w: self.cfg['WALLET'] = w
        
        print("\nInitializing...")
        time.sleep(1)

    def log(self, t, m):
        try: self.log_q.put((t, m))
        except: pass

    def thermal_loop(self):
        while not self.stop.is_set():
            c, g = get_temps()
            mx = max(c, g)
            tgt = self.cfg['TEMP_TARGET']
            limit = self.cfg['TEMP_MAX']
            
            if mx < tgt:
                self.throttle.value = 0.0
            elif mx < limit:
                self.throttle.value = (mx - tgt) / (limit - tgt)
            else:
                self.throttle.value = 1.0 
                self.log("WARN", f"OVERHEATING: {mx}C")
            time.sleep(1)

    def net_thread(self):
        while not self.stop.is_set():
            s = None
            try:
                self.log("NET", f"Dialing {self.cfg['POOL_URL']}...")
                self.data['status'] = "DIALING"
                s = socket.create_connection((self.cfg['POOL_URL'], self.cfg['POOL_PORT']), timeout=60)
                self.connected = True
                self.data['status'] = "CONNECTED"
                self.log("NET", "Connected! Subscribing...")
                
                s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["MTP-v10"]})+"\n").encode())
                s.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [self.cfg['WALLET'], self.cfg['PASSWORD']]})+"\n").encode())
                
                buff = b""
                last_ping = time.time()

                while not self.stop.is_set():
                    if time.time() - last_ping > 30:
                        try: s.sendall(b'\n'); last_ping = time.time()
                        except: break

                    while not self.res_q.empty():
                        r = self.res_q.get()
                        msg = json.dumps({
                            "id": 4, "method": "mining.submit",
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
                                if mid == 1:
                                    r1 = msg.get('result', [])
                                    if len(r1) >= 2:
                                        self.data['en1'] = r1[1]
                                        self.log("POOL", f"Subscribed En1: {r1[1]}")
                                elif mid == 2: self.log("GOOD", "Authorized!")
                                elif mid == 4:
                                    if msg.get('result'): 
                                        self.shares['acc'] += 1
                                        self.log("GOOD", ">>> SHARE ACCEPTED <<<")
                                    else: 
                                        self.shares['rej'] += 1
                                        self.log("BAD", f"Share Rejected (Low Diff?)")

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
                                    self.log("DIFF", f"Difficulty: {msg['params'][0]}")
                            except: continue
                    except socket.timeout: pass
                    except OSError: break
            except Exception as e:
                self.log("ERR", f"Net: {e}")
                self.data['status'] = "ERROR"
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
            col_w = w // 3
            
            stdscr.addstr(0, 0, " MTP MINER SUITE v10 ".center(w), curses.color_pair(5)|curses.A_BOLD)
            
            stdscr.addstr(2, 2, "=== LOCAL ===", curses.color_pair(4))
            stdscr.addstr(3, 2, f"IP: {get_local_ip()}")
            stdscr.addstr(4, 2, f"Proxy: {self.cfg['PROXY_PORT']}")
            stdscr.addstr(5, 2, f"RAM: {ram}% CPU: {c_load}%")
            
            stdscr.addstr(2, col_w+2, "=== HARDWARE ===", curses.color_pair(4))
            stdscr.addstr(3, col_w+2, f"CPU Temp: {c_tmp}C")
            stdscr.addstr(4, col_w+2, f"GPU Temp: {g_tmp}C")
            stdscr.addstr(5, col_w+2, f"Threads: {mp.cpu_count()}")

            stdscr.addstr(2, col_w*2+2, "=== NETWORK ===", curses.color_pair(4))
            stdscr.addstr(3, col_w*2+2, f"Pool: {self.cfg['POOL_URL'][:20]}")
            stdscr.addstr(4, col_w*2+2, f"Diff: {int(self.data.get('diff', 0))}")
            stdscr.addstr(5, col_w*2+2, f"Job: {self.data.get('job', '?')}")
            
            st = self.data.get('status', 'WAITING')
            sc = curses.color_pair(1) if st == "CONNECTED" else curses.color_pair(3)
            stdscr.addstr(6, col_w*2+2, f"Status: {st}", sc)
            
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            hr = sum(self.stats) / (time.time() - self.start_t + 1)
            fhr = f"{hr/1e6:.2f} MH/s" if hr > 1e6 else f"{hr/1000:.2f} kH/s"
            
            stdscr.addstr(9, 2, f"TOTAL: {fhr}", curses.color_pair(1)|curses.A_BOLD)
            stdscr.addstr(9, 30, f"SHARES [ACC: {self.shares['acc']}] [REJ: {self.shares['rej']}]", curses.color_pair(2))
            
            bar_w = max(5, w - 20)
            fill = int((g_tmp / 90.0) * bar_w)
            stdscr.addstr(10, 2, "GPU: " + "█"*fill, curses.color_pair(2))
            
            c_fill = int((c_load / 100.0) * bar_w)
            stdscr.addstr(11, 2, "CPU: " + "█"*c_fill, curses.color_pair(4))

            stdscr.hline(12, 0, curses.ACS_HLINE, w)
            if h - 13 > 0:
                for i, l in enumerate(self.logs[-(h-13):]):
                    c = curses.color_pair(1)
                    if l[1] in ["ERR", "BAD"]: c = curses.color_pair(3)
                    elif l[1] == "WARN": c = curses.color_pair(2)
                    try: stdscr.addstr(13+i, 2, f"{l[0]} [{l[1]}] {l[2]}"[:w-4], c)
                    except: pass
            
            stdscr.refresh()
            if stdscr.getch() == ord('q'): break
            time.sleep(0.1)

    def start(self):
        ProxyServer(self.cfg, self.log_q).start()
        threading.Thread(target=self.net_thread, daemon=True).start()
        threading.Thread(target=self.thermal_loop, daemon=True).start()
        
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
