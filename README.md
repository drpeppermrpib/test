#!/usr/bin/env python3
import socket
import ssl
import json
import time
import threading
import multiprocessing as mp
import curses
import binascii
import struct
import hashlib
import os
import sys
from datetime import datetime

# ================= USER CONFIGURATION =================
# 1. PRIMARY (SOLO)
SOLO_URL = "solo.stratum.braiins.com"
SOLO_PORT = 443
SOLO_USER = "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e"
SOLO_PASS = "x"

# 2. FAILOVER (POOL - Used if Solo disconnects)
FAILOVER_URL = "stratum.braiins.com"
FAILOVER_PORT = 3333 # Usually TCP
FAILOVER_USER = "drpeppermrpib.rlm"
FAILOVER_PASS = "x"

# 3. THERMAL TARGET
TARGET_TEMP = 76.0  # The miner will throttle to hold this temp
API_PORT = 60060

# ================= GPU CUDA CHECK =================
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import numpy as np
    HAS_CUDA = True
except:
    HAS_CUDA = False

# ================= SHARED MEMORY =================
class SharedData:
    def __init__(self, manager):
        self.job_queue = manager.Queue()
        self.result_queue = manager.Queue()
        self.log_queue = manager.Queue()
        
        # Sync Variables
        self.stop_event = manager.Event()
        self.throttle_delay = manager.Value('d', 0.0) # Dynamic sleep time (0.0 = Full Speed)
        
        # Stats Dictionary
        self.stats = manager.dict()
        self.stats['accepted'] = 0
        self.stats['rejected'] = 0
        self.stats['pool_name'] = "Initializing..."
        self.stats['connected'] = False
        self.stats['temp_cpu'] = 0.0
        self.stats['temp_gpu'] = 0.0
        self.stats['difficulty'] = 1000.0
        self.stats['gpu_load'] = 0 # 0-100%

# ================= NETWORK ENGINE (FAILOVER SUPPORT) =================
def network_thread(shared):
    """Manages connection, handles SSL EOF errors, switches pools"""
    
    # Pool list: (Name, URL, Port, User, Pass, UseSSL)
    pools = [
        ("SOLO (Primary)", SOLO_URL, SOLO_PORT, SOLO_USER, SOLO_PASS, True),
        ("FAILOVER (Stratum)", FAILOVER_URL, FAILOVER_PORT, FAILOVER_USER, FAILOVER_PASS, False)
    ]
    pool_index = 0

    while not shared.stop_event.is_set():
        name, url, port, user, pwd, use_ssl = pools[pool_index]
        shared.stats['pool_name'] = name
        sock = None
        
        try:
            shared.log_queue.put(("WARN", f"Connecting to {name} @ {url}:{port}..."))
            
            # TCP Connect
            raw_sock = socket.create_connection((url, port), timeout=10)
            
            # SSL Handling
            if use_ssl:
                try:
                    context = ssl.create_default_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    sock = context.wrap_socket(raw_sock, server_hostname=url)
                except ssl.SSLError as e:
                    shared.log_queue.put(("BAD", f"SSL Fail: {e}. Retrying without SSL..."))
                    sock = raw_sock # Fallback to TCP if SSL fails hard
            else:
                sock = raw_sock

            # Stratum Handshake
            sock.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["RLM/4.0"]}) + "\n").encode())
            sock.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [user, pwd]}) + "\n").encode())
            
            shared.stats['connected'] = True
            shared.log_queue.put(("GOOD", f"Connected to {name}!"))
            
            # Loop
            sock.settimeout(0.2)
            buffer = ""
            fails = 0
            
            while not shared.stop_event.is_set():
                # Send Shares
                while not shared.result_queue.empty():
                    res = shared.result_queue.get()
                    req = json.dumps({"id": 4, "method": "mining.submit", "params": [user, res['job_id'], res['extranonce2'], res['ntime'], res['nonce']]}) + "\n"
                    sock.sendall(req.encode())
                    shared.log_queue.put(("INFO", f"Submitting Share > {name}"))

                # Read Data
                try:
                    chunk = sock.recv(4096).decode()
                    if not chunk:
                        raise Exception("Pool closed connection")
                    buffer += chunk
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if not line: continue
                        msg = json.loads(line)
                        
                        if msg.get('method') == 'mining.notify':
                            p = msg['params']
                            # Job: id, prev, c1, c2, merkle, ver, nbits, ntime, clean
                            job = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
                            
                            if p[8]: # Clean
                                while not shared.job_queue.empty(): shared.job_queue.get()
                            
                            for _ in range(mp.cpu_count() + 2):
                                shared.job_queue.put(job)
                            
                        elif msg.get('method') == 'mining.set_difficulty':
                            shared.stats['difficulty'] = msg['params'][0]

                        elif msg.get('result') == True:
                            shared.stats['accepted'] += 1
                            shared.log_queue.put(("GOOD", ">>> SHARE ACCEPTED <<<"))
                        
                        elif msg.get('error'):
                            shared.stats['rejected'] += 1
                            shared.log_queue.put(("BAD", f"Reject: {msg['error']}"))

                except socket.timeout:
                    pass
                except Exception as e:
                    shared.log_queue.put(("BAD", f"Socket: {e}"))
                    break # Break inner loop to reconnect

        except Exception as e:
            shared.stats['connected'] = False
            shared.log_queue.put(("BAD", f"Connection Failed: {e}"))
            time.sleep(5)
            # Switch Pool Index
            pool_index = (pool_index + 1) % len(pools)
            shared.log_queue.put(("WARN", f"Switching to {pools[pool_index][0]}"))

# ================= SMART THERMAL CONTROLLER =================
def thermal_manager(shared):
    """PID-like controller to ramp up/down hashrate to hit 76C"""
    while not shared.stop_event.is_set():
        try:
            # 1. Read Temp
            temp = 0.0
            res = os.popen("sensors").read()
            for line in res.split("\n"):
                if "Tdie" in line or "Package id 0" in line:
                    parts = line.split("+")
                    if len(parts) > 1:
                        temp = float(parts[1].split("°")[0])
                        break
            
            shared.stats['temp_cpu'] = temp
            
            # 2. Dynamic Adjustment (Ramp logic)
            current_delay = shared.throttle_delay.value
            
            if temp > TARGET_TEMP:
                # Too hot! Slow down immediately
                shared.throttle_delay.value = min(current_delay + 0.05, 0.5) 
            elif temp > (TARGET_TEMP - 2.0):
                # Approaching limit, minor slowdown
                shared.throttle_delay.value = min(current_delay + 0.001, 0.5)
            elif temp < (TARGET_TEMP - 5.0):
                # Cool enough, speed up
                shared.throttle_delay.value = max(current_delay - 0.01, 0.0)
            
            # GPU Temp
            try:
                g = os.popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader").read().strip()
                shared.stats['temp_gpu'] = float(g)
            except: pass
            
        except: pass
        time.sleep(1)

# ================= WORKERS =================
def cpu_worker(id, shared):
    while not shared.stop_event.is_set():
        # SMART RAMP UP: Check throttle delay
        delay = shared.throttle_delay.value
        if delay > 0:
            time.sleep(delay)

        if shared.job_queue.empty():
            time.sleep(0.1)
            continue
            
        try:
            job = shared.job_queue.get()
            job_id, prev, c1, c2, branch, ver, nbits, ntime, clean = job
            
            # Standard Merkle/Header build ...
            en2 = struct.pack('<I', id).hex().zfill(8)
            cb = binascii.unhexlify(c1 + en2 + c2)
            cb_hash = hashlib.sha256(hashlib.sha256(cb).digest()).digest()
            root = cb_hash
            for b in branch:
                root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(b)).digest()).digest()
            
            header_pre = binascii.unhexlify(ver)[::-1] + binascii.unhexlify(prev)[::-1] + root + binascii.unhexlify(ntime)[::-1] + binascii.unhexlify(nbits)[::-1]
            
            # Diff Target
            d = shared.stats['difficulty']
            target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000 // int(d)
            
            nonce = id * 1000000
            # Small batch so we can check throttle frequently
            for n in range(nonce, nonce + 5000):
                h = header_pre + struct.pack('<I', n)
                ph = hashlib.sha256(hashlib.sha256(h).digest()).digest()
                if int.from_bytes(ph[::-1], 'big') <= target:
                    shared.result_queue.put({"job_id": job_id, "extranonce2": en2, "ntime": ntime, "nonce": f"{n:08x}"})

        except: pass

def gpu_worker(shared):
    if not HAS_CUDA: return
    # CUDA Kernel for Load
    CUDA_SRC = "__global__ void load(float *a) { int i = threadIdx.x; a[i] = a[i] * sin(a[i]); }"
    try:
        mod = SourceModule(CUDA_SRC)
        func = mod.get_function("load")
        
        while not shared.stop_event.is_set():
            # Throttle GPU based on CPU temp logic too (keep system stable)
            delay = shared.throttle_delay.value
            if delay > 0: time.sleep(delay)
            
            # Run load
            a = np.random.randn(512).astype(np.float32)
            a_gpu = cuda.mem_alloc(a.nbytes)
            cuda.memcpy_htod(a_gpu, a)
            func(a_gpu, block=(512,1,1), grid=(1024,1))
            
            shared.stats['gpu_load'] = 100 - int(delay * 200) # Fake metric based on delay
            if shared.stats['gpu_load'] < 0: shared.stats['gpu_load'] = 0
            if shared.stats['gpu_load'] > 100: shared.stats['gpu_load'] = 100
            
    except: pass

# ================= UI =================
def draw_bar(val, max_val, width=20, color=0):
    pct = val / max_val
    fill = int(width * pct)
    bar = "█" * fill + "░" * (width - fill)
    return bar

def main(stdscr):
    # Colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Good
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Warn
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)    # Bad
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Info
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)# GPU
    
    curses.curs_set(0)
    stdscr.nodelay(True)

    manager = mp.Manager()
    shared = SharedData(manager)

    # Threads
    t_net = threading.Thread(target=network_thread, args=(shared,), daemon=True)
    t_net.start()
    
    t_therm = threading.Thread(target=thermal_manager, args=(shared,), daemon=True)
    t_therm.start()

    # Processes
    procs = []
    n_cpu = mp.cpu_count()
    for i in range(n_cpu):
        p = mp.Process(target=cpu_worker, args=(i, shared))
        p.start()
        procs.append(p)
        
    if HAS_CUDA:
        p_gpu = mp.Process(target=gpu_worker, args=(shared,))
        p_gpu.start()
        procs.append(p_gpu)

    # UI Loop
    logs = []
    try:
        while True:
            # Logs
            while not shared.log_queue.empty():
                logs.append(shared.log_queue.get())
                if len(logs) > 50: logs.pop(0)

            stdscr.clear()
            h, w = stdscr.getmaxyx()
            
            # --- HEADER ---
            stdscr.addstr(0, 0, f" RLM ULTIMATE MINER | {shared.stats['pool_name']} ".center(w), curses.A_REVERSE | curses.color_pair(4))
            
            # --- STATUS & TEMP ---
            conn = shared.stats['connected']
            c_col = curses.color_pair(1) if conn else curses.color_pair(3)
            status_txt = "CONNECTED" if conn else "CONNECTING..."
            
            stdscr.addstr(2, 2, "STATUS:", curses.color_pair(4))
            stdscr.addstr(2, 12, status_txt, c_col)
            
            # Thermal Ramp Logic
            cur_temp = shared.stats['temp_cpu']
            throttle = shared.throttle_delay.value
            
            t_col = curses.color_pair(1)
            if cur_temp > TARGET_TEMP - 2: t_col = curses.color_pair(2)
            if cur_temp >= TARGET_TEMP: t_col = curses.color_pair(3)
            
            stdscr.addstr(2, 40, f"TEMP (TARGET {TARGET_TEMP}°C):", curses.color_pair(4))
            stdscr.addstr(2, 65, f"{cur_temp}°C", t_col)
            
            # Efficiency Bar (Inverse of Throttle)
            eff = max(0, 100 - (throttle * 200)) # Approx conversion
            bar = draw_bar(eff, 100, 20)
            stdscr.addstr(3, 40, f"CPU LOAD: [{bar}] {eff:.1f}%", curses.color_pair(2))

            # --- GPU BAR ---
            if HAS_CUDA:
                g_load = shared.stats['gpu_load']
                g_bar = draw_bar(g_load, 100, 20)
                stdscr.addstr(4, 40, f"GPU LOAD: [{g_bar}] {g_load}%", curses.color_pair(5))
                stdscr.addstr(5, 40, f"GPU TEMP: {shared.stats['temp_gpu']}°C", curses.color_pair(5))
            else:
                stdscr.addstr(4, 40, "GPU: NOT DETECTED (NO PYCUDA)", curses.color_pair(3))

            # --- SHARES ---
            stdscr.addstr(4, 2, f"ACCEPTED: {shared.stats['accepted']}", curses.color_pair(1))
            stdscr.addstr(5, 2, f"REJECTED: {shared.stats['rejected']}", curses.color_pair(3))
            stdscr.addstr(6, 2, f"DIFFICULTY: {shared.stats['difficulty']}", curses.color_pair(4))

            # --- LOGS ---
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(8, 2, " SYSTEM LOG ", curses.A_REVERSE)
            
            for i, (lvl, msg) in enumerate(logs[-(h-10):]):
                c = curses.color_pair(4)
                if lvl == "GOOD": c = curses.color_pair(1)
                elif lvl == "WARN": c = curses.color_pair(2)
                elif lvl == "BAD": c = curses.color_pair(3)
                
                try: stdscr.addstr(9+i, 1, f"[{lvl}] {msg}"[:w-2], c)
                except: pass

            stdscr.refresh()
            time.sleep(0.1)
            
            if stdscr.getch() == ord('q'): break

    except KeyboardInterrupt: pass
    finally:
        shared.stop_event.set()
        for p in procs: p.terminate()

if __name__ == "__main__":
    curses.wrapper(main)
