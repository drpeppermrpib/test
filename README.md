#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KXT MINER SUITE v43 - SIMPLE CORE
=================================
Target: solo.stratum.braiins.com:3333
Architecture: Heavy Beta v20 (Uncompressed)
Sensor: OLD / SIMPLE (No Die Sense)
Shutdown: STANDARD (No Terminator)
"""

import sys

# FIX: Large integer string conversion limit (screenshot match)
try:
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
except:
    pass

import os
import time
import socket
import json
import threading
import multiprocessing as mp
import binascii
import struct
import hashlib
import random
import select
import subprocess
import signal
import resource
import math
import re
import queue
from datetime import datetime

# ==============================================================================
# SECTION 1: STANDARD SYSTEM PREP (NO KILL SEQUENCE)
# ==============================================================================

# Global Event to signal all threads/processes to stop
EXIT_FLAG = mp.Event()

def signal_handler(signum, frame):
    """Standard, non-aggressive shutdown."""
    if not EXIT_FLAG.is_set():
        EXIT_FLAG.set()
        # Allow UI to close gracefully if running
        try:
            import curses
            curses.endwin()
        except: pass
        print("\n[KXT] Stopping...")
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def boot_checks():
    """System resource configuration."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))
    except Exception:
        pass

boot_checks()

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import curses
except ImportError:
    print("[ERROR] Curses library not found.")
    sys.exit(1)

# ==============================================================================
# SECTION 2: CONFIGURATION
# ==============================================================================

CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.rig1",
    "PASS": "x",
    "PROXY_PORT": 60060,
    "BENCH_DURATION": 60,
    "THROTTLE_TEMP": 79.0,
    "RESUME_TEMP": 75.0,
}

# ==============================================================================
# SECTION 3: CUDA KERNEL
# ==============================================================================

CUDA_SOURCE_CODE = """
extern "C" {
    #include <stdint.h>
    __global__ void kxt_heavy_load(uint32_t *output, uint32_t seed) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        volatile uint32_t a = 0x6a09e667 + idx + seed;
        volatile uint32_t b = 0xbb67ae85;
        #pragma unroll 128
        for(int i=0; i < 50000; i++) {
            a = (a << 5) | (a >> 27);
            b ^= a;
            a += b + 0xDEADBEEF;
            if (i % 2000 == 0) output[idx % 1024] = a;
        }
    }
}
"""

# ==============================================================================
# SECTION 4: HARDWARE HAL (OLD / SIMPLE LOGIC RESTORED)
# ==============================================================================

class HardwareHAL:
    @staticmethod
    def get_cpu_temp():
        # --- OLD SIMPLE LOGIC ---
        # 1. Try simple thermal_zone0
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                val = float(f.read().strip())
                if val > 1000: val /= 1000.0
                return val
        except: pass
        
        # 2. Try PSUTIL if available (Simple, no scanning)
        if HAS_PSUTIL:
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current > 0: return entry.current
            except: pass
            
        return 0.0

    @staticmethod
    def get_gpu_temp():
        try:
            cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader"
            output = subprocess.check_output(cmd, shell=True)
            return float(output.decode().strip())
        except:
            return 0.0

    @staticmethod
    def force_fan_speed():
        commands = [
            "nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'",
            "nvidia-settings -a 'GPUFanControlState=1' -a 'GPUTargetFanSpeed=100'"
        ]
        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except: pass

# ==============================================================================
# SECTION 5: TEXT-BASED BENCHMARK ENGINE
# ==============================================================================

def benchmark_cpu_worker(stop_event, throttle_event):
    import math
    size = 80
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    B = [[random.random() for _ in range(size)] for _ in range(size)]
    
    while not stop_event.is_set():
        throttle_event.wait()
        result = [[sum(a*b for a,b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

def benchmark_gpu_worker(stop_event, throttle_event):
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        
        mod = SourceModule(CUDA_SOURCE_CODE)
        func = mod.get_function("kxt_heavy_load")
        out_gpu = cuda.mem_alloc(4096)
        
        while not stop_event.is_set():
            throttle_event.wait()
            func(out_gpu, np.uint32(time.time()), block=(512,1,1), grid=(128,1))
            cuda.Context.synchronize()
            
    except: time.sleep(1)

def run_system_benchmark():
    os.system('clear')
    print("\n" + "="*60)
    print(" KXT v43 - SYSTEM AUDIT (SIMPLE SENSORS) ".center(60))
    print("="*60 + "\n")
    
    threading.Thread(target=lambda: [HardwareHAL.force_fan_speed(), time.sleep(5)], daemon=True).start()
    
    stop_benchmark = mp.Event()
    throttle_control = mp.Event()
    throttle_control.set()
    procs = []
    
    print(f"\n[STAGE 1] CPU THERMAL LOAD ({CONFIG['BENCH_DURATION']}s)")
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=benchmark_cpu_worker, args=(stop_benchmark, throttle_control))
        p.start(); procs.append(p)
        
    start_time = time.time()
    is_throttled = False
    
    try:
        while time.time() - start_time < CONFIG['BENCH_DURATION']:
            elapsed = time.time() - start_time
            remaining = int(CONFIG['BENCH_DURATION'] - elapsed)
            cpu_temp = HardwareHAL.get_cpu_temp()
            
            if not is_throttled:
                if cpu_temp >= CONFIG['THROTTLE_TEMP']:
                    throttle_control.clear(); is_throttled = True
            else:
                if cpu_temp <= CONFIG['RESUME_TEMP']:
                    throttle_control.set(); is_throttled = False
            
            status_text = "MAX POWER" if not is_throttled else f"THROTTLED (> {CONFIG['THROTTLE_TEMP']}C)"
            sys.stdout.write(f"\r  >> Time: {remaining}s | CPU: {cpu_temp:.1f}C | Status: {status_text}       ")
            sys.stdout.flush()
            time.sleep(0.5)
            
    except KeyboardInterrupt: pass
        
    print(f"\n\n[STAGE 2] FULL SYSTEM LOAD (+GPU) ({CONFIG['BENCH_DURATION']}s)")
    gpu_proc = mp.Process(target=benchmark_gpu_worker, args=(stop_benchmark, throttle_control))
    gpu_proc.start(); procs.append(gpu_proc)
    
    start_time = time.time()
    try:
        while time.time() - start_time < CONFIG['BENCH_DURATION']:
            elapsed = time.time() - start_time
            remaining = int(CONFIG['BENCH_DURATION'] - elapsed)
            cpu_temp = HardwareHAL.get_cpu_temp()
            gpu_temp = HardwareHAL.get_gpu_temp()
            
            if not is_throttled:
                if cpu_temp >= CONFIG['THROTTLE_TEMP']:
                    throttle_control.clear(); is_throttled = True
            else:
                if cpu_temp <= CONFIG['RESUME_TEMP']:
                    throttle_control.set(); is_throttled = False
                    
            status_text = "MAX POWER" if not is_throttled else "THROTTLED"
            sys.stdout.write(f"\r  >> Time: {remaining}s | CPU: {cpu_temp:.1f}C | GPU: {gpu_temp:.1f}C | Status: {status_text}      ")
            sys.stdout.flush()
            time.sleep(0.5)
            
    except KeyboardInterrupt: pass
        
    stop_benchmark.set()
    for p in procs: p.terminate()
    print("\n\n[SUCCESS] Benchmark Complete. Initializing Mining Interface...")
    time.sleep(2)

# ==============================================================================
# SECTION 6: STRATUM PROTOCOL
# ==============================================================================

class StratumClient(threading.Thread):
    def __init__(self, state, job_q, res_q, log_q):
        super().__init__()
        self.state = state; self.job_q = job_q; self.res_q = res_q; self.log_q = log_q
        self.sock = None; self.msg_id = 1; self.buffer = ""; self.daemon = True
        
    def run(self):
        while not EXIT_FLAG.is_set():
            try:
                self.sock = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=10)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                self.send_request("mining.subscribe", ["KXT-v43"])
                self.send_request("mining.authorize", [f"{CONFIG['WALLET']}.{CONFIG['WORKER']}_local", CONFIG['PASS']])
                self.state.connected.value = True
                self.log_q.put(("NET", f"Connected to {CONFIG['POOL_URL']}"))
                
                while not EXIT_FLAG.is_set():
                    while not self.res_q.empty():
                        share = self.res_q.get()
                        self.send_request("mining.submit", [
                            f"{CONFIG['WALLET']}.{CONFIG['WORKER']}_local",
                            share['jid'], share['en2'], share['ntime'], share['nonce']
                        ])
                        self.log_q.put(("TX", f"Submitting Nonce {share['nonce']}"))
                        with self.state.shares.get_lock(): self.state.shares.value += 1
                            
                    r, _, _ = select.select([self.sock], [], [], 0.1)
                    if r:
                        data = self.sock.recv(4096)
                        if not data: raise ConnectionError("Closed")
                        self.buffer += data.decode()
                        while '\n' in self.buffer:
                            line, self.buffer = self.buffer.split('\n', 1)
                            if line.strip(): self.handle_message(json.loads(line))
                                
            except Exception as e:
                self.state.connected.value = False
                time.sleep(5)

    def send_request(self, method, params):
        try:
            payload = json.dumps({"id": self.msg_id, "method": method, "params": params}) + "\n"
            self.sock.sendall(payload.encode())
            self.msg_id += 1
        except: pass

    def handle_message(self, msg):
        msg_id = msg.get('id'); result = msg.get('result')
        if msg_id == 1 and result:
            self.state.extranonce1 = result[1]
            self.state.extranonce2_size = result[2]
            
        if msg_id and msg_id > 2:
            if result:
                with self.state.accepted.get_lock(): self.state.accepted.value += 1
                self.log_q.put(("RX", "Share ACCEPTED"))
            else:
                with self.state.rejected.get_lock(): self.state.rejected.value += 1
                self.log_q.put(("RX", "Share REJECTED"))
                
        if msg.get('method') == 'mining.notify':
            params = msg['params']
            self.log_q.put(("JOB", f"Block {params[0][:8]}"))
            if params[8]:
                while not self.job_q.empty():
                    try: self.job_q.get_nowait()
                    except: pass
            job_package = (params, self.state.extranonce1, self.state.extranonce2_size)
            for _ in range(mp.cpu_count() * 2): self.job_q.put(job_package)

# ==============================================================================
# SECTION 7: BLOCK HASHING WORKER
# ==============================================================================

def mining_worker_process(worker_id, job_q, res_q, stop_event, throttle_event):
    nonce_start = worker_id * 10000000
    nonce = nonce_start
    current_job_data = None
    
    while not stop_event.is_set():
        throttle_event.wait()
        try:
            try:
                job_package = job_q.get(timeout=0.1)
                params, en1, en2_size = job_package
                if params[8]: nonce = nonce_start
                current_job_data = (params, en1, en2_size)
            except queue.Empty:
                if current_job_data is None: continue
            
            params, en1, en2_size = current_job_data
            job_id, prev_hash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean = params
            
            en2_bin = os.urandom(en2_size)
            en2_hex = binascii.hexlify(en2_bin).decode()
            coinbase_hex = coinb1 + en1 + en2_hex + coinb2
            coinbase_bin = binascii.unhexlify(coinbase_hex)
            coinbase_hash = hashlib.sha256(hashlib.sha256(coinbase_bin).digest()).digest()
            merkle_root = coinbase_hash
            for branch in merkle_branch:
                branch_bin = binascii.unhexlify(branch)
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + branch_bin).digest()).digest()
                
            header_prefix = (
                binascii.unhexlify(version)[::-1] +
                binascii.unhexlify(prev_hash)[::-1] +
                merkle_root +
                binascii.unhexlify(ntime)[::-1] +
                binascii.unhexlify(nbits)[::-1]
            )
            
            target_bin = b'\x00\x00'
            for n in range(nonce, nonce + 5000):
                nonce_bin = struct.pack('<I', n)
                header = header_prefix + nonce_bin
                block_hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                if block_hash.endswith(target_bin):
                    res_q.put({
                        'jid': job_id, 'en2': en2_hex, 'ntime': ntime,
                        'nonce': binascii.hexlify(nonce_bin).decode()
                    })
                    break
            nonce += 5000
        except Exception: continue

# ==============================================================================
# SECTION 8: PROXY SERVER
# ==============================================================================

class ProxyServer(threading.Thread):
    def __init__(self, log_q, state):
        super().__init__(); self.log_q = log_q; self.state = state; self.daemon = True
        
    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("0.0.0.0", CONFIG['PROXY_PORT'])); server.listen(50)
            self.log_q.put(("PRX", f"Proxy Listening on port {CONFIG['PROXY_PORT']}"))
            while not EXIT_FLAG.is_set():
                client, addr = server.accept()
                t = threading.Thread(target=self.handle_client, args=(client, addr))
                t.daemon = True
                t.start()
        except Exception as e:
            self.log_q.put(("ERR", f"Proxy Bind Error: {e}"))

    def handle_client(self, c_sock, addr):
        with self.state.proxy_clients.get_lock(): self.state.proxy_clients.value += 1
        p_sock = None
        try:
            p_sock = socket.create_connection((CONFIG['POOL_URL'], CONFIG['POOL_PORT']), timeout=10)
            inputs = [c_sock, p_sock]
            while not EXIT_FLAG.is_set():
                readable, _, _ = select.select(inputs, [], [], 1)
                if c_sock in readable:
                    data = c_sock.recv(4096)
                    if not data: break
                    p_sock.sendall(data)
                if p_sock in readable:
                    data = p_sock.recv(4096)
                    if not data: break
                    c_sock.sendall(data)
        except: pass
        finally:
            try: c_sock.close()
            except: pass
            try: p_sock.close()
            except: pass
            with self.state.proxy_clients.get_lock(): self.state.proxy_clients.value -= 1

# ==============================================================================
# SECTION 9: GUI
# ==============================================================================

def draw_tui_box(stdscr, y, x, h, w, title, color_pair):
    try:
        stdscr.attron(color_pair)
        stdscr.addch(y, x, curses.ACS_ULCORNER); stdscr.addch(y, x + w - 1, curses.ACS_URCORNER)
        stdscr.addch(y + h - 1, x, curses.ACS_LLCORNER); stdscr.addch(y + h - 1, x + w - 1, curses.ACS_LRCORNER)
        stdscr.hline(y, x + 1, curses.ACS_HLINE, w - 2); stdscr.hline(y + h - 1, x + 1, curses.ACS_HLINE, w - 2)
        stdscr.vline(y + 1, x, curses.ACS_VLINE, h - 2); stdscr.vline(y + 1, x + w - 1, curses.ACS_VLINE, h - 2)
        stdscr.addstr(y, x + 2, f" {title} "); stdscr.attroff(color_pair)
    except: pass

def draw_progress_bar(val, max_val, width=10):
    percent = max(0.0, min(1.0, val / max_val))
    fill = int(percent * width)
    bar = "|" * fill + " " * (width - fill)
    return f"[{bar}]"

def dashboard_main(stdscr, state, job_q, res_q, log_q):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.curs_set(0); stdscr.nodelay(True)
    
    StratumClient(state, job_q, res_q, log_q).start()
    ProxyServer(log_q, state).start()
    
    workers = []
    stop_workers = mp.Event()
    throttle_workers = mp.Event(); throttle_workers.set()
    
    for i in range(mp.cpu_count()):
        p = mp.Process(target=mining_worker_process, args=(i, job_q, res_q, stop_workers, throttle_workers))
        p.start(); workers.append(p)
    gp = mp.Process(target=benchmark_gpu_worker, args=(stop_workers, throttle_workers))
    gp.start(); workers.append(gp)
    
    log_history = []
    is_throttled = False
    
    while not EXIT_FLAG.is_set():
        while not log_q.empty():
            log_history.append(log_q.get())
            if len(log_history) > 20: log_history.pop(0)
                
        cpu_temp = HardwareHAL.get_cpu_temp()
        gpu_temp = HardwareHAL.get_gpu_temp()
        
        if not is_throttled and cpu_temp >= CONFIG['THROTTLE_TEMP']:
            throttle_workers.clear(); is_throttled = True
            log_q.put(("WARN", f"Temp {cpu_temp}C > 79C. Throttling..."))
        elif is_throttled and cpu_temp <= CONFIG['RESUME_TEMP']:
            throttle_workers.set(); is_throttled = False
            log_q.put(("INFO", "Resuming..."))

        stdscr.erase(); height, width = stdscr.getmaxyx()
        
        title = " KXT MINER v43 - SIMPLE CORE "
        stdscr.addstr(0, max(0, (width - len(title)) // 2), title, curses.A_REVERSE | curses.color_pair(4))
        
        draw_tui_box(stdscr, 2, 1, 6, 28, "LOCAL", curses.color_pair(4))
        stdscr.addstr(3, 3, f"CPU: {cpu_temp:.1f}C {draw_progress_bar(cpu_temp, 90, 5)}")
        stdscr.addstr(4, 3, f"GPU: {gpu_temp:.1f}C {draw_progress_bar(gpu_temp, 90, 5)}")
        if is_throttled: stdscr.addstr(5, 3, "STATUS: THROTTLED", curses.color_pair(3))
        else: stdscr.addstr(5, 3, "STATUS: ACTIVE", curses.color_pair(1))
            
        draw_tui_box(stdscr, 2, 30, 6, 28, "NETWORK", curses.color_pair(4))
        conn_str = "CONNECTED" if state.connected.value else "CONNECTING..."
        stdscr.addstr(3, 32, f"Link: {conn_str}", curses.color_pair(1) if state.connected.value else curses.color_pair(2))
        stdscr.addstr(4, 32, f"Shares Sent: {state.shares.value}")
        
        draw_tui_box(stdscr, 2, 59, 6, 28, "STATS", curses.color_pair(4))
        stdscr.addstr(3, 61, f"Accepted: {state.accepted.value}", curses.color_pair(1))
        stdscr.addstr(4, 61, f"Rejected: {state.rejected.value}", curses.color_pair(3))
        
        draw_tui_box(stdscr, 2, 88, 6, 24, "PROXY", curses.color_pair(4))
        stdscr.addstr(3, 90, f"Clients: {state.proxy_clients.value}")
        stdscr.addstr(4, 90, f"Port: {CONFIG['PROXY_PORT']}")
        
        stdscr.hline(8, 0, curses.ACS_HLINE, width)
        for idx, (level, message) in enumerate(log_history):
            row = 9 + idx
            if row >= height - 1: break
            color = curses.color_pair(4)
            if level == "RX": color = curses.color_pair(1)
            if level == "ERR": color = curses.color_pair(3)
            if level == "JOB": color = curses.color_pair(2)
            stdscr.addstr(row, 2, f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}", color)
            
        stdscr.refresh()
        try:
            if stdscr.getch() == ord('q'): EXIT_FLAG.set()
        except: pass
        time.sleep(0.1)
        
    stop_workers.set()
    for p in workers: p.terminate()

if __name__ == "__main__":
    try:
        run_system_benchmark()
        
        manager = mp.Manager()
        state = manager.Namespace()
        state.connected = manager.Value('b', False)
        state.shares = manager.Value('i', 0)
        state.accepted = manager.Value('i', 0)
        state.rejected = manager.Value('i', 0)
        state.proxy_clients = manager.Value('i', 0)
        state.extranonce1 = "00000000"
        state.extranonce2_size = 4
        
        job_queue = manager.Queue()
        res_queue = manager.Queue()
        log_queue = manager.Queue()
        
        curses.wrapper(dashboard_main, state, job_queue, res_queue, log_queue)
        
    except KeyboardInterrupt: pass
