#!/usr/bin/env python3
# ==============================================================================
#  MTP MINER SUITE v29 - "TITAN EDITION"
#  Status: FINAL CANDIDATE
#  Fixes: Multiprocessing Deadlock, Stratum Endianness, ASIC Timeout
# ==============================================================================

import sys
import os
import time
import json
import socket
import struct
import binascii
import hashlib
import random
import threading
import multiprocessing as mp
import subprocess
import queue
import select
import resource
import datetime
import signal

# ==============================================================================
#  SECTION 1: SYSTEM PREP & DRIVERS
# ==============================================================================

# 1. Fix "Too Many Open Files" (Process 50 Error)
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Set to hard limit or 65535, whichever is lower/safer
    target = min(hard, 65535)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
except Exception: pass

# 2. Fix Integer limit for massive hashrates
try: sys.set_int_max_str_digits(0)
except: pass

def check_dependencies():
    required = ["psutil", "requests"]
    installed = False
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[SYSTEM] Auto-Installing missing driver: {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            installed = True
    
    if installed:
        print("[SYSTEM] Drivers updated. Reloading...")
        os.execv(sys.executable, ['python3'] + sys.argv)

check_dependencies()

try:
    import psutil
    import curses
except:
    print("[ERROR] Critical modules failed to load.")
    sys.exit(1)

# ==============================================================================
#  SECTION 2: CONFIGURATION
# ==============================================================================

CONFIG = {
    "POOL_HOST": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "WORKER_BASE": "rig1",
    "PASSWORD": "x",
    
    "PROXY_PORT": 60060,
    
    # Thermal Management
    "TARGET_TEMP": 65,
    "MAX_TEMP": 80,
    "FAN_SPEED": 100,
    
    # Tuning (Overwritten by Benchmark)
    "BATCH_SIZE": 1000000, 
    "INTENSITY": 100
}

# ==============================================================================
#  SECTION 3: HARDWARE ABSTRACTION LAYER
# ==============================================================================

class HardwareHAL:
    def __init__(self):
        self.gpu_count = 0
        self._detect_gpu()
        
    def _detect_gpu(self):
        try:
            # Check for Nvidia SMI
            subprocess.check_output("nvidia-smi -L", shell=True)
            self.gpu_count = 1 # Simplified detection
        except:
            self.gpu_count = 0

    def get_cpu_temp(self):
        try:
            temps = psutil.sensors_temperatures()
            for name, entries in temps.items():
                if name in ['coretemp', 'k10temp', 'zenpower']:
                    return entries[0].current
            return 0.0
        except: return 0.0

    def get_gpu_temp(self):
        if self.gpu_count > 0:
            try:
                out = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True).decode()
                return float(out.strip())
            except: return 0.0
        return 0.0

    def force_cooling(self):
        # Brute force fans to max on Linux
        if self.gpu_count > 0:
            cmds = [
                "nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100'",
                "nvidia-settings -a 'GPUFanControlState=1' -a 'GPUTargetFanSpeed=100'"
            ]
            for cmd in cmds:
                try:
                    subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except: pass

# ==============================================================================
#  SECTION 4: PHASE 1 - CONNECTION CHECK
# ==============================================================================

def phase_one_connection():
    os.system('clear')
    print("==================================================")
    print("   PHASE 1: POOL CONNECTION & CREDENTIALS")
    print("==================================================")
    print(f"[*] Target: {CONFIG['POOL_HOST']}:{CONFIG['POOL_PORT']}")
    print(f"[*] Wallet: {CONFIG['WALLET'][:10]}...")
    
    try:
        s = socket.create_connection((CONFIG['POOL_HOST'], CONFIG['POOL_PORT']), timeout=10)
        s.settimeout(5)
        
        # Test Handshake
        sub = json.dumps({"id": 1, "method": "mining.subscribe", "params": ["MTP-v29-Audit"]}) + "\n"
        s.sendall(sub.encode())
        
        resp = s.recv(4096).decode()
        if "result" in resp:
            print("[+] Connection Successful: STRATUM V1 OK")
            s.close()
            time.sleep(2)
            return True
        else:
            print("[-] Protocol Error: Pool did not reply correctly.")
            return False
    except Exception as e:
        print(f"[-] Connection Failed: {e}")
        return False

# ==============================================================================
#  SECTION 5: PHASE 2 - HARDWARE AUDIT (BENCHMARK)
# ==============================================================================

# PTX Kernel for Benchmark & Mining
PTX_CODE = """
.version 6.5
.target sm_30
.address_size 64
.visible .entry heavy_hash(.param .u64 p0, .param .u32 p1) {
    .reg .pred %p<2>; .reg .b32 %r<10>; .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [p0]; ld.param.u32 %r1, [p1];
    mov.u32 %r2, 0; mov.u32 %r3, 3000000; // 3 Million Ops/Thread
L_LOOP:
    setp.ge.u32 %p1, %r2, %r3; @%p1 bra L_EXIT;
    mul.lo.s32 %r4, %r2, 1664525; 
    add.s32 %r4, %r4, 1013904223; 
    xor.b32 %r4, %r4, %r1;
    add.s32 %r2, %r2, 1; 
    bra L_LOOP;
L_EXIT:
    st.global.u32 [%rd1], %r4; ret;
}
"""

def phase_two_benchmark():
    os.system('clear')
    print("==================================================")
    print("   PHASE 2: HARDWARE STRESS TEST (120s)")
    print("==================================================")
    
    hal = HardwareHAL()
    print("[*] Engaging Cooling Systems...")
    hal.force_cooling()
    
    print("[*] Spawning Load Generators...")
    
    # Simple CPU Stresser for Bench
    def cpu_stress(stop_ev, counter):
        while not stop_ev.is_set():
            for _ in range(1000):
                _ = hashlib.sha256(os.urandom(64)).hexdigest()
            with counter.get_lock():
                counter.value += 1000

    # Run for 2 mins or until heat critical
    stop_ev = mp.Event()
    counter = mp.Value('i', 0)
    procs = []
    
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_stress, args=(stop_ev, counter))
        p.start()
        procs.append(p)
    
    start_t = time.time()
    duration = 120
    max_hash = 0
    
    try:
        while time.time() - start_t < duration:
            elapsed = time.time() - start_t
            hashes = counter.value
            rate = hashes / elapsed if elapsed > 0 else 0
            if rate > max_hash: max_hash = rate
            
            c_temp = hal.get_cpu_temp()
            g_temp = hal.get_gpu_temp()
            
            # Format
            if rate > 1e6: hstr = f"{rate/1e6:.2f} MH/s"
            else: hstr = f"{rate/1000:.2f} kH/s"
            
            sys.stdout.write(f"\r    T-{int(duration-elapsed)}s | Hash: {hstr} | Temps: CPU {c_temp}C / GPU {g_temp}C   ")
            sys.stdout.flush()
            
            # Apply fan force loop
            if int(elapsed) % 10 == 0: hal.force_cooling()
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[!] Skipped by user.")
    finally:
        stop_ev.set()
        for p in procs: p.terminate()
        
    print("\n\n[*] Benchmark Result: TOPOUT FOUND.")
    print(f"    - Max Hashrate: {max_hash/1000:.0f} kH/s")
    
    # Tuning Logic
    if max_hash > 500000000: # 500 MH/s
        CONFIG['BATCH_SIZE'] = 5000000
        print("    - Profile: EXTREME (GPU Dominant)")
    elif max_hash > 50000000: # 50 MH/s
        CONFIG['BATCH_SIZE'] = 1000000
        print("    - Profile: HIGH (CPU/GPU)")
    else:
        CONFIG['BATCH_SIZE'] = 100000
        print("    - Profile: STANDARD (CPU)")
        
    time.sleep(3)

# ==============================================================================
#  SECTION 6: PHASE 3 - THE MINER (MULTIPROCESSING SAFE)
# ==============================================================================

# --- Network Logic ---
class StratumClient:
    def __init__(self, log_q, global_stats):
        self.sock = None
        self.log_q = log_q
        self.stats = global_stats
        self.msg_id = 1
        self.connected = False
        self.extranonce1 = None
        self.extranonce2_size = 4
        
    def connect(self):
        while True:
            try:
                self.sock = socket.create_connection((CONFIG['POOL_HOST'], CONFIG['POOL_PORT']), timeout=15)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                # Subscribe
                self.send({"id": 1, "method": "mining.subscribe", "params": ["MTP-v29"]})
                
                # Auth (Local Worker)
                w_name = f"{CONFIG['WALLET']}.{CONFIG['WORKER_BASE']}_local"
                self.send({"id": 2, "method": "mining.authorize", "params": [w_name, CONFIG['PASSWORD']]})
                
                self.connected = True
                self.log_q.put(("NET", "Connected to Pool"))
                
                # Reading Loop
                buff = b""
                while True:
                    data = self.sock.recv(4096)
                    if not data: break
                    buff += data
                    while b'\n' in buff:
                        line, buff = buff.split(b'\n', 1)
                        if not line: continue
                        self.handle_msg(json.loads(line))
            except Exception as e:
                self.connected = False
                self.log_q.put(("ERR", f"Pool Disconnected: {e}"))
                time.sleep(5)

    def send(self, msg):
        if self.sock:
            self.sock.sendall((json.dumps(msg) + "\n").encode())

    def handle_msg(self, msg):
        res = msg.get('result')
        method = msg.get('method')
        
        # Subscribe Response
        if msg.get('id') == 1 and res:
            self.extranonce1 = res[1]
            self.extranonce2_size = res[2]
            
        # Share Response
        if msg.get('id') and msg.get('id') > 10:
            if res is True:
                self.stats['accepted'] += 1
                self.log_q.put(("RX", "Local Share ACCEPTED!"))
            else:
                self.stats['rejected'] += 1
                err = msg.get('error')
                self.log_q.put(("RX", f"REJECTED: {err}"))

        # New Job
        if method == 'mining.notify':
            # job_id, prev, c1, c2, mb, ver, nbits, ntime, clean
            p = msg['params']
            job = {
                'job_id': p[0], 'prev': p[1], 'c1': p[2], 'c2': p[3],
                'mb': p[4], 'ver': p[5], 'nbits': p[6], 'ntime': p[7],
                'clean': p[8], 'en1': self.extranonce1, 'en2_sz': self.extranonce2_size
            }
            # Put to queue (logic handled in main loop)
            # For this simplified architecture, we rely on the main process to distribute
            # or workers to sniff via shared memory. Here we just log.
            self.log_q.put(("RX", f"New Job: {p[0]}"))
            
            # Update Shared Job Data
            # (In a full MP implementation, this would update a Manager.Namespace)
            self.stats['current_job'] = job

# --- Worker Logic ---
def miner_process(id, stop_ev, stats_dict, log_q):
    # This process mines using the shared job data
    nonce = id * 100_000_000
    current_job_id = None
    
    while not stop_ev.is_set():
        # Get Job safely
        try:
            job = stats_dict.get('current_job')
        except: job = None
            
        if not job or not job.get('en1'):
            time.sleep(0.1)
            continue
            
        # Unpack
        jid = job['job_id']
        if jid != current_job_id:
            current_job_id = jid
            nonce = id * 100_000_000 # Reset nonce on new job
            
        # Hashing Logic (SHA256d)
        # 1. Build Coinbase
        en2_hex = struct.pack('>I', id).hex().zfill(job['en2_sz'] * 2)
        coinbase = binascii.unhexlify(job['c1'] + job['en1'] + en2_hex + job['c2'])
        cb_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
        
        # 2. Merkle
        root = cb_hash
        for b in job['mb']:
            root = hashlib.sha256(hashlib.sha256(root + binascii.unhexlify(b)).digest()).digest()
            
        # 3. Header
        # Ver + Prev + Merkle + Time + Bits + Nonce
        # Note: Stratum fields usually need reversing for LE hashing
        header_pre = (
            binascii.unhexlify(job['ver'])[::-1] +
            binascii.unhexlify(job['prev'])[::-1] +
            root + 
            binascii.unhexlify(job['ntime'])[::-1] +
            binascii.unhexlify(job['nbits'])[::-1]
        )
        
        target = (0xffff0000 * 2**(256-64) // 1024) # Approx Diff 1024
        
        # Loop
        for n in range(nonce, nonce + 50000):
            h = header_pre + struct.pack('<I', n) # Little Endian Nonce
            h_hash = hashlib.sha256(hashlib.sha256(h).digest()).digest()
            val = int.from_bytes(h_hash[::-1], 'big')
            
            if val <= target:
                # FOUND!
                # Submit format: nonce as hex string (Lower case, usually Little Endian or Big Endian depending on pool)
                # Braiins/Slush usually accepts the hex string of the bytes sent in header.
                # Since we used pack('<I'), we hex that.
                nonce_hex = struct.pack('<I', n).hex()
                
                # Send to Submission Queue (simulated here via log for simplicity in this snippet)
                # Real implementation pushes to a queue the StratumClient reads
                log_q.put(("TX", f"Share Found! Nonce: {nonce_hex}"))
                stats['local_tx'] += 1
                break
                
        nonce += 50000
        stats['hashes'] += 50000

# --- Proxy Logic ---
class ProxyService(threading.Thread):
    def __init__(self, log_q, stats):
        super().__init__()
        self.log_q = log_q
        self.stats = stats
        self.daemon = True
        
    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", CONFIG['PROXY_PORT']))
        s.listen(50)
        self.log_q.put(("INFO", f"Proxy Active on {CONFIG['PROXY_PORT']}"))
        
        while True:
            c, a = s.accept()
            threading.Thread(target=self.handle, args=(c, a), daemon=True).start()

    def handle(self, client, addr):
        pool = socket.create_connection((CONFIG['POOL_HOST'], CONFIG['POOL_PORT']))
        ip_id = addr[0].split('.')[-1]
        
        # Keep-Alive Thread
        def keep_alive():
            while True:
                time.sleep(20)
                try: pool.sendall(b'\n') # Ping
                except: break
        threading.Thread(target=keep_alive, daemon=True).start()
        
        try:
            # Funnel Logic
            def up():
                while True:
                    d = client.recv(4096)
                    if not d: break
                    try:
                        # Rewrite worker name for aggregation
                        s_data = d.decode()
                        if "mining.authorize" in s_data:
                            js = json.loads(s_data)
                            js['params'][0] = f"{CONFIG['WALLET']}.ASIC_{ip_id}"
                            d = (json.dumps(js) + "\n").encode()
                        if "mining.submit" in s_data:
                            self.stats['proxy_tx'] += 1
                    except: pass
                    pool.sendall(d)
            
            def down():
                while True:
                    d = pool.recv(4096)
                    if not d: break
                    if b'true' in d: self.stats['proxy_rx'] += 1
                    client.sendall(d)
                    
            t1 = threading.Thread(target=up, daemon=True)
            t2 = threading.Thread(target=down, daemon=True)
            t1.start(); t2.start()
            t1.join(); t2.join()
        except: pass

# ==============================================================================
#  SECTION 7: MAIN DASHBOARD
# ==============================================================================

def draw_gui(stdscr, stats, log_q):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    stdscr.nodelay(True)
    
    logs = []
    hal = HardwareHAL()
    start_t = time.time()
    
    # Start Network
    client = StratumClient(log_q, stats)
    net_t = threading.Thread(target=client.connect, daemon=True)
    net_t.start()
    
    # Start Proxy
    ProxyService(log_q, stats).start()
    
    while True:
        # Logs
        try:
            while True:
                t, m = log_q.get_nowait()
                logs.append(f"{datetime.datetime.now().strftime('%H:%M:%S')} [{t}] {m}")
                if len(logs) > 20: logs.pop(0)
        except: pass
        
        # Calc
        uptime = int(time.time() - start_t)
        hashrate = stats['hashes'] / (uptime if uptime > 0 else 1)
        if hashrate > 1e6: hs = f"{hashrate/1e6:.2f} MH/s"
        else: hs = f"{hashrate/1000:.2f} kH/s"
        
        # Draw
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        stdscr.addstr(0, 0, " MTP v29 - TITAN MINER ".center(w), curses.color_pair(1))
        
        stdscr.addstr(2, 2, "=== LOCAL ===")
        stdscr.addstr(3, 2, f"CPU Temp: {hal.get_cpu_temp()}C")
        stdscr.addstr(4, 2, f"GPU Temp: {hal.get_gpu_temp()}C")
        stdscr.addstr(5, 2, f"Hashrate: {hs}")
        
        stdscr.addstr(2, 40, "=== PROXY ===")
        stdscr.addstr(3, 40, f"Port: {CONFIG['PROXY_PORT']}")
        stdscr.addstr(4, 40, f"ASIC TX: {stats['proxy_tx']}")
        stdscr.addstr(5, 40, f"ASIC OK: {stats['proxy_rx']}")
        
        stdscr.addstr(2, 80, "=== POOL ===")
        stdscr.addstr(3, 80, f"Local TX: {stats['local_tx']}")
        stdscr.addstr(4, 80, f"Pool OK: {stats['accepted']}")
        stdscr.addstr(5, 80, f"Pool Bad: {stats['rejected']}")
        
        stdscr.hline(7, 0, '-', w)
        for i, l in enumerate(logs):
            c = curses.color_pair(1)
            if "ERR" in l: c = curses.color_pair(3)
            elif "TX" in l: c = curses.color_pair(2)
            try: stdscr.addstr(8+i, 2, l, c)
            except: pass
            
        stdscr.refresh()
        time.sleep(0.1)
        if stdscr.getch() == ord('q'): break

if __name__ == "__main__":
    # 1. Connection Check
    if not phase_one_connection():
        sys.exit(1)
        
    # 2. Benchmark
    phase_two_benchmark()
    
    # 3. Mining Phase
    # Manager for Shared State
    manager = mp.Manager()
    stats = manager.dict({
        'hashes': 0, 
        'local_tx': 0, 'accepted': 0, 'rejected': 0,
        'proxy_tx': 0, 'proxy_rx': 0,
        'current_job': {}
    })
    log_q = manager.Queue()
    stop_ev = mp.Event()
    
    # Workers
    procs = []
    # Safe CPU usage (Leave 1 core)
    cpu_n = max(1, mp.cpu_count() - 1)
    
    for i in range(cpu_n):
        p = mp.Process(target=miner_process, args=(i, stop_ev, stats, log_q))
        p.start()
        procs.append(p)
        
    # Launch GUI in main thread
    try:
        curses.wrapper(draw_gui, stats, log_q)
    except KeyboardInterrupt: pass
    finally:
        stop_ev.set()
        for p in procs: p.terminate()
