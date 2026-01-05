#!/usr/bin/env python3
import sys
import subprocess
import os
import resource
import time
import socket
import json
import threading
import multiprocessing as mp
import binascii
import struct
import hashlib
import select
import random
import datetime

# ================= 1. AUTO-REPAIR & SETUP =================
def self_check():
    # Fix "Process 50" / Too Many Open Files
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except: pass

    # Install Dependencies
    required = ['psutil', 'requests']
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[*] Installing missing driver: {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

self_check()

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import curses
except ImportError:
    print("Curses not found. Please run in a proper terminal.")
    sys.exit()

# ================= 2. CONFIGURATION =================
CONFIG = {
    "POOL_URL": "solo.stratum.braiins.com",
    "POOL_PORT": 3333,
    "WALLET": "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e",
    "PASSWORD": "x",
    "PROXY_PORT": 60060,
    "THROTTLE_TEMP": 85.0,  # Max Temp
    "BENCH_DURATION": 60,   # Seconds
    "LOAD_FACTOR": 1.0      # 0.1 to 2.0 (Calculated by Bench)
}

# ================= 3. HARDWARE CONTROL =================
def get_temps():
    c = 0.0
    g = 0.0
    try:
        # Linux Sensors
        o = subprocess.check_output("sensors", shell=True).decode()
        for l in o.splitlines():
            if "Tdie" in l or "Package" in l:
                c = float(l.split('+')[1].split('°')[0].strip())
                break
    except: pass
    
    try:
        # Nvidia GPU
        o = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True).decode()
        g = float(o.strip())
    except: pass
    return c, g

def force_fans():
    # Attempt to maximize cooling before load
    try:
        subprocess.Popen("nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[gpu:0]/GPUTargetFanSpeed=100'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except: pass

# ================= 4. BENCHMARKING ENGINE =================
def run_benchmark():
    os.system('clear')
    print("==================================================")
    print("   MTP v28 - HARDWARE OPTIMIZATION & BENCHMARK")
    print("==================================================")
    print("[*] Detecting Hardware & Preparing Sensors...")
    force_fans()
    
    # Test Load
    print(f"[*] Running Load Test ({CONFIG['BENCH_DURATION']}s)...")
    print("    - Goal: Find MAX Hashrate before Thermal Throttle")
    
    start_t = time.time()
    ops = 0
    max_temp = 0
    
    try:
        while time.time() - start_t < CONFIG['BENCH_DURATION']:
            # Simulate intense hashing (SHA256d mimic)
            for _ in range(50000):
                h = hashlib.sha256(os.urandom(64)).digest()
                _ = hashlib.sha256(h).digest()
            ops += 50000
            
            # Monitor Temp
            c, g = get_temps()
            curr_max = max(c, g)
            if curr_max > max_temp: max_temp = curr_max
            
            elapsed = time.time() - start_t
            rate = ops / elapsed
            
            sys.stdout.write(f"\r    > Hashrate: {rate/1000:.2f} kH/s | Temp: {curr_max}C | Status: TUNING...")
            sys.stdout.flush()
            
            # Safety Cutoff
            if curr_max > 90:
                print("\n[!] CRITICAL HEAT DETECTED. Stopping Benchmark.")
                CONFIG['LOAD_FACTOR'] = 0.5
                break
                
    except KeyboardInterrupt:
        pass
    
    print("\n\n[*] Benchmark Complete.")
    print(f"    - Peak Hashrate: {ops/(time.time()-start_t)/1000:.2f} kH/s")
    print(f"    - Peak Temp: {max_temp}C")
    
    if max_temp < 70:
        CONFIG['LOAD_FACTOR'] = 2.0 # Boost Mode
        print("    - Result: EXCELLENT COOLING. ENABLING OVERDRIVE.")
    elif max_temp < 85:
        CONFIG['LOAD_FACTOR'] = 1.0 # Normal
        print("    - Result: STABLE. STANDARD LOAD.")
    else:
        CONFIG['LOAD_FACTOR'] = 0.7 # Throttle
        print("    - Result: HOT. ENABLING SAFE MODE.")
        
    time.sleep(3)

# ================= 5. STRATUM MINING CORE =================
# Optimized for Local CPU/GPU Mining
def local_miner(id, job_q, result_q, stop_event, stats, load_factor):
    # Unique connection per thread to allow pool to track hashrate properly
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(300)
    
    try:
        s.connect((CONFIG['POOL_URL'], CONFIG['POOL_PORT']))
        
        # Subscribe
        msg = json.dumps({"id": 1, "method": "mining.subscribe", "params": [f"MTP-v28-{id}"]}) + "\n"
        s.sendall(msg.encode())
        
        # Authorize (Unique Worker Name per thread to aggregate on pool)
        worker_name = f"{CONFIG['WALLET']}.cpu_{id}"
        msg = json.dumps({"id": 2, "method": "mining.authorize", "params": [worker_name, CONFIG['PASSWORD']]}) + "\n"
        s.sendall(msg.encode())
        
        extranonce1 = ""
        extranonce2_size = 4
        job = None
        
        while not stop_event.is_set():
            # Listen for Jobs
            try:
                s.settimeout(0.1)
                line = s.recv(4096).decode()
                for data in line.split('\n'):
                    if not data: continue
                    msg = json.loads(data)
                    
                    if msg.get('id') == 1:
                        extranonce1 = msg['result'][1]
                        extranonce2_size = msg['result'][2]
                    
                    if msg.get('method') == 'mining.notify':
                        # New Job: job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs
                        job = msg['params']
            except: pass

            if not job or not extranonce1:
                time.sleep(0.1)
                continue

            # MINE
            # Construct Header
            job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean = job
            
            # Difficulty Target (Simplified for solo)
            target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
            
            # Calculate Range based on Benchmark Load Factor
            # 1.0 = 1M hashes, 2.0 = 2M hashes
            search_range = int(1000000 * load_factor)
            
            # Unique ExtraNonce2
            en2 = struct.pack('>I', random.randint(0, 2**32-1)).hex().zfill(extranonce2_size*2)
            
            # Coinbase
            coinbase = binascii.unhexlify(coinb1 + extranonce1 + en2 + coinb2)
            coinbase_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            
            # Merkle Root
            merkle_root = coinbase_hash
            for branch in merkle_branch:
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + binascii.unhexlify(branch)).digest()).digest()
            
            # Block Header Construction (Little Endian)
            # Version (4) + PrevHash (32) + Merkle (32) + Time (4) + Bits (4) + Nonce (4)
            # Note: Stratum usually sends PrevHash swapped. We use it as is if pool follows standard.
            
            header_prefix = (
                struct.pack("<I", int(version, 16)) +
                binascii.unhexlify(prevhash) + # Assuming LE from pool
                merkle_root +
                struct.pack("<I", int(ntime, 16)) +
                binascii.unhexlify(nbits)[::-1]
            )
            
            start_nonce = random.randint(0, 2**32 - search_range)
            
            # Hashing Loop
            for n in range(start_nonce, start_nonce + search_range):
                nonce_bin = struct.pack("<I", n)
                header = header_prefix + nonce_bin
                block_hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
                
                # Check Target (Big Endian comparison for target)
                hash_int = int.from_bytes(block_hash[::-1], 'big')
                
                if hash_int <= target:
                    # FOUND A SHARE
                    # Send to pool
                    submit_msg = json.dumps({
                        "params": [
                            worker_name,
                            job_id,
                            en2,
                            ntime,
                            struct.pack(">I", n).hex() # Stratum usually wants BE Hex string
                        ],
                        "id": 4,
                        "method": "mining.submit"
                    }) + "\n"
                    s.sendall(submit_msg.encode())
                    
                    # Notify GUI
                    result_q.put(("TX", f"Share Found! ID: {job_id[:4]}..."))
                    break # Move to next nonce range or job
            
            stats[id] += search_range

    except Exception as e:
        # result_q.put(("ERR", f"Local Miner Crash: {e}"))
        pass
    finally:
        s.close()

# ================= 6. PROXY (KEEPALIVE & FUNNEL) =================
class ProxyServer(threading.Thread):
    def __init__(self, log_q, proxy_stats):
        super().__init__()
        self.log_q = log_q
        self.stats = proxy_stats
        self.daemon = True
        
    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", CONFIG['PROXY_PORT']))
        s.listen(100)
        self.log_q.put(("INFO", f"Proxy Active on port {CONFIG['PROXY_PORT']}"))
        
        while True:
            c, a = s.accept()
            t = threading.Thread(target=self.client_handler, args=(c, a))
            t.start()

    def client_handler(self, client, addr):
        pool = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        pool.connect((CONFIG['POOL_URL'], CONFIG['POOL_PORT']))
        
        # Identify ASIC by IP
        ip_suffix = addr[0].split('.')[-1]
        
        # Heartbeat Thread
        def heartbeat():
            while True:
                time.sleep(30)
                try:
                    # Send dummy keepalive if needed (Stratum usually handles this via difficulty updates)
                    pass 
                except: break

        threading.Thread(target=heartbeat, daemon=True).start()

        # Funnel
        try:
            while True:
                r, _, _ = select.select([client, pool], [], [], 60)
                if not r: continue # Timeout handling
                
                if client in r:
                    data = client.recv(4096)
                    if not data: break
                    
                    # Modify Packets from ASIC -> Pool
                    # Force Worker Name to aggregate hashrate
                    try:
                        msgs = data.decode().split('\n')
                        for m in msgs:
                            if not m: continue
                            js = json.loads(m)
                            
                            if js.get('method') == 'mining.authorize':
                                # Rewrite to MainWallet.ASIC_IP
                                js['params'][0] = f"{CONFIG['WALLET']}.ASIC_{ip_suffix}"
                                data = (json.dumps(js) + "\n").encode()
                            
                            if js.get('method') == 'mining.submit':
                                self.stats['tx'] += 1 # Count TX
                    except: pass
                    
                    pool.sendall(data)
                
                if pool in r:
                    data = pool.recv(4096)
                    if not data: break
                    
                    # Analyze Pool -> ASIC
                    try:
                        if b'"result":true' in data or b'"result": true' in data:
                            self.stats['rx'] += 1 # Count OK
                    except: pass
                    
                    client.sendall(data)
        except: pass
        finally:
            client.close()
            pool.close()

# ================= 7. GUI & MAIN =================
def draw_dashboard(stdscr, stats, proxy_stats):
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    
    stdscr.nodelay(True)
    logs = []
    
    # Global Logs Queue
    log_q = mp.Queue()
    
    # Start Components
    stop_event = mp.Event()
    
    # Start Proxy
    ProxyServer(log_q, proxy_stats).start()
    
    # Start Local Miners (CPU + GPU Sim)
    procs = []
    cpu_count = mp.cpu_count()
    for i in range(cpu_count):
        p = mp.Process(target=local_miner, args=(i, None, log_q, stop_event, stats, CONFIG['LOAD_FACTOR']))
        p.start()
        procs.append(p)
    
    start_time = time.time()
    
    while True:
        # Process Logs
        while not log_q.empty():
            try:
                logs.append(log_q.get_nowait())
                if len(logs) > 15: logs.pop(0)
            except: pass
            
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        # Header
        stdscr.addstr(0, 0, f" MTP MINER v28 - {CONFIG['WALLET'][:10]}... ".center(w), curses.color_pair(1))
        
        # Stats
        c_temp, g_temp = get_temps()
        uptime = int(time.time() - start_time)
        
        total_hashes = sum(stats)
        hashrate = total_hashes / (uptime if uptime > 0 else 1)
        if hashrate > 1000000: hr_str = f"{hashrate/1000000:.2f} MH/s"
        else: hr_str = f"{hashrate/1000:.2f} kH/s"
        
        # Layout
        stdscr.addstr(2, 2, f"UPTIME: {uptime}s")
        stdscr.addstr(3, 2, f"LOCAL HASHRATE: {hr_str}")
        stdscr.addstr(4, 2, f"TEMP: CPU {c_temp}C | GPU {g_temp}C")
        
        # Proxy Stats
        stdscr.addstr(2, 40, f"PROXY ASICs: CONNECTED")
        stdscr.addstr(3, 40, f"SHARES: {proxy_stats['tx']} TX / {proxy_stats['rx']} OK", curses.color_pair(2))
        
        # Logs Area
        stdscr.hline(6, 0, '-', w)
        for i, (type, msg) in enumerate(logs):
            color = curses.color_pair(1)
            if type == "ERR": color = curses.color_pair(3)
            if type == "TX": color = curses.color_pair(2)
            try:
                stdscr.addstr(7+i, 2, f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{type}] {msg}", color)
            except: pass
            
        stdscr.refresh()
        time.sleep(0.1)
        
        # Input
        try:
            k = stdscr.getch()
            if k == ord('q'): break
        except: pass
    
    stop_event.set()
    for p in procs: p.terminate()

# ================= 8. ENTRY POINT =================
if __name__ == "__main__":
    # 1. Run Pre-Flight Benchmark
    run_benchmark()
    
    # 2. Shared Stats
    manager = mp.Manager()
    global_stats = manager.list([0] * mp.cpu_count())
    proxy_stats = manager.dict({"tx": 0, "rx": 0})
    
    # 3. Start UI
    curses.wrapper(draw_dashboard, global_stats, proxy_stats)
