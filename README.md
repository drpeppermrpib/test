#!/usr/bin/env python3

import multiprocessing
import requests
import binascii
import hashlib
import random
import socket
import time
import json
import sys
import os
import curses
import argparse
import signal
import threading
import subprocess  # for accurate temp

# ======================  diff_to_target ======================
def diff_to_target(diff):
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    target_int = diff1 // int(diff)
    return format(target_int, '064x')

# ======================  CPU TEMPERATURE (accurate for HiveOS/AMD) ======================
def get_cpu_temp():
    # HiveOS/AMD: use 'sensors' for Tctl/Tdie
    try:
        result = subprocess.check_output(["sensors"], text=True)
        temps = []
        for line in result.splitlines():
            if 'Tctl' in line or 'Tdie' in line or 'edge' in line or 'junction' in line:
                match = re.search(r'[\+\-]?[\d\.]+°C', line)
                if match:
                    temps.append(float(match.group(0).replace('°C', '').strip()))
        if temps:
            avg = sum(temps) / len(temps)
            max_temp = max(temps)
            return f"{avg:.1f}°C (avg) / {max_temp:.1f}°C (max)"
    except:
        pass

    # Fallback thermal zones
    temps = []
    for zone in range(20):
        path = f"/sys/class/thermal/thermal_zone{zone}/temp"
        if os.path.exists(path):
            try:
                with open(path) as f:
                    temp = int(f.read().strip()) / 1000
                    temps.append(temp)
            except:
                pass
    if temps:
        avg = sum(temps) / len(temps)
        max_temp = max(temps)
        return f"{avg:.1f}°C (avg) / {max_temp:.1f}°C (max)"
    return "N/A"

# ======================  GLOBALS (shared via Manager) ======================
manager = multiprocessing.Manager()
fShutdown = manager.Event()
hashrates = manager.list()
accepted = manager.Value('i', 0)
rejected = manager.Value('i', 0)
accepted_timestamps = manager.list()
rejected_timestamps = manager.list()

# job data
job_id = prevhash = coinb1 = coinb2 = None
merkle_branch = version = nbits = ntime = None
extranonce1 = extranonce2 = extranonce2_size = None
sock = None
target = None
host = port = user = password = None

# Global log lines for display
log_lines = []
max_log = 15

# ======================  LOGGER ======================
def logg(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()
    log_lines.append(msg)
    try:
        logging.basicConfig(level=logging.INFO, filename="miner.log",
                            format='%(asctime)s %(message)s', force=True)
        logging.info(msg)
    except:
        pass

logg("[*] Miner starting...")

# ======================  CONFIG ======================
PROHASHING_HOST = 'us.mining.prohashing.com'
PROHASHING_PORT = 3335

num_cores = os.cpu_count()
num_processes = num_cores  # Use all physical cores for true scaling

# ======================  SIGNAL ======================
def signal_handler(sig, frame):
    fShutdown.set()
    logg("\n[!] Shutting down...")

signal.signal(signal.SIGINT, signal_handler)

# ======================  MERKLE ROOT ======================
def calculate_merkle_root():
    if None in (coinb1, extranonce1, extranonce2, coinb2, merkle_branch):
        return "0" * 64
    coinbase = coinb1 + extranonce1 + extranonce2 + coinb2
    h = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()
    for b in merkle_branch:
        h = hashlib.sha256(hashlib.sha256(h + binascii.unhexlify(b)).digest()).digest()
    return binascii.hexlify(h).decode()[::-1]

# ======================  SUBMIT SHARE ======================
def submit_share(nonce):
    payload = {
        "id": 1,
        "method": "mining.submit",
        "params": [user, job_id, extranonce2, ntime, f"{nonce:08x}"]
    }
    try:
        logg(f"stratum_api: tx: {json.dumps(payload)}")
        sock.sendall((json.dumps(payload) + "\n").encode())
        resp = sock.recv(1024).decode().strip()
        logg(f"stratum_task: rx: {resp}")
        logg("stratum_task: message result accepted" if "true" in resp.lower() else "[!] Share rejected")

        if "true" in resp.lower():
            with accepted.get_lock():
                accepted.value += 1
            accepted_timestamps.append(time.time())
            sys.stdout.write("\n" + "="*60 + "\n")
            sys.stdout.write("*** SHARE ACCEPTED ***\n")
            sys.stdout.write(f"Nonce: {nonce:08x}\n")
            sys.stdout.write(f"Time : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            sys.stdout.write("="*60 + "\n")
            sys.stdout.flush()
            sys.stdout.write("\a")
            sys.stdout.flush()
        else:
            with rejected.get_lock():
                rejected.value += 1
            rejected_timestamps.append(time.time())
    except BrokenPipeError:
        logg("[!] Broken pipe – connection lost")
    except Exception as e:
        logg(f"[!] Submit failed: {e}")

# ======================  MINING PROCESS ======================
def bitcoin_miner_process(process_id):
    global nbits, version, prevhash, ntime, target

    # Wait for job
    while None in (nbits, version, prevhash, ntime):
        time.sleep(0.5)

    header_static = version + prevhash + calculate_merkle_root() + ntime + nbits
    header_bytes = binascii.unhexlify(header_static)

    # Network (block) target
    network_target = (nbits[2:] + '00' * (int(nbits[:2],16) - 3)).zfill(64)

    # Share target from pool (ProHashing sets reasonable diff)
    share_target = target if target else diff_to_target(4096)

    nonce = 0xffffffff
    hashes_done = 0
    last_report = time.time()

    while not fShutdown.is_set():
        for _ in range(500000):
            nonce = (nonce - 1) & 0xffffffff
            h = hashlib.sha256(hashlib.sha256(header_bytes + nonce.to_bytes(4,'little')).digest()).digest()
            h_hex = binascii.hexlify(h[::-1]).decode()

            hashes_done += 1

            if h_hex < share_target:
                is_block = h_hex < network_target
                submit_share(nonce)
                if is_block:
                    sys.stdout.write("\n" + "="*80 + "\n")
                    sys.stdout.write("█" + " "*78 + "█\n")
                    sys.stdout.write("█" + " "*28 + "BLOCK SOLVED!!!" + " "*33 + "█\n")
                    sys.stdout.write("█" + " "*78 + "█\n")
                    sys.stdout.write(f"█  Nonce      : {nonce:08x}\n")
                    sys.stdout.write(f"█  Time      : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    sys.stdout.write("█" + " "*78 + "█\n")
                    sys.stdout.write("="*80 + "\n")
                    sys.stdout.flush()
                    sys.stdout.write("\a" * 5)
                    sys.stdout.flush()

            if hashes_done % 500000 == 0:
                now = time.time()
                elapsed = now - last_report
                if elapsed > 0:
                    hr = int(500000 / elapsed)
                    hashrates[process_id] = hr
                last_report = now

# ======================  STRATUM ======================
def stratum_worker():
    global sock, job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce1, extranonce2_size

    while not fShutdown.is_set():
        try:
            s = socket.socket()
            s.connect((host, port))
            sock = s

            s.sendall(b'{"id":1,"method":"mining.subscribe","params":[]}\n')

            auth = {"id":2,"method":"mining.authorize","params":[user,password]}
            s.sendall((json.dumps(auth)+"\n").encode())

            buf = b""
            while not fShutdown.is_set():
                data = s.recv(4096)
                if not data:
                    logg("[!] Connection lost – reconnecting...")
                    break
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    if not line.strip(): continue
                    msg = json.loads(line)
                    logg(f"stratum_task: rx: {json.dumps(msg)}")
                    if "result" in msg and msg["id"] == 1:
                        extranonce1 = msg["result"][1]
                        extranonce2_size = msg["result"][2]
                        logg(f"Subscribed – extranonce1: {extranonce1}, size: {extranonce2_size}")
                    elif msg.get("method") == "mining.notify":
                        (job_id, prevhash, coinb1, coinb2,
                         merkle_branch, version, nbits, ntime, _) = msg["params"]
                        logg(f"create_jobs_task: New Work Dequeued {job_id}")
                    elif msg.get("method") == "mining.set_difficulty":
                        target = diff_to_target(msg["params"][0])
                        logg(f"[*] Difficulty set to {msg['params'][0]}")
        except Exception as e:
            logg(f"[!] Stratum error: {e} – reconnecting in 10s...")
            time.sleep(10)

# ======================  DISPLAY ======================
def display_worker():
    global log_lines
    stdscr = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN,  curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED,    curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN,   curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.noecho(); curses.cbreak(); stdscr.keypad(True)

    try:
        while not fShutdown.is_set():
            stdscr.clear()
            screen_height, screen_width = stdscr.getmaxyx()

            now = time.time()
            a_min = sum(1 for t in accepted_timestamps if now-t<60)
            r_min = sum(1 for t in rejected_timestamps if now-t<60)

            cpu_temp = get_cpu_temp()

            # Top right: Ctrl+C to quit
            stdscr.addstr(0, max(0, screen_width - 20), "Ctrl+C to quit", curses.color_pair(3))

            # Title
            title = "Bitcoin Miner (CPU) - ProHashing"
            stdscr.addstr(2, 0, title, curses.color_pair(4)|curses.A_BOLD)

            # Static stats
            try:
                block_height = requests.get('https://blockchain.info/q/getblockcount',timeout=3).text
            except:
                block_height = "???"
            stdscr.addstr(4, 0, f"Block height : ~{block_height}", curses.color_pair(3))
            stdscr.addstr(5, 0, f"Hashrate     : {sum(hashrates):,} H/s", curses.color_pair(1))
            stdscr.addstr(6, 0, f"CPU Temp     : {cpu_temp}", curses.color_pair(3))
            stdscr.addstr(7, 0, f"Processes    : {num_processes}", curses.color_pair(3))
            stdscr.addstr(8, 0, f"Shares       : {accepted.value} accepted / {rejected.value} rejected")
            stdscr.addstr(9, 0, f"Last minute  : {a_min} acc / {r_min} rej")

            # Yellow line
            stdscr.addstr(11, 0, "─" * (screen_width - 1), curses.color_pair(3))

            # Scrolling log area (stable)
            start_y = 12
            for i, line in enumerate(log_lines[-max_log:]):
                if start_y + i >= screen_height:
                    break
                stdscr.addstr(start_y + i, 0, line[:screen_width-1], curses.color_pair(5))

            stdscr.refresh()
            time.sleep(1)
    finally:
        curses.endwin()

# ======================  MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProHashing SHA-256 CPU Miner")
    parser.add_argument("--username", type=str, required=True, help="ProHashing username")
    parser.add_argument("--worker", type=str, default="001", help="Worker name (e.g., 001)")
    parser.add_argument("--group", type=str, default="kx2000", help="Group name (e.g., kx2000)")
    parser.add_argument("--wattage", type=int, default=1000, help="Wattage for tracking (e.g., 1000)")
    parser.add_argument("--price", type=float, default=1.45, help="Electricity price $/kWh (e.g., 1.45)")
    args = parser.parse_args()

    # Construct ProHashing password string
    password = f"a=sha-256,c=bitcoin,w={args.wattage},p={args.price},n={args.worker},o={args.group}"

    host = PROHASHING_HOST
    port = PROHASHING_PORT
    user = args.username
    password = password

    # Initialize shared hashrates list
    for _ in range(num_processes):
        hashrates.append(0)

    # Start stratum
    p_stratum = multiprocessing.Process(target=stratum_worker, daemon=True)
    p_stratum.start()
    time.sleep(3)

    # Start mining processes
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=bitcoin_miner_process, args=(i,))
        p.start()
        processes.append(p)

    # Display
    threading.Thread(target=display_worker, daemon=True).start()

    logg("[*] Miner running – press Ctrl+C to stop")
    try:
        while not fShutdown.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    fShutdown.set()
    for p in processes:
        p.terminate()
        p.join()
    p_stratum.terminate()
    p_stratum.join()
    logg("[*] Shutdown complete")
