#!/usr/bin/env python3

import multiprocessing
import requests
import binascii
import hashlib
import logging
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

# ======================  diff_to_target ======================
def diff_to_target(diff):
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    target_int = diff1 // int(diff)
    return format(target_int, '064x')

# ======================  CPU TEMPERATURE (improved for HiveOS) ======================
def get_cpu_temp():
    # HiveOS often uses sensors command or different zones
    try:
        # Try 'sensors' command if available
        import subprocess
        result = subprocess.check_output(["sensors"], text=True)
        temps = []
        for line in result.splitlines():
            if 'Core' in line or 'Tdie' in line or 'Tctl' in line:
                match = re.search(r'\+([\d\.]+)°C', line)
                if match:
                    temps.append(float(match.group(1)))
        if temps:
            avg = sum(temps) / len(temps)
            max_temp = max(temps)
            return f"{avg:.1f}°C (avg) / {max_temp:.1f}°C (max)"
    except:
        pass

    # Fallback to /sys thermal zones
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

# ======================  CKPOOL STATS ======================
def get_ckpool_stats():
    url = "https://solostats.ckpool.org/users/bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e"
    try:
        r = requests.get(url, timeout=10)
        text = r.text
        hashrate = re.search(r'Hashrate</td><td>([^<]+)', text)
        last_share = re.search(r'Last Share</td><td>([^<]+)', text)
        best_share = re.search(r'Best Share</td><td>([^<]+)', text)
        shares = re.search(r'Shares</td><td>([^<]+)', text)

        return {
            "hashrate": hashrate.group(1).strip() if hashrate else "N/A",
            "last_share": last_share.group(1).strip() if last_share else "N/A",
            "best_share": best_share.group(1).strip() if best_share else "N/A",
            "shares": shares.group(1).strip() if shares else "N/A"
        }
    except:
        return {"hashrate": "N/A", "last_share": "N/A", "best_share": "N/A", "shares": "N/A"}

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
mode = "solo"
host = port = user = password = None

# ======================  LOGGER ======================
def logg(msg):
    print(msg)
    try:
        logging.basicConfig(level=logging.INFO, filename="miner.log",
                            format='%(asctime)s %(message)s', force=True)
        logging.info(msg)
    except:
        pass

logg("[*] Miner starting...")

# ======================  CONFIG ======================
SOLO_HOST = 'solo.ckpool.org'
SOLO_PORT = 3333
SOLO_ADDRESS = 'bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.001'

POOL_HOST = 'ss.antpool.com'
POOL_PORT = 3333
POOL_WORKER = 'Xk2000.001'
POOL_PASSWORD = 'x'

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
        sock.sendall((json.dumps(payload) + "\n").encode())
        resp = sock.recv(1024).decode().strip()
        logg(f"[+] Share submitted → {resp}")

        if "true" in resp.lower():
            with accepted.get_lock():
                accepted.value += 1
            accepted_timestamps.append(time.time())
            print("\n" + "="*60)
            print("*** SHARE ACCEPTED ***")
            print(f"Nonce: {nonce:08x}")
            print(f"Time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60 + "\n")
            print("\a", end="", flush=True)
        else:
            with rejected.get_lock():
                rejected.value += 1
            rejected_timestamps.append(time.time())
            logg("[!] Share rejected")
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

    # Share target for reporting hashrate (low in solo mode)
    share_target = diff_to_target(4096) if mode == "solo" else target

    nonce = 0xffffffff
    hashes_done = 0
    last_report = time.time()

    while not fShutdown.is_set():
        # Larger batch for better efficiency and faster hashrate reporting
        for _ in range(500000):
            nonce = (nonce - 1) & 0xffffffff
            h = hashlib.sha256(hashlib.sha256(header_bytes + nonce.to_bytes(4,'little')).digest()).digest()
            h_hex = binascii.hexlify(h[::-1]).decode()

            hashes_done += 1

            # Submit if meets share target
            if h_hex < share_target:
                is_block = h_hex < network_target
                submit_share(nonce)
                if is_block:
                    print("\n" + "="*80)
                    print("█" + " "*78 + "█")
                    print("█" + " "*28 + "BLOCK SOLVED!!!" + " "*33 + "█")
                    print("█" + " "*78 + "█")
                    print(f"█  Nonce      : {nonce:08x}")
                    print(f"█  Time      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print("█" + " "*78 + "█")
                    print("="*80 + "\n")
                    print("\a" * 5, end="", flush=True)  # loud beep for block

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

    s = socket.socket()
    s.connect((host, port))
    sock = s

    s.sendall(b'{"id":1,"method":"mining.subscribe","params":[]}\n')
    lines = s.recv(4096).decode().split('\n')
    response = json.loads(lines[0])
    extranonce1 = response['result'][1]
    extranonce2_size = response['result'][2]
    logg(f"[*] Subscribed – extranonce1: {extranonce1}, size: {extranonce2_size}")

    auth = {"id":2,"method":"mining.authorize","params":[user,password]}
    s.sendall((json.dumps(auth)+"\n").encode())

    buf = b""
    while not fShutdown.is_set():
        try:
            data = s.recv(4096)
            if not data: break
            buf += data
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                if not line.strip(): continue
                msg = json.loads(line)
                if msg.get("method") == "mining.notify":
                    (job_id, prevhash, coinb1, coinb2,
                     merkle_branch, version, nbits, ntime, _) = msg["params"]
                    logg(f"[*] New job #{job_id} | Prevhash: {prevhash}")
                elif msg.get("method") == "mining.set_difficulty":
                    target = diff_to_target(msg["params"][0])
                    logg(f"[*] Difficulty set to {msg['params'][0]}")
        except Exception as e:
            logg(f"[!] Stratum error: {e}")
            break

# ======================  DISPLAY ======================
def display_worker():
    stdscr = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN,  curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED,    curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN,   curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.noecho(); curses.cbreak(); stdscr.keypad(True)

    log_lines = []
    max_log = 20

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
            title = f"Bitcoin {mode.upper()} Miner (CPU)"
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

            # ckpool stats
            stats = get_ckpool_stats()
            stdscr.addstr(11, 0, "─" * (screen_width - 1), curses.color_pair(3))
            stdscr.addstr(12, 0, f"ckpool Hashrate : {stats['hashrate']}", curses.color_pair(1))
            stdscr.addstr(13, 0, f"Last Share      : {stats['last_share']}", curses.color_pair(3))
            stdscr.addstr(14, 0, f"Best Share      : {stats['best_share']}", curses.color_pair(1))
            stdscr.addstr(15, 0, f"Total Shares    : {stats['shares']}", curses.color_pair(3))

            # Log area (scrolling, stable)
            start_y = 17
            for i, line in enumerate(log_lines[-max_log:]):
                if start_y + i >= screen_height:
                    break
                stdscr.addstr(start_y + i, 0, line[:screen_width-1], curses.color_pair(5))

            stdscr.refresh()
            time.sleep(1)
    finally:
        curses.endwin()

# Override print to capture logs for display (stable list)
original_print = print
def custom_print(*args, **kwargs):
    original_print(*args, **kwargs)
    msg = " ".join(str(a) for a in args)
    log_lines.append(msg)

print = custom_print

# ======================  MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["solo","pool"], default="solo")
    args = parser.parse_args()

    mode = args.mode
    if mode == "solo":
        host, port, user, password = SOLO_HOST, SOLO_PORT, SOLO_ADDRESS, "x"
    else:
        host, port, user, password = POOL_HOST, POOL_PORT, POOL_WORKER, POOL_PASSWORD

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
        p.join()
    p_stratum.join()
    logg("[*] Shutdown complete")
