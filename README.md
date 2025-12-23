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
import subprocess  # for accurate temp

# ======================  diff_to_target ======================
def diff_to_target(diff):
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    target_int = diff1 // int(diff)
    return format(target_int, '064x')

# ======================  CPU TEMPERATURE (accurate for HiveOS/AMD) ======================
def get_cpu_temp():
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

# job data (shared)
job_id = manager.Value('c', None)
prevhash = manager.Value('c', None)
coinb1 = manager.Value('c', None)
coinb2 = manager.Value('c', None)
merkle_branch = manager.list()
version = manager.Value('c', None)
nbits = manager.Value('c', None)
ntime = manager.Value('c', None)
extranonce1 = manager.Value('c', "00000000")
extranonce2 = manager.Value('c', "00000000")
extranonce2_size = manager.Value('i', 4)
target = manager.Value('c', None)
pool_diff = manager.Value('i', 1)  # low for maximum submits

# Global log lines for display
log_lines = manager.list()
max_log = 40

# Connection status (shared)
connected = manager.Value('b', False)

# Last error time
last_error_time = manager.Value('d', 0)

# ======================  LOGGER (LV06 style with ₿ timestamp) ======================
def logg(msg):
    timestamp = int(time.time() * 100000)
    prefixed_msg = f"₿ ({timestamp}) {msg}"
    log_lines.append(prefixed_msg)

logg("Miner starting...")

# ======================  CONFIG ======================
BRAIINS_HOST = 'stratum.braiins.com'
BRAIINS_PORT = 3333

num_cores = os.cpu_count()
num_processes = num_cores * 2  # heavy load for high hashrate / ~45 load average

# ======================  SIGNAL ======================
def signal_handler(sig, frame):
    fShutdown.set()
    logg("\n[!] Shutting down...")

signal.signal(signal.SIGINT, signal_handler)

# ======================  MERKLE ROOT ======================
def calculate_merkle_root():
    coinbase = coinb1.value + extranonce1.value + extranonce2.value + coinb2.value
    h = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()
    for b in merkle_branch:
        h = hashlib.sha256(hashlib.sha256(h + binascii.unhexlify(b)).digest()).digest()
    return binascii.hexlify(h).decode()[::-1]

# ======================  SUBMIT SHARE (LV06 style logs, skip if disconnected) ======================
def submit_share(nonce):
    if not connected.value:
        return  # skip completely if disconnected

    payload = {
        "id": 1,
        "method": "mining.submit",
        "params": [user, job_id.value, extranonce2.value, ntime.value, f"{nonce:08x}"]
    }
    try:
        logg(f"stratum_api: tx: {json.dumps(payload)}")
        sock.sendall((json.dumps(payload) + "\n").encode())
        resp = sock.recv(1024).decode().strip()
        logg(f"stratum_task: rx: {resp}")
        logg("stratum_task: message result accepted" if "true" in resp.lower() else "[!] Share rejected")

        if "true" in resp.lower():
            accepted.value += 1
            accepted_timestamps.append(time.time())
            log_lines.append("*** SHARE ACCEPTED ***")
            log_lines.append(f"Nonce: {nonce:08x}")
            log_lines.append(f"Time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
            curses.beep()
        else:
            rejected.value += 1
            rejected_timestamps.append(time.time())
    except BrokenPipeError:
        connected.value = False
        current_time = time.time()
        if current_time - last_error_time.value > 10:
            logg("[!] Broken pipe – connection lost")
            last_error_time.value = current_time
    except Exception as e:
        current_time = time.time()
        if current_time - last_error_time.value > 10:
            logg(f"[!] Submit failed: {e}")
            last_error_time.value = current_time

# ======================  MINING PROCESS (optimized for high hashrate) ======================
def bitcoin_miner_process(process_id):
    hashes_done = 0
    last_report = time.time()

    # Initial values
    header_bytes = b''
    nonce = 0

    while not fShutdown.is_set():
        if job_id.value is None:
            time.sleep(0.5)
            continue

        # Rebuild header on new job
        if hashes_done == 0 or job_id.value != last_job_id:
            last_job_id = job_id.value

            logg(f"create_jobs_task: New Work Dequeued {job_id.value}")

            # Reset extranonce2
            extranonce2.value = "00" * extranonce2_size.value

            # Shuffle nonce
            nonce = random.randint(0, 0xffffffff)

            header_static = version.value + prevhash.value + coinb1.value + extranonce1.value + extranonce2.value + coinb2.value + ntime.value + nbits.value
            header_bytes = binascii.unhexlify(header_static)

            # Network target
            network_target = (nbits.value[2:] + '00' * (int(nbits.value[:2],16) - 3)).zfill(64)

            # Very low local target for maximum submits
            share_target = diff_to_target(1)

        # Very large batch for max hashrate
        for _ in range(2000000):
            nonce = (nonce - 1) & 0xffffffff
            h = hashlib.sha256(hashlib.sha256(header_bytes + nonce.to_bytes(4,'little')).digest()).digest()
            h_hex = binascii.hexlify(h[::-1]).decode()

            hashes_done += 1

            if h_hex < share_target:
                diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
                share_diff = diff1 / int(h_hex, 16)
                logg(f"asic_result: Nonce difficulty {share_diff:.2f} of 1.")
                is_block = h_hex < network_target
                submit_share(nonce)
                if is_block:
                    log_lines.append("*** BLOCK SOLVED!!! ***")
                    log_lines.append(f"Nonce: {nonce:08x}")
                    log_lines.append(f"Time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    curses.flash()
                    curses.beep()
                    curses.beep()
                    curses.beep()
                    curses.beep()
                    curses.beep()

            if hashes_done % 200000 == 0:
                now = time.time()
                elapsed = now - last_report
                if elapsed > 0:
                    hr = int(200000 / elapsed)
                    hashrates[process_id] = hr
                last_report = now

        # Increment extranonce2 on wrap
        extranonce2_int = int(extranonce2.value, 16)
        extranonce2_int += 1
        extranonce2.value = f"{extranonce2_int:0{extranonce2_size.value*2}x}"

# ======================  STRATUM (LV06 style logs, reconnection) ======================
def stratum_worker():
    global sock, job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, target, extranonce1, extranonce2_size, pool_diff, connected

    while not fShutdown.is_set():
        try:
            s = socket.socket()
            s.settimeout(30)
            s.connect((host, port))
            sock = s
            connected.value = True
            logg(f"Connected to {host}:{port}")

            s.sendall(b'{"id":1,"method":"mining.subscribe","params":[]}\n')

            auth = {"id":2,"method":"mining.authorize","params":[user,password]}
            s.sendall((json.dumps(auth)+"\n").encode())

            buf = b""
            while not fShutdown.is_set():
                data = s.recv(4096)
                if not data:
                    connected.value = False
                    logg("[!] Connection lost – reconnecting...")
                    break
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    if not line.strip(): continue
                    msg = json.loads(line)
                    logg(f"stratum_task: rx: {json.dumps(msg)}")
                    if "result" in msg and msg["id"] == 1:
                        extranonce1.value = msg["result"][1]
                        extranonce2_size.value = msg["result"][2]
                        logg(f"Subscribed – extranonce1: {extranonce1.value}, size: {extranonce2_size.value}")
                    elif msg.get("method") == "mining.notify":
                        params = msg["params"]
                        job_id.value = params[0]
                        prevhash.value = params[1]
                        coinb1.value = params[2]
                        coinb2.value = params[3]
                        merkle_branch[:] = params[4]
                        version.value = params[5]
                        nbits.value = params[6]
                        ntime.value = params[7]
                        logg(f"create_jobs_task: New Work Dequeued {job_id.value}")
                    elif msg.get("method") == "mining.set_difficulty":
                        pool_diff.value = msg["params"][0]
                        target.value = diff_to_target(pool_diff.value)
                        logg(f"[*] Difficulty set to {pool_diff.value}")
        except socket.timeout:
            connected.value = False
            logg("[!] Timeout – reconnecting...")
        except Exception as e:
            connected.value = False
            logg(f"[!] Stratum error: {e} – reconnecting in 10s...")
            time.sleep(10)

# ======================  DISPLAY (stable top bar, scrolling logs) ======================
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
            title = "Bitcoin Miner (CPU) - Braiins Pool"
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

            # Connection status
            status = "Connected" if connected.value else "Disconnected"
            stdscr.addstr(10, 0, f"Status       : {status}", curses.color_pair(1 if connected.value else 2))

            # Yellow line
            stdscr.addstr(12, 0, "─" * (screen_width - 1), curses.color_pair(3))

            # Scrolling log area (stable)
            start_y = 13
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
    parser = argparse.ArgumentParser(description="Braiins Pool SHA-256 CPU Miner")
    parser.add_argument("--username", type=str, required=True, help="Braiins username or payout address")
    parser.add_argument("--worker", type=str, default="cpu002", help="Worker name")
    args = parser.parse_args()

    host = BRAIINS_HOST
    port = BRAIINS_PORT
    user = f"{args.username}.{args.worker}"
    password = "x"

    # Initialize shared hashrates list
    for _ in range(num_processes):
        hashrates.append(0)

    # Start stratum
    p_stratum = multiprocessing.Process(target=stratum_worker, daemon=True)
    p_stratum.start()
    time.sleep(5)

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
