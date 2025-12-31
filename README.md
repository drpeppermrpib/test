#!/usr/bin/env python3

import multiprocessing as mp
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
import subprocess
import struct
from multiprocessing import Process, Queue, Value as mpValue, Array as mpArray

# ======================  diff_to_target ======================
def diff_to_target(diff):
    diff1 = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    target_int = diff1 // int(diff)
    return format(target_int, '064x')

# ======================  CPU TEMPERATURE ======================
def get_cpu_temp():
    try:
        result = subprocess.check_output(["sensors"], text=True)
        for line in result.splitlines():
            if 'Tctl' in line or 'Tdie' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    temp = parts[1].strip().split(' ')[0]
                    return f"{temp}"
    except:
        pass

    try:
        for hwmon in os.listdir('/sys/class/hwmon'):
            name_path = f"/sys/class/hwmon/{hwmon}/name"
            if os.path.exists(name_path):
                with open(name_path) as f:
                    name = f.read().strip()
                if name in ['k10temp', 'zenpower']:
                    temp_path = f"/sys/class/hwmon/{hwmon}/temp1_input"
                    if os.path.exists(temp_path):
                        with open(temp_path) as f:
                            temp = int(f.read().strip()) / 1000
                        return f"{temp:.1f}°C"
    except:
        pass

    return "N/A"

# ======================  FETCH LIVE DATA ======================
def fetch_live_data():
    try:
        hr_data = requests.get("https://api.blockchain.info/charts/hash-rate?timespan=1days&format=json", timeout=5).json()
        network_hr = f"{hr_data['values'][-1]['y'] / 1e6:.2f} EH/s" if hr_data else "N/A"

        diff_data = requests.get("https://api.blockchain.info/charts/difficulty?timespan=1days&format=json", timeout=5).json()
        difficulty = f"{diff_data['values'][-1]['y'] / 1e12:.2f} T" if diff_data else "N/A"

        fees_data = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=5).json()
        block_fees = f"{fees_data['hourFee']} SAT/vB" if fees_data else "N/A"

        return [
            f"Network HR   : {network_hr}",
            "Avg 30d HR   : N/A",
            f"Difficulty   : {difficulty}",
            f"Block Fees   : {block_fees}",
            "Hash Value   : ~0.0004 BTC/PH/Day",
            "Hash Price   : ~$37/PH/Day",
            "Profit Ex.   : N/A"
        ]
    except Exception:
        return ["Live Data: Offline"] * 7

LIVE_DATA = fetch_live_data()

# ======================  MINING WORKER PROCESS ======================
def mining_worker(thread_id, job_queue, result_queue, shutdown_flag, hashrate_array, log_queue, pool_diff_shared):
    last_job = None
    hashes_done = 0
    last_report = time.time()

    while not shutdown_flag.value:
        try:
            job = job_queue.get(timeout=1)
        except:
            continue

        job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime = job

        if job_id != last_job:
            last_job = job_id
            log_queue.put(f"Process {thread_id}: New job {job_id}")

            extranonce2_int = (thread_id << 24) | (int(time.time() * 1000) & 0xFFFFFF)
            extranonce2 = f"{extranonce2_int:08x}"

        header_prefix = (
            binascii.unhexlify(version)[::-1] +
            binascii.unhexlify(prevhash)[::-1] +
            binascii.unhexlify(calculate_merkle_root(coinb1, coinb2, extranonce2, merkle_branch))[::-1] +
            binascii.unhexlify(ntime)[::-1] +
            binascii.unhexlify(nbits)[::-1]
        )

        current_diff = pool_diff_shared.value
        share_target_int = int(diff_to_target(current_diff if current_diff > 0 else 1), 16)

        nonce = random.randint(0, 0xFFFFFFFF)
        for _ in range(16000000):
            if shutdown_flag.value:
                return

            nonce_bytes = struct.pack("<I", nonce)
            full_header = header_prefix + nonce_bytes

            hash_result = hashlib.sha256(hashlib.sha256(full_header).digest()).digest()
            hash_int = int.from_bytes(hash_result[::-1], 'big')

            hashes_done += 1

            if hash_int < share_target_int:
                nonce_hex = f"{nonce:08x}"
                log_queue.put(f"*** FOUND SHARE! Process {thread_id} nonce {nonce_hex} ***")
                result_queue.put(("share", nonce, extranonce2, ntime))

            nonce = (nonce + 1) & 0xFFFFFFFF

            if hashes_done % 50000 == 0:
                now = time.time()
                elapsed = now - last_report
                if elapsed > 0:
                    hr = int(50000 / elapsed)
                    hashrate_array[thread_id] = hr
                last_report = now

# ======================  MERKLE ROOT ======================
def calculate_merkle_root(coinb1, coinb2, extranonce2_local, merkle_branch):
    coinbase = coinb1 + "00000000" + extranonce2_local + coinb2
    coinbase_hash = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()
    merkle = coinbase_hash
    for branch in merkle_branch:
        merkle = hashlib.sha256(hashlib.sha256(merkle + binascii.unhexlify(branch)).digest()).digest()
    return binascii.hexlify(merkle[::-1]).decode()

# ======================  STRATUM WORKER ======================
def stratum_worker(job_queue, shutdown_flag, log_queue, connected_flag, pool_diff_shared, user_str, password_str):
    global sock

    while not shutdown_flag.value:
        try:
            s = socket.socket()
            s.settimeout(120)
            s.connect(('stratum.antpool.com', 3333))
            sock = s
            connected_flag.value = True
            log_queue.put("Connected to AntPool")

            s.sendall(b'{"id":1,"method":"mining.subscribe","params":["alfa5.py/1.0"]}\n')

            auth = {"id":2,"method":"mining.authorize","params":[user_str,password_str]}
            s.sendall((json.dumps(auth)+"\n").encode())

            last_keepalive = time.time()

            buf = b""
            while not shutdown_flag.value:
                current_time = time.time()
                if current_time - last_keepalive > 25:
                    try:
                        s.sendall(b'\n')
                        last_keepalive = current_time
                    except:
                        pass

                try:
                    data = s.recv(4096)
                except socket.timeout:
                    continue

                if not data:
                    connected_flag.value = False
                    log_queue.put("[!] Connection lost – reconnecting...")
                    break

                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    log_queue.put(f"RX: {json.dumps(msg)}")
                    if "result" in msg and msg["id"] == 1:
                        extranonce1 = msg["result"][1]
                        extranonce2_size = msg["result"][2]
                    elif msg.get("method") == "mining.set_difficulty":
                        new_diff = int(msg["params"][0])
                        pool_diff_shared.value = new_diff
                        log_queue.put(f"Difficulty set to {new_diff}")
                    elif msg.get("method") == "mining.notify":
                        params = msg["params"]
                        if len(params) >= 9:
                            job_queue.put((
                                params[0], params[1], params[2], params[3], params[4],
                                params[5], params[6], params[7]
                            ))
        except Exception as e:
            connected_flag.value = False
            log_queue.put(f"[!] Connection error: {e} – retrying...")
            time.sleep(5)

# ======================  SUBMIT SHARE (in main process) ======================
def submit_share(nonce, extranonce2, ntime):
    payload = {
        "id": None,
        "method": "mining.submit",
        "params": [user, job_id, extranonce2, ntime, f"{nonce:08x}"]
    }
    try:
        msg = json.dumps(payload) + "\n"
        sock.sendall(msg.encode())
        log_lines.append(f"Submitted share: nonce={nonce:08x}")
        resp = sock.recv(4096).decode(errors='ignore').strip()
        log_lines.append(f"Pool response: {resp}")
        if '"result":true' in resp or '"result": true' in resp:
            accepted.value += 1
            log_lines.append("*** SHARE ACCEPTED ***")
        else:
            rejected.value += 1
            log_lines.append("[!] Share rejected")
    except Exception as e:
        log_lines.append(f"[!] Submit error: {e}")

# ======================  MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="alfa5.py - AntPool BTC Miner (Multiprocessing)")
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--worker", type=str, default="cpu002")
    args = parser.parse_args()

    user = f"{args.username}.{args.worker}"

    num_cores = os.cpu_count() or 24
    max_threads = 48

    mp.set_start_method('spawn')

    job_queue = mp.Queue()
    result_queue = mp.Queue()
    log_queue = mp.Queue()
    shutdown_flag = mpValue('b', False)
    hashrate_array = mpArray('i', [0] * max_threads)
    connected_flag = mpValue('b', False)
    pool_diff_shared = mpValue('i', 1)
    accepted = mpValue('i', 0)
    rejected = mpValue('i', 0)

    # log_lines in main process
    log_lines = []
    max_log = 40

    # Start stratum worker (pass user and password as strings)
    p_stratum = Process(target=stratum_worker, args=(job_queue, shutdown_flag, log_queue, connected_flag, pool_diff_shared, user, "x"))
    p_stratum.daemon = True
    p_stratum.start()

    # Start mining workers
    miners = []
    for i in range(max_threads):
        p = Process(target=mining_worker, args=(i, job_queue, result_queue, shutdown_flag, hashrate_array, log_queue, pool_diff_shared))
        p.daemon = True
        p.start()
        miners.append(p)

    # Display in main process
    stdscr = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)

    try:
        while True:
            stdscr.clear()
            h, w = stdscr.getmaxyx()

            title = " alfa5.py - AntPool BTC Miner (Multiprocessing) "
            stdscr.addstr(0, 0, title.center(w), curses.color_pair(5) | curses.A_BOLD)

            status = "ONLINE" if connected_flag.value else "OFFLINE"
            color = 1 if connected_flag.value else 2
            stdscr.addstr(2, 2, f"Status    : {status}", curses.color_pair(color) | curses.A_BOLD)

            try:
                block_height = requests.get('https://mempool.space/api/blocks/tip/height', timeout=3).text
            except:
                block_height = "???"
            stdscr.addstr(3, 2, f"Block     : {block_height}", curses.color_pair(3))

            total_hr = sum(hashrate_array)
            mh_s = total_hr / 1_000_000
            stdscr.addstr(4, 2, f"Real Hashrate: {mh_s:.2f} MH/s ({total_hr:,} H/s)", curses.color_pair(1) | curses.A_BOLD)

            stdscr.addstr(5, 2, f"Processes : {max_threads}", curses.color_pair(4))

            cpu_temp = get_cpu_temp()
            stdscr.addstr(6, 2, f"Temp      : {cpu_temp}", curses.color_pair(3))

            stdscr.addstr(7, 2, f"Accepted  : {accepted.value}", curses.color_pair(1))
            stdscr.addstr(8, 2, f"Rejected  : {rejected.value}", curses.color_pair(2))

            stdscr.addstr(9, 2, f"Pool Diff : {pool_diff_shared.value if pool_diff_shared.value > 0 else 'Waiting...'}", curses.color_pair(4))

            stdscr.addstr(11, 0, "─" * w, curses.color_pair(3))

            start_y = 12
            while not log_queue.empty():
                log_msg = log_queue.get()
                log_lines.append(log_msg)
            for i, line in enumerate(log_lines[-max_log:]):
                if start_y + i >= h:
                    break
                color = 1 if "accepted" in line.lower() else (2 if "rejected" in line.lower() or "error" in line.lower() else 3)
                stdscr.addstr(start_y + i, 2, line[:w-4], curses.color_pair(6))

            # Handle share results
            while not result_queue.empty():
                result = result_queue.get()
                if result[0] == "share":
                    submit_share(result[1], result[2], result[3])

            stdscr.refresh()
            time.sleep(0.4)

    except KeyboardInterrupt:
        pass
    finally:
        shutdown_flag.value = True
        p_stratum.join()
        for p in miners:
            p.join()
        curses.endwin()
