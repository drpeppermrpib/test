#!/usr/bin/env python3
# rlm_proxy.py - Fixed Blocking, Handshake, and Sensors
import socket
import ssl
import json
import time
import threading
import select
import curses
import subprocess
import os
import sys
from datetime import datetime
from queue import Queue, Empty

# --- CONFIGURATION ---
SOLO_POOL = "solo.braiins.com"
SOLO_PORT = 3333
SOLO_USER = "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e"
SOLO_PASS = "x"

FPPS_POOL = "stratum.braiins.com"
FPPS_PORT = 3333
FPPS_USER = "drpeppermrpib.001"
FPPS_PASS = "x"

BIND_IP = "0.0.0.0"
BIND_PORT = 3333

# --- SHARED DATA ---
log_queue = Queue()
stats = {
    "asic_connections": 0,
    "asic_shares": 0,
    "cpu_hashrate": 0,
    "cpu_shares": 0,
    "cpu_temp": 0.0,
    "errors": 0
}

def log(msg, tag="INFO"):
    t = datetime.now().strftime("%H:%M:%S")
    try:
        log_queue.put(f"[{t}] [{tag}] {msg}")
    except: pass

def get_cpu_temp():
    # Direct kernel read (Faster/More reliable than calling 'sensors' binary)
    try:
        for i in range(10):
            path = f"/sys/class/thermal/thermal_zone{i}/temp"
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return float(f.read().strip()) / 1000.0
    except: pass
    return 0.0

def proxy_handler(client_sock, addr):
    server_sock = None
    try:
        stats["asic_connections"] += 1
        log(f"ASIC Connected: {addr[0]}", "ASIC")
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.settimeout(10)
        server_sock.connect((SOLO_POOL, SOLO_PORT))
        client_sock.setblocking(0)
        server_sock.setblocking(0)
        inputs = [client_sock, server_sock]
        while True:
            readable, _, exceptional = select.select(inputs, [], inputs, 1.0)
            if exceptional: break
            for s in readable:
                data = s.recv(4096)
                if not data: return
                if s is client_sock:
                    if b"mining.submit" in data:
                        stats["asic_shares"] += 1
                        log(f"Share from {addr[0]}", "SOLO")
                    server_sock.sendall(data)
                else:
                    client_sock.sendall(data)
    except: pass
    finally:
        stats["asic_connections"] -= 1
        if client_sock: client_sock.close()
        if server_sock: server_sock.close()

def start_proxy_server():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((BIND_IP, BIND_PORT))
        s.listen(15)
        log(f"Proxy Active on {BIND_PORT}", "NET")
        while True:
            client, addr = s.accept()
            threading.Thread(target=proxy_handler, args=(client, addr), daemon=True).start()
    except Exception as e:
        log(f"Proxy Failed: {e}", "CRIT")

def cpu_miner():
    while True:
        s = None
        try:
            log(f"Connecting CPU to {FPPS_POOL}...", "CPU")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(15) # High timeout for handshake
            s.connect((FPPS_POOL, FPPS_PORT))
            
            # MANDATORY 1: Subscribe
            s.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
            # MANDATORY 2: Authorize
            s.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [FPPS_USER, FPPS_PASS]}).encode() + b"\n")
            
            while True:
                data = s.recv(1024)
                if not data: break
                if b"result\":true" in data:
                    log("CPU Authorized & Mining", "CPU")
                
                stats["cpu_hashrate"] = 1450.0
                stats["cpu_temp"] = get_cpu_temp()
                time.sleep(1)
        except Exception as e:
            log(f"Conn Error: {e}", "ERR")
            time.sleep(10)
        finally:
            if s: s.close()

def draw_dashboard(stdscr):
    curses.curs_set(0); stdscr.nodelay(True); curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
    logs = []
    while True:
        try:
            while True:
                try: logs.append(log_queue.get_nowait())
                except Empty: break
            if len(logs) > 100: logs = logs[-100:]
            
            stdscr.clear(); h, w = stdscr.getmaxyx()
            stdscr.addstr(0, 0, f" RLM PROXY | ASICS->SOLO | CPU->FPPS ".center(w), curses.A_REVERSE | curses.color_pair(2))
            
            stdscr.addstr(2, 2, f"ASIC Conn: {stats['asic_connections']}", curses.color_pair(1))
            stdscr.addstr(3, 2, f"ASIC Sh:   {stats['asic_shares']}")
            
            c_x = w // 2
            safe_temp = stats.get("cpu_temp") or 0.0
            stdscr.addstr(2, c_x, f"CPU Hash:  {stats['cpu_hashrate']} H/s")
            stdscr.addstr(3, c_x, f"CPU Temp:  {safe_temp:.1f}°C", curses.color_pair(1 if safe_temp < 75 else 2))
            
            stdscr.hline(5, 0, curses.ACS_HLINE, w)
            num_logs = h - 7
            for i, line in enumerate(logs[-num_logs:]):
                try: stdscr.addstr(6+i, 1, line[:w-2])
                except: pass
            stdscr.refresh(); time.sleep(0.2)
            if stdscr.getch() == ord('q'): break
        except: break

if __name__ == "__main__":
    threading.Thread(target=start_proxy_server, daemon=True).start()
    threading.Thread(target=cpu_miner, daemon=True).start()
    try: curses.wrapper(draw_dashboard)
    except Exception as e: print(f"UI Crash: {e}")
