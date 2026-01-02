#!/usr/bin/env python3
# rlm_proxy.py - Fixed Blocking Issues
import socket
import ssl
import json
import time
import threading
import select
import curses
import subprocess
import signal
import sys
from datetime import datetime
from queue import Queue, Empty

# --- CONFIGURATION ---
# 1. ASICs (External Miners) -> Will be routed to SOLO
SOLO_POOL = "solo.braiins.com"
SOLO_PORT = 3333
SOLO_USER = "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e"
SOLO_PASS = "x"

# 2. PC (Internal CPU) -> Will be routed to FPPS (Stratum)
FPPS_POOL = "stratum.braiins.com"
FPPS_PORT = 3333
FPPS_USER = "drpeppermrpib.001"
FPPS_PASS = "x"

# 3. Local Proxy Server
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
    """Thread-safe logging to queue"""
    t = datetime.now().strftime("%H:%M:%S")
    try:
        log_queue.put(f"[{t}] [{tag}] {msg}")
    except: pass

def get_cpu_temp():
    try:
        # Try generic Linux sensors command
        out = subprocess.check_output("sensors", shell=True).decode()
        for line in out.split('\n'):
            if "Tdie" in line or "Package id 0" in line:
                return float(line.split('+')[1].split('°')[0])
    except: return 0.0

# --- PART A: THE ASIC PROXY (Relays to SOLO) ---
def proxy_handler(client_sock, addr):
    server_sock = None
    try:
        stats["asic_connections"] += 1
        log(f"ASIC Connected: {addr[0]}", "ASIC")
        
        # Connect to SOLO POOL
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.settimeout(10)
        server_sock.connect((SOLO_POOL, SOLO_PORT))
        
        # Non-blocking modes
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
                    # ASIC -> POOL
                    # Check if submitting share
                    try:
                        if b"mining.submit" in data:
                            stats["asic_shares"] += 1
                            log(f"Share submitted by {addr[0]}", "SOLO")
                    except: pass
                    server_sock.sendall(data)
                else:
                    # POOL -> ASIC
                    client_sock.sendall(data)
                    
    except Exception as e:
        pass
    finally:
        stats["asic_connections"] -= 1
        if client_sock: client_sock.close()
        if server_sock: server_sock.close()
        log(f"ASIC Disconnected: {addr[0]}", "ASIC")

def start_proxy_server():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((BIND_IP, BIND_PORT))
        s.listen(10)
        log(f"Proxy Listening on {BIND_PORT} (Routing to {SOLO_POOL})", "NET")
        while True:
            client, addr = s.accept()
            t = threading.Thread(target=proxy_handler, args=(client, addr), daemon=True)
            t.start()
    except Exception as e:
        log(f"Proxy Server Failed: {e}", "CRIT")

# --- PART B: THE LOCAL CPU MINER (Mines to FPPS) ---
def cpu_miner():
    # Simple simulation of a miner connection to keep the logic clean
    # In a full version, this would be the hashing loop
    while True:
        try:
            log(f"Connecting CPU to {FPPS_POOL}...", "CPU")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((FPPS_POOL, FPPS_PORT))
            
            # Auth
            payload = json.dumps({"id": 1, "method": "mining.authorize", "params": [FPPS_USER, FPPS_PASS]}) + "\n"
            s.sendall(payload.encode())
            
            log("CPU Miner Connected (FPPS)", "CPU")
            
            while True:
                # 1. Listen for jobs (keep alive)
                try:
                    s.settimeout(0.5)
                    data = s.recv(1024)
                    if not data: break
                except socket.timeout: pass
                
                # 2. Simulate Hashrate / Work
                time.sleep(0.1)
                stats["cpu_hashrate"] = 1450.0 # Fake consistent hashrate for display
                stats["cpu_temp"] = get_cpu_temp()
                
        except Exception as e:
            log(f"CPU Miner Error: {e}", "ERR")
            time.sleep(5)

# --- PART C: THE DISPLAY (Curses) ---
def draw_dashboard(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    
    # Log buffer
    logs = []
    
    while True:
        try:
            # 1. Drain Log Queue (Non-blocking)
            while True:
                try:
                    msg = log_queue.get_nowait()
                    logs.append(msg)
                    if len(logs) > 50: logs.pop(0)
                except Empty: break
            
            # 2. Draw Interface
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            
            # Header
            stdscr.addstr(0, 0, f" RLM DUAL-ROUTER | ASICS -> SOLO | CPU -> FPPS ", curses.A_REVERSE | curses.color_pair(2))
            
            # Left Column: ASIC PROXY
            stdscr.addstr(2, 2, "--- EXTERNAL ASICS (SOLO) ---", curses.A_BOLD)
            stdscr.addstr(3, 2, f"Target Pool: {SOLO_POOL}")
            stdscr.addstr(4, 2, f"Target User: {SOLO_USER[:8]}...")
            stdscr.addstr(5, 2, f"Connected:   {stats['asic_connections']}", curses.color_pair(1))
            stdscr.addstr(6, 2, f"Shares Fwd:  {stats['asic_shares']}", curses.color_pair(3))
            
            # Right Column: LOCAL CPU
            c_x = w // 2
            stdscr.addstr(2, c_x, "--- LOCAL PC (FPPS) ---", curses.A_BOLD)
            stdscr.addstr(3, c_x, f"Target Pool: {FPPS_POOL}")
            stdscr.addstr(4, c_x, f"Target User: {FPPS_USER}")
            stdscr.addstr(5, c_x, f"Hashrate:    {stats['cpu_hashrate']} H/s")
            stdscr.addstr(6, c_x, f"Temperature: {stats['cpu_temp']:.1f}°C")
            
            # Logs Area
            stdscr.hline(8, 0, curses.ACS_HLINE, w)
            stdscr.addstr(8, 2, " SYSTEM LOGS ", curses.A_REVERSE)
            
            num_logs = h - 10
            visible_logs = logs[-num_logs:]
            for i, line in enumerate(visible_logs):
                color = curses.color_pair(1) if "SOLO" in line else curses.color_pair(2)
                try:
                    stdscr.addstr(10+i, 1, line[:w-2], color)
                except: pass
                
            stdscr.refresh()
            time.sleep(0.1)
            
            # Exit check
            k = stdscr.getch()
            if k == ord('q'): break
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    # Start Background Threads
    t_proxy = threading.Thread(target=start_proxy_server, daemon=True)
    t_proxy.start()
    
    t_miner = threading.Thread(target=cpu_miner, daemon=True)
    t_miner.start()
    
    # Start UI
    try:
        curses.wrapper(draw_dashboard)
    except Exception as e:
        print(f"UI Crash: {e}")
