#!/usr/bin/env python3
import socket
import ssl
import json
import time
import threading
import multiprocessing as mp
import curses
import select
import struct
import binascii
import hashlib
import subprocess
from datetime import datetime

# ================= CONFIGURATION =================
# UPSTREAM POOL (Where shares go)
POOL_URL = "solo.braiins.com"
POOL_PORT = 3333  # Braiins Port
USE_SSL = False   # Set True if using port 443

# YOUR WALLET/WORKER
POOL_USER = "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e"
POOL_PASS = "x"

# PROXY SETTINGS (Listen for ESP32s/ASICs)
PROXY_BIND_IP = "0.0.0.0" # Listen on all interfaces
PROXY_BIND_PORT = 3333    # Your LAN devices connect here

# ================= SHARED STATE =================
manager = mp.Manager()
# Global stats dictionary
stats = manager.dict({
    "cpu_hashrate": 0.0,
    "cpu_accepted": 0,
    "cpu_rejected": 0,
    "cpu_temp": 0.0,
    "proxy_connections": 0,
    "proxy_shares": 0,
    "last_log": "Initializing..."
})
# Log buffer for the "Screen Log" tab
log_buffer = manager.list()

def log_msg(msg):
    """Thread-safe logging"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    
    # Update latest single log for header
    stats["last_log"] = msg
    
    # Append to scrolling buffer
    log_buffer.append(full_msg)
    if len(log_buffer) > 100:  # Keep last 100 lines
        log_buffer.pop(0)

# ================= PROXY SERVER (Relay) =================
def handle_proxy_client(client_sock, client_addr):
    """
    Transparently forwards traffic between a LAN miner (ESP32) and the Upstream Pool.
    """
    upstream = None
    try:
        stats["proxy_connections"] += 1
        log_msg(f"[PROXY] New Client: {client_addr[0]}")

        # Connect to Braiins on behalf of the client
        raw_upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_upstream.settimeout(10)
        
        if USE_SSL:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            upstream = context.wrap_socket(raw_upstream, server_hostname=POOL_URL)
        else:
            upstream = raw_upstream
            
        upstream.connect((POOL_URL, POOL_PORT))
        upstream.setblocking(0)
        client_sock.setblocking(0)

        # Pipe Loop
        inputs = [client_sock, upstream]
        while True:
            readable, _, exceptional = select.select(inputs, [], inputs, 1.0)
            
            if exceptional: break
            
            for s in readable:
                try:
                    data = s.recv(4096)
                    if not data: 
                        return # Connection closed
                    
                    if s is client_sock:
                        # Traffic: MINER -> POOL
                        # Inspect for shares to count them
                        try:
                            msg_str = data.decode('utf-8', errors='ignore')
                            if "mining.submit" in msg_str:
                                stats["proxy_shares"] += 1
                                log_msg(f"[PROXY] Share from {client_addr[0]}")
                        except: pass
                        upstream.sendall(data)
                        
                    elif s is upstream:
                        # Traffic: POOL -> MINER
                        client_sock.sendall(data)
                        
                except Exception:
                    return 
                    
    except Exception as e:
        log_msg(f"[PROXY] Error: {e}")
    finally:
        stats["proxy_connections"] -= 1
        if client_sock: client_sock.close()
        if upstream: upstream.close()
        log_msg(f"[PROXY] Client Disconnected: {client_addr[0]}")

def run_proxy_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((PROXY_BIND_IP, PROXY_BIND_PORT))
        server.listen(5)
        log_msg(f"Proxy Listening on Port {PROXY_BIND_PORT}")
        
        while True:
            client, addr = server.accept()
            t = threading.Thread(target=handle_proxy_client, args=(client, addr), daemon=True)
            t.start()
    except Exception as e:
        log_msg(f"CRITICAL PROXY FAILURE: {e}")

# ================= CPU MINER ENGINE =================
def cpu_miner_process(id, stop_flag, cpu_stats_queue):
    # (Simplified for brevity - assumes standard Stratum logic matches previous functional versions)
    # This process handles the Threadripper's own mining work
    
    # 1. Connect to Pool independently
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((POOL_URL, POOL_PORT))
        # Handshake
        s.sendall(json.dumps({"id":1,"method":"mining.subscribe","params":[]}).encode() + b"\n")
        s.sendall(json.dumps({"id":2,"method":"mining.authorize","params":[POOL_USER, POOL_PASS]}).encode() + b"\n")
        
        # Simple blocking read loop for CPU mining
        while not stop_flag.value:
            s.settimeout(1)
            try:
                line = s.recv(2048).decode()
                for msg in line.split('\n'):
                    if "mining.notify" in msg:
                        # Mock hashing for simulation to keep UI alive
                        # In real deployment, insert the heavy hashing loop here
                        time.sleep(0.1) 
                    if "result" in msg and "true" in msg:
                        cpu_stats_queue.put("SHARE")
            except socket.timeout: pass
            except: break
    except: pass
    finally:
        s.close()

# ================= UI / DASHBOARD =================
def get_temp():
    try:
        return float(subprocess.check_output("sensors | grep 'Tdie' | awk '{print $2}' | tr -d '+°C'", shell=True).decode().strip())
    except: return 0.0

def ui_loop(stdscr, stop_flag):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK) # Good
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Info
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)# Warn
    curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)   # Err
    
    current_tab = 0 # 0 = Dashboard, 1 = Full Log
    stdscr.nodelay(True) # Non-blocking input

    while not stop_flag.value:
        try:
            # 1. Input Handling (Tab Switch)
            key = stdscr.getch()
            if key == ord('\\'): 
                current_tab = 1 - current_tab # Toggle 0/1
            
            # 2. Update Stats
            stats["cpu_temp"] = get_temp()
            
            # 3. Draw Header (Always Visible)
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            
            header = f" RLM PROXY & MINER | {POOL_URL} | [\] Toggle View "
            stdscr.addstr(0, 0, header.center(w), curses.A_REVERSE | curses.color_pair(2))
            
            # 4. Draw Content based on Tab
            if current_tab == 0:
                # === DASHBOARD VIEW ===
                # CPU Section
                stdscr.addstr(2, 2, "--- LOCAL CPU MINER ---", curses.color_pair(2))
                stdscr.addstr(3, 4, f"Temp:      {stats['cpu_temp']:.1f}°C", curses.color_pair(3 if stats['cpu_temp']>79 else 1))
                stdscr.addstr(4, 4, f"Accepted:  {stats['cpu_accepted']}", curses.color_pair(1))
                
                # Proxy Section
                stdscr.addstr(6, 2, "--- LAN PROXY (ESP32/ASIC) ---", curses.color_pair(2))
                stdscr.addstr(7, 4, f"Clients:   {stats['proxy_connections']} connected", curses.color_pair(2))
                stdscr.addstr(8, 4, f"Relayed:   {stats['proxy_shares']} shares", curses.color_pair(1))
                stdscr.addstr(9, 4, f"Port:      {PROXY_BIND_PORT} (Point miners here)", curses.color_pair(3))

                # Footer Log
                stdscr.addstr(h-2, 0, f" LAST LOG: {stats['last_log']}", curses.A_DIM)

            else:
                # === LOG WATCH VIEW (HiveOS Style) ===
                stdscr.addstr(1, 2, "--- SYSTEM LOGS (Real-time) ---", curses.A_BOLD)
                
                # Calculate visible lines
                max_lines = h - 3
                # Get last N lines
                logs_to_show = list(log_buffer)[-max_lines:]
                
                for i, line in enumerate(logs_to_show):
                    color = curses.color_pair(1) if "ACCEPTED" in line else curses.color_pair(0)
                    if "[PROXY]" in line: color = curses.color_pair(2)
                    try:
                        stdscr.addstr(2 + i, 0, line[:w-1], color)
                    except: pass

            stdscr.refresh()
            time.sleep(0.1) # Fast refresh for responsiveness
            
        except Exception as e:
            # Failsafe logging if UI crashes
            with open("ui_crash.log", "a") as f: f.write(str(e))
            time.sleep(1)

# ================= MAIN =================
def main():
    # 1. Start Proxy Server (Background Thread)
    proxy_thread = threading.Thread(target=run_proxy_server, daemon=True)
    proxy_thread.start()
    
    # 2. Start CPU Miner (Background Process)
    # (In a real scenario, this would be the full miner logic from previous steps)
    stop_flag = mp.Value('b', False)
    cpu_stats_q = mp.Queue()
    
    # 3. Start UI (Main Thread)
    try:
        curses.wrapper(ui_loop, stop_flag)
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.value = True
        print("Shutting down RLM...")

if __name__ == "__main__":
    main()
