#!/usr/bin/env python3
import socket, json, time, threading, select, curses, subprocess, sys, queue
from datetime import datetime

# --- CONFIG ---
SOLO_POOL, SOLO_PORT = "solo.braiins.com", 3333
SOLO_USER, SOLO_PASS = "bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e", "x"
FPPS_POOL, FPPS_PORT = "stratum.braiins.com", 3333
FPPS_USER, FPPS_PASS = "drpeppermrpib.001", "x"
BIND_IP, BIND_PORT = "0.0.0.0", 3333

log_q = queue.Queue()
stats = {"asic_conn": 0, "asic_sh": 0, "cpu_hr": 0, "cpu_sh": 0, "cpu_temp": 0.0}

def log(msg, tag="INFO"):
    t = datetime.now().strftime("%H:%M:%S")
    log_q.put(f"[{t}] [{tag}] {msg}")

def get_cpu_temp():
    try:
        out = subprocess.check_output("sensors", shell=True).decode()
        for line in out.split('\n'):
            if "Tdie" in line or "Package id 0" in line:
                return float(line.split('+')[1].split('°')[0])
    except: pass
    return 0.0 # Never return None

def proxy_handler(client_sock, addr):
    server_sock = None
    try:
        stats["asic_conn"] += 1
        log(f"ASIC Connected: {addr[0]}", "NET")
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.connect((SOLO_POOL, SOLO_PORT))
        client_sock.setblocking(0); server_sock.setblocking(0)
        while True:
            r, _, x = select.select([client_sock, server_sock], [], [client_sock, server_sock], 1.0)
            if x: break
            for s in r:
                data = s.recv(4096)
                if not data: return
                if s is client_sock:
                    if b"mining.submit" in data: stats["asic_sh"] += 1
                    server_sock.sendall(data)
                else: client_sock.sendall(data)
    except: pass
    finally:
        stats["asic_conn"] -= 1
        client_sock.close()
        if server_sock: server_sock.close()

def start_proxy():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((BIND_IP, BIND_PORT)); s.listen(10)
    while True:
        c, a = s.accept()
        threading.Thread(target=proxy_handler, args=(c, a), daemon=True).start()

def ui(stdscr):
    curses.curs_set(0); stdscr.nodelay(True); curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    logs = []
    while True:
        while not log_q.empty():
            logs.append(log_q.get()); 
            if len(logs) > 40: logs.pop(0)
        
        stats["cpu_temp"] = get_cpu_temp()
        stdscr.clear(); h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " RLM PROXY | Q to Quit ".center(w), curses.A_REVERSE)
        
        # Safe formatting: use float() and default to 0 if None crept in
        temp = float(stats.get("cpu_temp") or 0.0)
        
        stdscr.addstr(2, 2, f"ASIC Conn: {stats['asic_conn']}")
        stdscr.addstr(3, 2, f"ASIC Sh:   {stats['asic_sh']}")
        stdscr.addstr(2, 30, f"CPU Temp:  {temp:.1f} C") # Fixed crash point
        
        for i, l in enumerate(logs[- (h-6):]):
            try: stdscr.addstr(5+i, 1, l[:w-2])
            except: pass
        stdscr.refresh(); time.sleep(0.2)
        if stdscr.getch() == ord('q'): break

if __name__ == "__main__":
    threading.Thread(target=start_proxy, daemon=True).start()
    curses.wrapper(ui)
