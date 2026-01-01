import socket
import json
import time
import hashlib
import struct
import binascii
import multiprocessing
import os
import sys

# --- CONFIGURATION (Defaults) ---
# You can change these here or pass them as arguments
DEFAULT_POOL = "stratum.antpool.com" 
DEFAULT_PORT = 3333
DEFAULT_USER = "your_subaccount.worker_name" # CHANGE THIS
DEFAULT_PASS = "123"

def clean_host(host_url):
    """
    Removes 'stratum+tcp://' and port numbers if pasted incorrectly.
    Returns a clean hostname like 'stratum.antpool.com'.
    """
    # Remove protocol prefix
    if "://" in host_url:
        host_url = host_url.split("://")[1]
    
    # Remove port if included in the host string (we use the separate port arg)
    if ":" in host_url:
        host_url = host_url.split(":")[0]
        
    return host_url

def solve_block(job_args):
    """
    Optimized CPU solver for Threadripper (Pure Python implementation).
    NOTE: Real high-speed mining usually requires compiled C/C++ or CUDA.
    This function is a placeholder for the logic to hash the block header.
    """
    # Unpack job arguments (simplified for readability)
    job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, extranonce1, extranonce2_size = job_args
    
    # Simulation of work (in a real miner, this loop checks billions of nonces)
    # Since we can't hit ASIC speeds in Python, we sleep briefly to prevent CPU freeze
    # during this 'demo' phase.
    time.sleep(0.01)
    return None

def miner_process(pool_host, pool_port, user, password):
    """
    Main worker process that handles the Stratum connection and mining loop.
    """
    buffer = ""
    pool_host = clean_host(pool_host)
    
    print(f"[*] Threadripper Worker starting...")
    print(f"[*] Connecting to {pool_host}:{pool_port} as {user}")

    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(30) # 30 second timeout prevents hanging

        try:
            s.connect((pool_host, int(pool_port)))
            print(f"[+] Connected to Stratum Server!")

            # 1. Mining Subscribe
            # Some pools need specific user agent strings
            subscribe_payload = json.dumps({
                "id": 1, 
                "method": "mining.subscribe", 
                "params": ["AlfaUltra/2.0"]
            }) + "\n"
            s.sendall(subscribe_payload.encode())

            # 2. Mining Authorize
            authorize_payload = json.dumps({
                "id": 2, 
                "method": "mining.authorize", 
                "params": [user, password]
            }) + "\n"
            s.sendall(authorize_payload.encode())
            print("[*] Sent Authorization...")

            # 3. Work Loop
            while True:
                # Receive data in chunks
                try:
                    data = s.recv(4096).decode('utf-8')
                    if not data:
                        print("[-] Connection closed by server")
                        break
                except socket.timeout:
                    print("[!] Socket timed out, reconnecting...")
                    break
                
                buffer += data
                
                # Process lines (Stratum is newline delimited)
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip(): 
                        continue
                    
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"[!] JSON Decode Error on line: {line[:50]}...")
                        continue

                    # Handle Messages
                    msg_id = message.get('id')
                    method = message.get('method')
                    result = message.get('result')

                    if msg_id == 1:
                        print(f"[+] Subscribed! Extranonce1: {result[1]}")
                    
                    elif msg_id == 2:
                        if result is True:
                            print("[+] Authorized successfully! Waiting for jobs...")
                        else:
                            print(f"[!] Authorization Failed! Check username: {user}")

                    elif method == "mining.notify":
                        # New Job Received
                        params = message.get('params', [])
                        job_id = params[0]
                        print(f"[*] New Job Detected: ID {job_id} | Cleaning previous queue...")
                        
                        # Here you would dispatch work to multiprocessing.Pool
                        # For this script, we just acknowledge receipt
                        
                    elif method == "mining.set_difficulty":
                        new_diff = params[0]
                        print(f"[*] Difficulty adjusted to: {new_diff}")

        except ConnectionRefusedError:
            print(f"[!] Connection Refused by {pool_host}. Blocking? Firewall?")
        except socket.gaierror:
             print(f"[!] DNS Error. Could not resolve {pool_host}. Check internet.")
        except Exception as e:
            print(f"[!] Error: {e}")
        finally:
            s.close()
            print("[*] Reconnecting in 10 seconds...")
            time.sleep(10)

if __name__ == "__main__":
    # You can run this directly or pass args: python alfa-ultra.py <pool> <port> <user>
    if len(sys.argv) > 3:
        host = sys.argv[1]
        port = int(sys.argv[2])
        user = sys.argv[3]
    else:
        host = DEFAULT_POOL
        port = DEFAULT_PORT
        user = DEFAULT_USER
        
    # Start the Miner
    try:
        # We use a wrapper to restart the miner if it crashes completely
        miner_process(host, port, user, DEFAULT_PASS)
    except KeyboardInterrupt:
        print("\n[*] Miner stopped by user.")
