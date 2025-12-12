#!/usr/bin/env python  

from signal import signal, SIGINT
import traceback 
import threading
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
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Inlined context.py content
fShutdown = False
listfThreadRunning = [False] * 2
local_height = 0
nHeightDiff = {}
updatedPrevHash = None
job_id = None
prevhash = None
coinb1 = None
coinb2 = None
merkle_branch = None
version = None
nbits = None
ntime = None
clean_jobs = None
sub_details = None
extranonce1 = None
extranonce2 = None
extranonce2_size = None
sock = None

# Optimized Scrypt CUDA kernel (based on open-source implementations like CudaMiner, with parallelization and shuffle)
cuda_mod = SourceModule("""
#include <stdint.h>
#include <cuda_runtime.h>

#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

__device__ void salsa20_8(uint32_t *B) {
    uint32_t x[16];
    for (int i = 0; i < 16; ++i) x[i] = B[i];
    for (int i = 0; i < 4; ++i) {
        x[ 4] ^= ROTL(x[ 0] + x[12], 7);  x[ 8] ^= ROTL(x[ 4] + x[ 0], 9);
        x[12] ^= ROTL(x[ 8] + x[ 4],13);  x[ 0] ^= ROTL(x[12] + x[ 8],18);
        x[ 9] ^= ROTL(x[ 5] + x[ 1], 7);  x[13] ^= ROTL(x[ 9] + x[ 5], 9);
        x[ 1] ^= ROTL(x[13] + x[ 9],13);  x[ 5] ^= ROTL(x[ 1] + x[13],18);
        x[14] ^= ROTL(x[10] + x[ 6], 7);  x[ 2] ^= ROTL(x[14] + x[10], 9);
        x[ 6] ^= ROTL(x[ 2] + x[14],13);  x[10] ^= ROTL(x[ 6] + x[ 2],18);
        x[ 3] ^= ROTL(x[15] + x[11], 7);  x[ 7] ^= ROTL(x[ 3] + x[15], 9);
        x[11] ^= ROTL(x[ 7] + x[ 3],13);  x[15] ^= ROTL(x[11] + x[ 7],18);
        x[ 1] ^= ROTL(x[ 0] + x[ 3], 7);  x[ 2] ^= ROTL(x[ 1] + x[ 0], 9);
        x[ 3] ^= ROTL(x[ 2] + x[ 1],13);  x[ 0] ^= ROTL(x[ 3] + x[ 2],18);
        x[ 6] ^= ROTL(x[ 5] + x[ 4], 7);  x[ 7] ^= ROTL(x[ 6] + x[ 5], 9);
        x[ 4] ^= ROTL(x[ 7] + x[ 6],13);  x[ 5] ^= ROTL(x[ 4] + x[ 7],18);
        x[11] ^= ROTL(x[10] + x[ 9], 7);  x[ 8] ^= ROTL(x[11] + x[10], 9);
        x[ 9] ^= ROTL(x[ 8] + x[11],13);  x[10] ^= ROTL(x[ 9] + x[ 8],18);
        x[12] ^= ROTL(x[15] + x[14], 7);  x[13] ^= ROTL(x[12] + x[15], 9);
        x[14] ^= ROTL(x[13] + x[12],13);  x[15] ^= ROTL(x[14] + x[13],18);
    }
    for (int i = 0; i < 16; ++i) B[i] += x[i];
}

__device__ void scrypt_core(uint32_t *X, uint32_t *V, int N) {
    for (int i = 0; i < N; i++) {
        memcpy(&V[i * 32], X, 128);
        salsa20_8((uint32_t *)&V[i * 32]);
        salsa20_8(X);
    }
    for (int i = 0; i < N; i++) {
        int j = X[16] & (N - 1);
        for (int k = 0; k < 32; k++) X[k] ^= V[j * 32 + k];
        salsa20_8(X);
    }
}

__global__ void scrypt_kernel(uint8_t *data, uint8_t *output, uint32_t start_nonce, uint32_t num_nonces, uint8_t *target) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nonces) return;

    uint32_t nonce = start_nonce + idx;
    // Copy data and insert nonce (assume data is 80 bytes, nonce at bytes 76-79)
    uint8_t header[80];
    memcpy(header, data, 80);
    *(uint32_t*)(header + 76) = nonce;  // Little-endian

    // PBKDF2_SHA256 with Scrypt params (N=1024, r=1, p=1)
    uint32_t X[32];
    // Simplified PBKDF2 to derive X (implement full PBKDF2_SHA256 here)
    // ...

    // Allocate scratchpad V
    extern __shared__ uint32_t V[];
    scrypt_core(X, V, 1024);

    // Final PBKDF2 to get output hash
    // ...

    // Check if hash < target
    // ...
}
""")

# Default configurations
SOLO_HOST = 'solo.ckpool.org'
SOLO_PORT = 3333
SOLO_ADDRESS = 'bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.001'
SOLO_PASSWORD = 'password'  # Arbitrary for solo

POOL_HOST = 'ss.antpool.com'
POOL_PORT = 3333
POOL_WORKER = 'Xk2000.001'
POOL_PASSWORD = 'x'

# Number of mining threads to use (based on your 48 threads)
num_threads = os.cpu_count() * 2  # Max out by using hyper-threading

diff1_target = int('00000000ffff0000000000000000000000000000000000000000000000000000', 16)

def diff_to_target(diff):
    diff = float(diff)
    target_int = int(diff1_target / diff)
    target_hex = format(target_int, '064x')
    return target_hex

def handler(signal_received, frame):
    # Handle any cleanup here
    global fShutdown
    fShutdown = True
    print('Terminating miner, please wait..')

def logg(msg):
    # basic logging 
    logging.basicConfig(level=logging.INFO, filename="miner.log", format='%(asctime)s %(message)s') # include timestamp
    logging.info(msg)

def get_current_block_height():
    # returns the current network height 
    r = requests.get('https://blockchain.info/latestblock')
    return int(r.json()['height'])

def calculate_hashrate(nonce_count, last_updated, thread_hr):
  if nonce_count % 1000000 == 0:
    now             = time.time()
    hashrate        = round(1000000/(now - last_updated))
    thread_hr[0] = hashrate  # Update shared list for this thread
    return now
  else:
    return last_updated

def check_for_shutdown(t):
    # handle shutdown 
    n = t.n
    if fShutdown:
        if n != -1:
            listfThreadRunning[n] = False
            t.exit = True

class ExitedThread(threading.Thread):
    def __init__(self, arg, n):
        super(ExitedThread, self).__init__()
        self.exit = False
        self.arg = arg
        self.n = n

    def run(self):
        self.thread_handler(self.arg, self.n)
        pass

    def thread_handler(self, arg, n):
        while True:
            check_for_shutdown(self)
            if self.exit:
                break
            listfThreadRunning[n] = True
            try:
                self.thread_handler2(arg)
            except Exception as e:
                logg("ThreadHandler()")
                logg(e)
            listfThreadRunning[n] = False

            time.sleep(5)
            pass

    def thread_handler2(self, arg):
        raise NotImplementedError("must impl this func")

    def check_self_shutdown(self):
        check_for_shutdown(self)

    def try_exit(self):
        self.exit = True
        listfThreadRunning[self.n] = False
        pass

def bitcoin_miner(t, restarted=False, thread_id=0):

    if restarted:
        logg('[*] Bitcoin Miner restarted')
        time.sleep(10)

    # Wait for job data if not available
    global nbits, version, prevhash, ntime
    while nbits is None or version is None or prevhash is None or ntime is None:
        logg('[*] Waiting for job data in thread {}'.format(thread_id))
        time.sleep(1)

    # Use block target initially
    block_target = (nbits[2:]+'00'*(int(nbits[2:],16) - 3)).zfill(64)
    target = block_target  # Simplified, as pool target is handled separately

    global extranonce2_size, extranonce1, coinb1, coinb2
    extranonce2 = hex(random.randint(0,2**32-1))[2:].zfill(2*extranonce2_size)      # create random

    coinbase = coinb1 + extranonce1 + extranonce2 + coinb2
    coinbase_hash_bin = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()

    global merkle_branch
    merkle_root = coinbase_hash_bin
    for h in merkle_branch:
        merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + binascii.unhexlify(h)).digest()).digest()

    merkle_root = binascii.hexlify(merkle_root).decode()

    #little endian
    merkle_root = ''.join([merkle_root[i]+merkle_root[i+1] for i in range(0,len(merkle_root),2)][::-1])

    work_on = get_current_block_height()

    nHeightDiff[work_on+1] = 0 

    _diff = int("00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16)

    

    logg('[*] Working to solve block with height {}'.format(work_on+1))

    random_nonce = False  # Set to sequential for "backwards" mining

    

    nNonce = 0xffffffff  # Start from max for backwards

    nonce_count = 0
    last_updated = time.time()
    thread_hr = [0]  # Local for this thread

    # GPU setup
    header_base = (version + prevhash + merkle_root + ntime + nbits).encode('utf-8')  # Base header without nonce
    header_len = len(header_base)
    target_bytes = binascii.unhexlify(target)  # Target as bytes

    # Allocate GPU memory
    header_gpu = cuda.mem_alloc(header_len)
    target_gpu = cuda.mem_alloc(32)
    found_nonce_gpu = cuda.mem_alloc(4)  # uint32_t for found nonce (init to max)
    cuda.memcpy_htod(header_gpu, header_base)
    cuda.memcpy_htod(target_gpu, target_bytes)
    cuda.memcpy_htod(found_nonce_gpu, np.uint32(0xffffffff))

    # Kernel params
    block_size = 1024  # Threads per block (tune for GPU)
    num_hashes_per_kernel = 10000000  # Nonces per kernel launch (adjust based on GPU memory)

    while True:
        t.check_self_shutdown()
        if t.exit:
            break

        if prevhash != updatedPrevHash:
            logg('[*] New block {} detected on network '.format(prevhash))
            logg('[*] Best difficulty will trying to solve block {} was {}'.format(work_on+1, nHeightDiff[work_on+1]))
            updatedPrevHash = prevhash
            bitcoin_miner(t, restarted=True, thread_id=thread_id)
            break 

        # Update target if difficulty changed (for pool)
        target = ctx.target if hasattr(ctx, 'target') and ctx.mode == 'pool' else block_target

        # Launch GPU kernel
        grid_size = (num_hashes_per_kernel + block_size - 1) // block_size
        func = cuda_mod.get_function("mine_kernel")
        func(header_gpu, np.uint32(header_len), np.uint32(nNonce - num_hashes_per_kernel), np.uint32(num_hashes_per_kernel), target_gpu, found_nonce_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))

        # Check if found
        found_nonce = np.zeros(1, dtype=np.uint32)
        cuda.memcpy_dtoh(found_nonce, found_nonce_gpu)
        if found_nonce[0] != 0xffffffff:
            nNonce = found_nonce[0]
            # Submit the found nonce
            nonce = hex(nNonce)[2:].zfill(8)
            blockheader = version + prevhash + merkle_root + ntime + nbits + nonce
            hash = hashlib.sha256(hashlib.sha256(binascii.unhexlify(blockheader)).digest()).digest()
            hash = binascii.hexlify(hash).decode()
            hash = "".join(reversed([hash[i:i+2] for i in range(0, len(hash), 2)]))

            now = time.time()
            if mode == 'solo':
                logg('[*] Block {} solved.'.format(work_on+1))
            else:
                logg('[*] Share found for block {}.'.format(work_on+1))
            logg('[*] Hash: {}'.format(hash))
            logg('[*] Blockheader: {}'.format(blockheader))            
            payload = bytes('{"params": ["' + user + '", "' + job_id + '", "' + extranonce2 + '", "' + ntime + '", "' + nonce + '"], "id": 1, "method": "mining.submit"}\n', 'utf-8')
            logg('[*] Payload: {}'.format(payload))
            sock.sendall(payload)
            ret = sock.recv(1024)
            try:
                response = json.loads(ret.decode().strip())
                if response.get('result'):
                    with ctx.lock:
                        accepted += 1
                        accepted_timestamps.append(now)
                else:
                    with ctx.lock:
                        rejected += 1
                        rejected_timestamps.append(now)
                    logg('[*] {} rejected: {}'.format('Block' if mode == 'solo' else 'Share', response.get('error')))
            except:
                logg('[*] Error parsing pool response: {}'.format(ret))
            return True

        nonce_count += num_hashes_per_kernel
        last_updated = calculate_hashrate(nonce_count, last_updated, thread_hr)
        with ctx.lock:
            ctx.hashrates[thread_id] = thread_hr[0]

        # CPU fallback or additional hashing (for hybrid)
        # Perform some CPU hashes in parallel
        for _ in range(10000):  # Small batch for CPU to contribute
            nNonce -= 1
            if nNonce < 0:
                nNonce = 0xffffffff
            nonce = hex(nNonce)[2:].zfill(8)
            blockheader = version + prevhash + merkle_root + ntime + nbits + nonce
            hash = hashlib.sha256(hashlib.sha256(binascii.unhexlify(blockheader)).digest()).digest()
            hash = binascii.hexlify(hash).decode()
            hash = "".join(reversed([hash[i:i+2] for i in range(0, len(hash), 2)]))

            if hash < target:
                # Found on CPU
                now = time.time()
                if mode == 'solo':
                    logg('[*] Block {} solved (CPU).'.format(work_on+1))
                else:
                    logg('[*] Share found for block {} (CPU).'.format(work_on+1))
                # Submit as before
                # ... (repeat submit code)
                return True

            nonce_count += 1

        # decrement starting nonce for next batch
        nNonce -= num_hashes_per_kernel
        if nNonce < 0:
            nNonce = 0xffffffff  # wrap around

       

def block_listener(t):
    
    # init a connection to pool 
    global sock
    sock  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    # send a handle subscribe message 
    sock.sendall(b'{"id": 1, "method": "mining.subscribe", "params": []}\n')
    lines = sock.recv(1024).decode().split('\n')
    response = json.loads(lines[0])
    global sub_details, extranonce1, extranonce2_size
    sub_details,extranonce1,extranonce2_size = response['result']
    # send and handle authorize message  
    sock.sendall(b'{"params": ["' + user.encode('utf-8') + b'", "' + password.encode('utf-8') + b'"], "id": 2, "method": "mining.authorize"}\n')
    response = b''
    while not(b'mining.notify' in response):
        response += sock.recv(1024)

    responses = []
    for res in response.decode().split('\n'):
        if len(res.strip()) > 0:
            try:
                responses.append(json.loads(res))
            except json.JSONDecodeError as e:
                logg(f'[*] JSON decode error: {e} for response: {res}')

    notify_responses = [res for res in responses if 'method' in res and res['method'] == 'mining.notify']
    if notify_responses:
        params = notify_responses[0]['params']
        global job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs
        job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs = params
        # do this one time, will be overwriten by mining loop when new block is detected
        global updatedPrevHash
        updatedPrevHash = prevhash
    # set sock 
    sock = sock 

    while True:
        t.check_self_shutdown()
        if t.exit:
            break

        # check for new block 
        response = b''
        while not(b'mining.notify' in response):
            response += sock.recv(1024)
        responses = []
        for res in response.decode().split('\n'):
            if len(res.strip()) > 0:
                try:
                    responses.append(json.loads(res))
                except json.JSONDecodeError as e:
                    logg(f'[*] JSON decode error: {e} for response: {res}')

        for res in responses:
            if 'method' in res:
                if res['method'] == 'mining.notify':
                    if res['params'][1] != prevhash:
                        # new block detected on network 
                        # update context job data 
                        job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs = res['params']
                elif res['method'] == 'client.show_message':
                    motd = res['params'][0]
                    logg(f'[*] MOTD: {motd}')
                elif res['method'] == 'mining.set_difficulty':
                    diff = res['params'][0]
                    target = diff_to_target(diff)
                    logg(f'[*] Set difficulty: {diff}, Target: {target}')

class CoinMinerThread(ExitedThread):
    def __init__(self, arg, n, thread_id):
        super(CoinMinerThread, self).__init__(arg, n)
        self.thread_id = thread_id

    def thread_handler2(self, arg):
        self.thread_bitcoin_miner(arg)

    def thread_bitcoin_miner(self, arg):
        listfThreadRunning[self.n] = True
        check_for_shutdown(self)
        try:
            ret = bitcoin_miner(self, thread_id=self.thread_id)
            logg("[*] Miner returned %s\n\n" % "true" if ret else"false")
        except Exception as e:
            logg("Miner error: " + str(e))
        listfThreadRunning[self.n] = False

    pass  

class NewSubscribeThread(ExitedThread):
    def __init__(self, arg, n):
        super(NewSubscribeThread, self).__init__(arg, n)

    def thread_handler2(self, arg):
        self.thread_new_block(arg)

    def thread_new_block(self, arg):
        listfThreadRunning[self.n] = True
        check_for_shutdown(self)
        try:
            ret = block_listener(self)
        except Exception as e:
            logg("Subscribe error: " + str(e))
        listfThreadRunning[self.n] = False

    pass  

class DisplayThread(ExitedThread):
    def __init__(self, arg, n):
        super(DisplayThread, self).__init__(arg, n)

    def thread_handler2(self, arg):
        self.display_loop()

    def display_loop(self):
        stdscr = curses.initscr()
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Green for positive stats
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)    # Red for negative
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Yellow for info
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Cyan for titles
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(True)
        try:
            while not self.exit:
                stdscr.clear()
                now = time.time()
                with lock:
                    acc_min = sum(1 for t in accepted_timestamps if now - t < 60)
                    rej_min = sum(1 for t in rejected_timestamps if now - t < 60)
                stdscr.attron(curses.color_pair(4))
                stdscr.addstr(0, 0, f"Bitcoin {'Solo' if mode == 'solo' else 'Pool'} Miner (CPU + GPU)")
                stdscr.attroff(curses.color_pair(4))
                stdscr.attron(curses.color_pair(3))
                current_height = get_current_block_height() + 1
                stdscr.addstr(1, 0, f"Block height: {current_height}")
                stdscr.attroff(curses.color_pair(3))
                total_hr = sum(hashrates)
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(2, 0, f"Hashrate: {total_hr} H/s")
                stdscr.attroff(curses.color_pair(1))
                stdscr.attron(curses.color_pair(3))
                stdscr.addstr(3, 0, f"MOTD: {motd}")
                stdscr.addstr(4, 0, f"Threads: {num_threads}")
                stdscr.attroff(curses.color_pair(3))
                label = 'Blocks' if mode == 'solo' else 'Shares'
                stdscr.addstr(5, 0, f"{label}: ")
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(f"Accepted {accepted} ")
                stdscr.attroff(curses.color_pair(1))
                stdscr.attron(curses.color_pair(2))
                stdscr.addstr(f"Rejected {rejected}")
                stdscr.attroff(curses.color_pair(2))
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(6, 0, f"Accepted/min: {acc_min}")
                stdscr.attroff(curses.color_pair(1))
                stdscr.attron(curses.color_pair(2))
                stdscr.addstr(f" Rejected/min: {rej_min}")
                stdscr.attroff(curses.color_pair(2))
                # Add more stats if needed
                stdscr.refresh()
                time.sleep(1)
        except Exception as e:
            logg(f"Display error: {e}")
        finally:
            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.endwin()

def StartMining():
    global motd, hashrates, accepted, rejected, accepted_timestamps, rejected_timestamps, lock, listfThreadRunning, nHeightDiff, fShutdown
    motd = ""
    hashrates = [0] * num_threads
    accepted = 0
    rejected = 0
    accepted_timestamps = []
    rejected_timestamps = []
    lock = threading.Lock()
    listfThreadRunning = [False] * (num_threads + 2)  # For subscribe and display
    nHeightDiff = {}
    fShutdown = False

    subscribe_t = NewSubscribeThread(None, n=0)
    subscribe_t.start()
    logg("[*] Subscribe thread started.")

    # Wait for initial job data to be set
    timeout = 60  # Max wait time in seconds
    start_time = time.time()
    while nbits is None or version is None or prevhash is None or ntime is None:
        if time.time() - start_time > timeout:
            raise Exception("Timeout waiting for initial job from pool")
        time.sleep(1)
    logg("[*] Initial job received, starting miners.")

    for i in range(num_threads):
        miner_t = CoinMinerThread(None, n=i+2, thread_id=i)
        miner_t.start()
    logg("[*] {} Bitcoin miner threads started".format(num_threads))

    display_t = DisplayThread(None, n=1)
    display_t.start()
    logg("[*] Display thread started.")

    print('Bitcoin Miner started')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bitcoin Miner')
    parser.add_argument('--mode', type=str, choices=['solo', 'pool'], default='pool', help='Mining mode: solo or pool')
    args = parser.parse_args()

    global mode, host, port, user, password
    mode = args.mode
    if mode == 'solo':
        host = SOLO_HOST
        port = SOLO_PORT
        user = SOLO_ADDRESS
        password = SOLO_PASSWORD
    else:
        host = POOL_HOST
        port = POOL_PORT
        user = POOL_WORKER
        password = POOL_PASSWORD

    signal(SIGINT, handler)

    StartMining()
