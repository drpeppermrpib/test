#!/usr/bin/env python  
# Copyright (c) 2021-2022 iceland
# Copyright (c) 2022-2023 Papa Crouz
# Distributed under the MIT/X11 software license, see the accompanying
# file license http://www.opensource.org/licenses/mit-license.php.

from signal import signal, SIGINT
import context as ctx 
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

# Default configurations
SOLO_HOST = 'solo.ckpool.org'
SOLO_PORT = 3333
SOLO_ADDRESS = 'bc1q0xqv0m834uvgd8fljtaa67he87lzu8mpa37j7e.001'
SOLO_PASSWORD = 'x'  # Arbitrary for solo

POOL_HOST = 'ss.antpool.com'
POOL_PORT = 3333
POOL_WORKER = 'Xk2000.001'
POOL_PASSWORD = 'x'

# Number of mining threads to use (based on your 48 threads)
num_threads = 48  # Adjust if needed, e.g., os.cpu_count() * 2

diff1_target = int('00000000ffff0000000000000000000000000000000000000000000000000000', 16)

def diff_to_target(diff):
    diff = float(diff)
    target_int = int(diff1_target / diff)
    target_hex = format(target_int, '064x')
    return target_hex

def handler(signal_received, frame):
    # Handle any cleanup here
    ctx.fShutdown = True
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
    if ctx.fShutdown:
        if n != -1:
            ctx.listfThreadRunning[n] = False
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
            ctx.listfThreadRunning[n] = True
            try:
                self.thread_handler2(arg)
            except Exception as e:
                logg("ThreadHandler()")
                logg(e)
            ctx.listfThreadRunning[n] = False

            time.sleep(5)
            pass

    def thread_handler2(self, arg):
        raise NotImplementedError("must impl this func")

    def check_self_shutdown(self):
        check_for_shutdown(self)

    def try_exit(self):
        self.exit = True
        ctx.listfThreadRunning[self.n] = False
        pass

def bitcoin_miner(t, restarted=False, thread_id=0):

    if restarted:
        logg('[*] Bitcoin Miner restarted')
        time.sleep(10)

    # Wait for job data if not available
    while ctx.nbits is None or ctx.version is None or ctx.prevhash is None or ctx.ntime is None:
        logg('[*] Waiting for job data in thread {}'.format(thread_id))
        time.sleep(1)

    # Use block target initially
    block_target = (ctx.nbits[2:]+'00'*(int(ctx.nbits[:2],16) - 3)).zfill(64)
    target = ctx.target if hasattr(ctx, 'target') and ctx.mode == 'pool' else block_target

    extranonce2 = hex(random.randint(0,2**32-1))[2:].zfill(2*ctx.extranonce2_size)      # create random

    coinbase = ctx.coinb1 + ctx.extranonce1 + extranonce2 + ctx.coinb2
    coinbase_hash_bin = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()

    merkle_root = coinbase_hash_bin
    for h in ctx.merkle_branch:
        merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + binascii.unhexlify(h)).digest()).digest()

    merkle_root = binascii.hexlify(merkle_root).decode()

    #little endian
    merkle_root = ''.join([merkle_root[i]+merkle_root[i+1] for i in range(0,len(merkle_root),2)][::-1])

    work_on = get_current_block_height()

    ctx.nHeightDiff[work_on+1] = 0 

    _diff = int("00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16)

    

    logg('[*] Working to solve block with height {}'.format(work_on+1))

    random_nonce = False  # Set to sequential for "backwards" mining

    

    nNonce = 0xffffffff  # Start from max for backwards

    nonce_count = 0
    last_updated = time.time()
    thread_hr = [0]  # Local for this thread

    while True:
        t.check_self_shutdown()
        if t.exit:
            break

        if ctx.prevhash != ctx.updatedPrevHash:
            logg('[*] New block {} detected on network '.format(ctx.prevhash))
            logg('[*] Best difficulty will trying to solve block {} was {}'.format(work_on+1, ctx.nHeightDiff[work_on+1]))
            ctx.updatedPrevHash = ctx.prevhash
            bitcoin_miner(t, restarted=True, thread_id=thread_id)
            break 

        # Update target if difficulty changed (for pool)
        target = ctx.target if hasattr(ctx, 'target') and ctx.mode == 'pool' else block_target

        nonce = hex(nNonce)[2:].zfill(8)

        blockheader = ctx.version + ctx.prevhash + merkle_root + ctx.ntime + ctx.nbits + nonce
        
        hash = hashlib.sha256(hashlib.sha256(binascii.unhexlify(blockheader)).digest()).digest()
        hash = binascii.hexlify(hash).decode()
        hash = "".join(reversed([hash[i:i+2] for i in range(0, len(hash), 2)]))

        # Logg all hashes that start with 7 zeros or more
        if hash.startswith('0000000'): logg('[*] New hash: {} for block {}'.format(hash, work_on+1))

        this_hash = int(hash, 16)

        difficulty = _diff / this_hash

        if ctx.nHeightDiff[work_on+1] < difficulty:
            # new best difficulty for block at x height
            ctx.nHeightDiff[work_on+1] = difficulty
        

        nonce_count += 1
        last_updated = calculate_hashrate(nonce_count, last_updated, thread_hr)
        with ctx.lock:
            ctx.hashrates[thread_id] = thread_hr[0]

        if hash < target :
            now = time.time()
            if ctx.mode == 'solo':
                logg('[*] Block {} solved.'.format(work_on+1))
            else:
                logg('[*] Share found for block {}.'.format(work_on+1))
            logg('[*] Hash: {}'.format(hash))
            logg('[*] Blockheader: {}'.format(blockheader))            
            payload = bytes('{"params": ["'+ctx.user+'", "'+ctx.job_id+'", "'+extranonce2 \
                +'", "'+ctx.ntime+'", "'+nonce+'"], "id": 1, "method": "mining.submit"}\n', 'utf-8')
            logg('[*] Payload: {}'.format(payload))
            ctx.sock.sendall(payload)
            ret = ctx.sock.recv(1024)
            try:
                response = json.loads(ret.decode().strip())
                if response.get('result'):
                    with ctx.lock:
                        ctx.accepted += 1
                        ctx.accepted_timestamps.append(now)
                else:
                    with ctx.lock:
                        ctx.rejected += 1
                        ctx.rejected_timestamps.append(now)
                    logg('[*] {} rejected: {}'.format('Block' if ctx.mode == 'solo' else 'Share', response.get('error')))
            except:
                logg('[*] Error parsing pool response: {}'.format(ret))
            return True
        
        # decrement nonce for backwards
        nNonce -=1
        if nNonce < 0:
            nNonce = 0xffffffff  # wrap around

       

def block_listener(t):
    
    # init a connection to pool 
    sock  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ctx.host, ctx.port))
    # send a handle subscribe message 
    sock.sendall(b'{"id": 1, "method": "mining.subscribe", "params": []}\n')
    lines = sock.recv(1024).decode().split('\n')
    response = json.loads(lines[0])
    ctx.sub_details,ctx.extranonce1,ctx.extranonce2_size = response['result']
    # send and handle authorize message  
    sock.sendall(b'{"params": ["'+ctx.user.encode()+b'", "'+ctx.password+'"], "id": 2, "method": "mining.authorize"}\n')
    response = b''
    while response.count(b'\n') < 4 and not(b'mining.notify' in response):response += sock.recv(1024)

    responses = [json.loads(res) for res in response.decode().split('\n') if len(res.strip())>0 and 'mining.notify' in res]
    ctx.job_id, ctx.prevhash, ctx.coinb1, ctx.coinb2, ctx.merkle_branch, ctx.version, ctx.nbits, ctx.ntime, ctx.clean_jobs = responses[0]['params']
    # do this one time, will be overwriten by mining loop when new block is detected
    ctx.updatedPrevHash = ctx.prevhash
    # set sock 
    ctx.sock = sock 

    while True:
        t.check_self_shutdown()
        if t.exit:
            break

        # check for new block 
        response = b''
        while response.count(b'\n') < 4 and not(b'mining.notify' in response):response += sock.recv(1024)
        responses = [json.loads(res) for res in response.decode().split('\n') if len(res.strip())>0]     
        
        for res in responses:
            if 'method' in res:
                if res['method'] == 'mining.notify':
                    if res['params'][1] != ctx.prevhash:
                        # new block detected on network 
                        # update context job data 
                        ctx.job_id, ctx.prevhash, ctx.coinb1, ctx.coinb2, ctx.merkle_branch, ctx.version, ctx.nbits, ctx.ntime, ctx.clean_jobs = res['params']
                elif res['method'] == 'client.show_message':
                    ctx.motd = res['params'][0]
                    logg(f'[*] MOTD: {ctx.motd}')
                elif res['method'] == 'mining.set_difficulty':
                    ctx.diff = res['params'][0]
                    ctx.target = diff_to_target(ctx.diff)
                    logg(f'[*] Set difficulty: {ctx.diff}, Target: {ctx.target}')

class CoinMinerThread(ExitedThread):
    def __init__(self, arg, n, thread_id):
        super(CoinMinerThread, self).__init__(arg, n)
        self.thread_id = thread_id

    def thread_handler2(self, arg):
        self.thread_bitcoin_miner(arg)

    def thread_bitcoin_miner(self, arg):
        ctx.listfThreadRunning[self.n] = True
        check_for_shutdown(self)
        try:
            ret = bitcoin_miner(self, thread_id=self.thread_id)
            logg("[*] Miner returned %s\n\n" % "true" if ret else"false")
        except Exception as e:
            logg("[*] Miner()")
            logg(e)
            traceback.print_exc()
        ctx.listfThreadRunning[self.n] = False

    pass  

class NewSubscribeThread(ExitedThread):
    def __init__(self, arg, n):
        super(NewSubscribeThread, self).__init__(arg, n)

    def thread_handler2(self, arg):
        self.thread_new_block(arg)

    def thread_new_block(self, arg):
        ctx.listfThreadRunning[self.n] = True
        check_for_shutdown(self)
        try:
            ret = block_listener(self)
        except Exception as e:
            logg("[*] Subscribe thread()")
            logg(e)
            traceback.print_exc()
        ctx.listfThreadRunning[self.n] = False

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
                with ctx.lock:
                    acc_min = sum(1 for t in ctx.accepted_timestamps if now - t < 60)
                    rej_min = sum(1 for t in ctx.rejected_timestamps if now - t < 60)
                stdscr.attron(curses.color_pair(4))
                stdscr.addstr(0, 0, f"Bitcoin {'Solo' if ctx.mode == 'solo' else 'Pool'} Miner")
                stdscr.attroff(curses.color_pair(4))
                stdscr.attron(curses.color_pair(3))
                current_height = get_current_block_height() + 1
                stdscr.addstr(1, 0, f"Block height: {current_height}")
                stdscr.attroff(curses.color_pair(3))
                total_hr = sum(ctx.hashrates)
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(2, 0, f"Hashrate: {total_hr} H/s")
                stdscr.attroff(curses.color_pair(1))
                stdscr.attron(curses.color_pair(3))
                stdscr.addstr(3, 0, f"MOTD: {ctx.motd}")
                stdscr.addstr(4, 0, f"Threads: {num_threads}")
                stdscr.attroff(curses.color_pair(3))
                label = 'Blocks' if ctx.mode == 'solo' else 'Shares'
                stdscr.addstr(5, 0, f"{label}: ")
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(f"Accepted {ctx.accepted} ")
                stdscr.attroff(curses.color_pair(1))
                stdscr.attron(curses.color_pair(2))
                stdscr.addstr(f"Rejected {ctx.rejected}")
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
    ctx.motd = ""
    ctx.hashrates = [0] * num_threads
    ctx.accepted = 0
    ctx.rejected = 0
    ctx.accepted_timestamps = []
    ctx.rejected_timestamps = []
    ctx.lock = threading.Lock()
    ctx.listfThreadRunning = [False] * (num_threads + 2)  # For subscribe and display
    ctx.nHeightDiff = {}
    ctx.fShutdown = False

    subscribe_t = NewSubscribeThread(None, n=0)
    subscribe_t.start()
    logg("[*] Subscribe thread started.")

    # Wait for initial job data to be set
    timeout = 60  # Max wait time in seconds
    start_time = time.time()
    while ctx.nbits is None or ctx.version is None or ctx.prevhash is None or ctx.ntime is None:
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

    ctx.mode = args.mode
    if ctx.mode == 'solo':
        ctx.host = SOLO_HOST
        ctx.port = SOLO_PORT
        ctx.user = SOLO_ADDRESS
        ctx.password = SOLO_PASSWORD
    else:
        ctx.host = POOL_HOST
        ctx.port = POOL_PORT
        ctx.user = POOL_WORKER
        ctx.password = POOL_PASSWORD

    signal(SIGINT, handler)

    StartMining()
```
