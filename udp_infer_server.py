import socket
import multiprocessing
import threading
import time
import signal
import sys
import os

# Ensure the script can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_worker import run_inference_stream

# Defaults/server config
LISTEN_HOST = '0.0.0.0'
LISTEN_PORT = 55055
HEARTBEAT_TIMEOUT = 10  # seconds

# Format: (client_addr, client_port, source) -> {'proc': Process, 'last_seen': float}
active_jobs = {}

def job_key(addr, port, source):
    return (addr, port, str(source))

def kill_job(key):
    entry = active_jobs.get(key)
    if entry and entry['proc'].is_alive():
        entry['proc'].terminate()
        entry['proc'].join()
    active_jobs.pop(key, None)

def heartbeat_cleaner():
    while True:
        now = time.time()
        kills = []
        for key, entry in list(active_jobs.items()):
            if now - entry['last_seen'] > HEARTBEAT_TIMEOUT:
                print(f"[Server] Killing job {key} due to heartbeat timeout.")
                kills.append(key)
        for key in kills:
            kill_job(key)
        time.sleep(1)

def udp_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_HOST, LISTEN_PORT))
    print(f"[Server] UDP inference server listening on {LISTEN_HOST}:{LISTEN_PORT}")

    # Start heartbeat monitor
    threading.Thread(target=heartbeat_cleaner, daemon=True).start()

    while True:
        msg, (client_addr, client_port) = sock.recvfrom(4096)
        try:
            msg = msg.decode('utf-8').strip()
        except Exception:
            print(f"[Server] Received undecodable message from {client_addr}:{client_port}")
            continue

        # Protocol: COMMAND|source|params (simple split)
        fields = msg.split('|')
        if not fields:
            continue
        cmd = fields[0].upper()
        source = fields[1] if len(fields) > 1 else None
        # Allow explicit client receive port
        dest_port = int(fields[2]) if len(fields) > 2 and fields[2].isdigit() else client_port
        key = job_key(client_addr, dest_port, source)

        if cmd == 'START' and source:
            print(f"[Server][DEBUG] Received START command from {client_addr}:{client_port} with source: {source}, dest_port: {dest_port}")
            # Already running?
            if key in active_jobs and active_jobs[key]['proc'].is_alive():
                active_jobs[key]['last_seen'] = time.time()  # Refresh
                print(f"[Server] Refreshed job {key}")
            else:
                print(f"[Server] Launching new job for {key}")
                # Fork worker
                proc = multiprocessing.Process(
                    target=run_inference_stream,
                    args=(source, client_addr, dest_port),
                    kwargs={},
                    daemon=True
                )
                print(f"[Server][DEBUG] Starting Process with args: {(source, client_addr, dest_port)}")
                proc.start()
                active_jobs[key] = {
                    'proc': proc,
                    'last_seen': time.time(),
                }
        elif cmd == 'STOP' and source:
            print(f"[Server] Stopping job {key}")
            kill_job(key)
        elif cmd == 'HEARTBEAT' and source:
            # Just refresh
            if key in active_jobs:
                active_jobs[key]['last_seen'] = time.time()
        else:
            print(f"[Server] Unknown command from {client_addr}:{client_port}: {msg}")

def shutdown_handler(signum, frame):
    print("[Server] Shutting down, killing all jobs.")
    for key in list(active_jobs):
        kill_job(key)
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    udp_server()

