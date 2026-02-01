import socket
import multiprocessing
import threading
import time
import signal
import sys
import os
from loguru import logger

# Ensure the script can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_worker import run_inference_stream

# Defaults/server config
LISTEN_HOST = '0.0.0.0'
LISTEN_PORT = 55055
HEARTBEAT_TIMEOUT = 10  # seconds

# Format: source -> {'proc': Process, 'queue': Queue, 'clients': {(addr,port): last_seen}}
active_jobs = {}

def job_key(source: str) -> str:
    return str(source)

def kill_job(key):
    entry = active_jobs.get(key)
    if entry and entry.get("proc") and entry["proc"].is_alive():
        logger.info("Terminating job {key}", key=key)
        # Signal forwarder/worker to shutdown via the queue if present
        q = entry.get("queue")
        try:
            if q:
                q.put(None)
        except Exception:
            pass
        entry["proc"].terminate()
        entry["proc"].join()
    active_jobs.pop(key, None)

def heartbeat_cleaner():
    while True:
        now = time.time()
        # For each job (source), remove clients whose heartbeat expired. If no clients remain, kill the job.
        kills = []
        for key, entry in list(active_jobs.items()):
            clients = entry.get("clients", {})
            stale = []
            for client, ts in list(clients.items()):
                if now - ts > HEARTBEAT_TIMEOUT:
                    logger.warning("Removing client {client} from job {key} due to heartbeat timeout", client=client, key=key)
                    stale.append(client)
            for c in stale:
                clients.pop(c, None)
            if not clients:
                logger.warning("No active clients for job {key}; terminating worker.", key=key)
                kills.append(key)
        for key in kills:
            kill_job(key)
        time.sleep(1)

def udp_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_HOST, LISTEN_PORT))
    logger.info("UDP inference server listening on {host}:{port}", host=LISTEN_HOST, port=LISTEN_PORT)

    # Start heartbeat monitor
    threading.Thread(target=heartbeat_cleaner, daemon=True).start()

    # Helper: per-job thread that forwards worker queue messages to all clients
    def server_queue_forwarder(job_key):
        entry = active_jobs.get(job_key)
        if not entry:
            return
        q = entry.get("queue")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            while True:
                item = q.get()
                if item is None:
                    logger.info("Worker for job {key} signalled shutdown", key=job_key)
                    break
                frame_count, message = item
                clients = list(entry.get("clients", {}).keys())
                for (addr, port) in clients:
                    try:
                        sock.sendto(message, (addr, port))
                        logger.debug("Forwarded frame {f} to {addr}:{port}", f=frame_count, addr=addr, port=port)
                    except Exception as e:
                        logger.exception("Error forwarding to {addr}:{port}: {err}", addr=addr, port=port, err=e)
        finally:
            sock.close()

    while True:
        msg, (client_addr, client_port) = sock.recvfrom(4096)
        try:
            msg = msg.decode('utf-8').strip()
        except Exception:
            logger.warning("Received undecodable message from {addr}:{port}", addr=client_addr, port=client_port)
            continue

        # Protocol: COMMAND|source|params (simple split)
        fields = msg.split('|')
        if not fields:
            continue
        cmd = fields[0].upper()
        source = fields[1] if len(fields) > 1 else None
        if not source:
            logger.warning("Received command without source from {addr}:{port}: {msg}", addr=client_addr, port=client_port, msg=msg)
            continue
        # Allow explicit client receive port
        dest_port = int(fields[2]) if len(fields) > 2 and fields[2].isdigit() else client_port
        key = job_key(source)
        client = (client_addr, dest_port)

        if cmd == 'START' and source:
            logger.debug(
                "Received START command from {addr}:{port} with source: {src}, dest_port: {dport}",
                addr=client_addr,
                port=client_port,
                src=source,
                dport=dest_port,
            )
            # Already running? If so add this client; otherwise launch worker and forwarder
            if key in active_jobs and active_jobs[key].get("proc") and active_jobs[key]["proc"].is_alive():
                active_jobs[key]["clients"][client] = time.time()
                logger.debug("Added/Refreshed client {client} for job {key}", client=client, key=key)
            else:
                logger.info("Launching new job for {key}", key=key)
                out_queue = multiprocessing.Queue()
                proc = multiprocessing.Process(
                    target=run_inference_stream,
                    args=(source, out_queue),
                    kwargs={},
                    daemon=True,
                )
                logger.debug("Starting Process for source: {src}", src=source)
                proc.start()
                active_jobs[key] = {
                    "proc": proc,
                    "queue": out_queue,
                    "clients": {client: time.time()},
                }
                threading.Thread(target=server_queue_forwarder, args=(key,), daemon=True).start()
        elif cmd == 'STOP' and source:
            logger.info("STOP received for client {client} and job {key}", client=client, key=key)
            if key in active_jobs:
                clients = active_jobs[key].get("clients", {})
                clients.pop(client, None)
                if not clients:
                    kill_job(key)
                else:
                    logger.debug("Removed client {client}; remaining clients: {n}", client=client, n=len(clients))
        elif cmd == 'HEARTBEAT' and source:
            # Just refresh the client's last_seen
            if key in active_jobs:
                active_jobs[key].get("clients", {})[client] = time.time()
                logger.debug("Refreshed heartbeat for client {client} on job {key}", client=client, key=key)
            else:
                logger.debug("Received HEARTBEAT for unknown job {key}", key=key)
        else:
            logger.warning("Unknown command from {addr}:{port}: {msg}", addr=client_addr, port=client_port, msg=msg)

def shutdown_handler(signum, frame):
    logger.info("Shutting down, killing all jobs.")
    for key in list(active_jobs):
        kill_job(key)
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    udp_server()
