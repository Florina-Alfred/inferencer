import socket
import threading
import time
import sys
import json
from loguru import logger

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 55055
# Allow specifying receive port as optional second arg so multiple clients can run on one host
CLIENT_RECEIVE_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 56060
SOURCE = sys.argv[1] if len(sys.argv) > 1 else '0'

recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
try:
    recv_sock.bind(('0.0.0.0', CLIENT_RECEIVE_PORT))
except OSError as bind_err:
    # Address already in use -> fall back to ephemeral port
    logger.warning("Port {port} in use, falling back to ephemeral port", port=CLIENT_RECEIVE_PORT)
    recv_sock.bind(('0.0.0.0', 0))
recv_sock.settimeout(1.0)

actual_port = recv_sock.getsockname()[1]
server_addr = (SERVER_HOST, SERVER_PORT)
client_addr = ('127.0.0.1', actual_port)

# Send START (include actual bound port so server can respond correctly)
start_msg = f'START|{SOURCE}|{actual_port}'
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_sock.sendto(start_msg.encode(), server_addr)

running = True
def heartbeat():
    count = 0
    while running:
        count += 1
        try:
            # Include the client receive port so the server can match the job key
            hb_msg = f'HEARTBEAT|{SOURCE}|{CLIENT_RECEIVE_PORT}'
            send_sock.sendto(hb_msg.encode(), server_addr)
            logger.debug("[Client][Heartbeat] Sent HEARTBEAT {count} for source {src}", count=count, src=SOURCE)
        except Exception as hb_err:
            logger.exception("[Client][Heartbeat] ERROR sending heartbeat: {err}", err=hb_err)
        time.sleep(3)

# Start heartbeat thread
threading.Thread(target=heartbeat, daemon=True).start()

logger.info("[Client] Listening for inference results on port {port}...", port=CLIENT_RECEIVE_PORT)
received_count = 0
try:
    while True:
        try:
            msg, _ = recv_sock.recvfrom(65536)
            result = json.loads(msg.decode('utf-8'))
            received_count += 1
            logger.info("[Client] Received result {n}: {payload}", n=received_count, payload=json.dumps(result)[:200])
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            break
finally:
    running = False
    # Send STOP to server (include receive port so server kills correct job)
    stop_msg = f'STOP|{SOURCE}|{CLIENT_RECEIVE_PORT}'
    try:
        send_sock.sendto(stop_msg.encode(), server_addr)
    except Exception:
        logger.exception("[Client] Error sending STOP to server")
    recv_sock.close()
    send_sock.close()
    logger.info("[Client] Client shut down.")
