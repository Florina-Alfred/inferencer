import socket
import threading
import time
import sys
import json

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 55055
CLIENT_RECEIVE_PORT = 56060  # Change as needed
SOURCE = sys.argv[1] if len(sys.argv) > 1 else '0'

recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
recv_sock.bind(('0.0.0.0', CLIENT_RECEIVE_PORT))
recv_sock.settimeout(1.0)

server_addr = (SERVER_HOST, SERVER_PORT)
client_addr = ('127.0.0.1', CLIENT_RECEIVE_PORT)

# Send START
start_msg = f'START|{SOURCE}|{CLIENT_RECEIVE_PORT}'
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_sock.sendto(start_msg.encode(), server_addr)

running = True
def heartbeat():
    while running:
        hb_msg = f'HEARTBEAT|{SOURCE}'
        send_sock.sendto(hb_msg.encode(), server_addr)
        time.sleep(3)

# Start heartbeat thread
threading.Thread(target=heartbeat, daemon=True).start()

print(f"[Client] Listening for inference results on port {CLIENT_RECEIVE_PORT}...")
received_count = 0
try:
    while True:
        try:
            msg, _ = recv_sock.recvfrom(65536)
            result = json.loads(msg.decode('utf-8'))
            received_count += 1
            print(f"[Client] Received result {received_count}: {json.dumps(result)[:200]}")
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            break
finally:
    running = False
    # Send STOP to server
    stop_msg = f'STOP|{SOURCE}'
    send_sock.sendto(stop_msg.encode(), server_addr)
    recv_sock.close()
    send_sock.close()
    print("[Client] Client shut down.")

