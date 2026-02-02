"""Simple RTSP viewer that subscribes to the UDP inference server and overlays detections.

Usage (examples):
  uv run viewer.py rtsp://localhost:9192/topic1 127.0.0.1 55055 56070
  uv run viewer.py /dev/video0 127.0.0.1 55055 56071

Arguments:
  source       - camera index, device path, or stream URL
  server_host  - UDP server host (default: 127.0.0.1)
  server_port  - UDP server port (default: 55055)
  recv_port    - local UDP port to bind and receive inference datagrams (default: 56060)

Behavior:
  - Sends START|<source>|<recv_port> to the server to subscribe.
  - Sends periodic HEARTBEAT messages to keep the subscription alive.
  - Receives JSON detection datagrams on recv_port on a background thread and
    keeps the latest detections in memory.
  - Main thread reads frames from the source, overlays latest detections, and
    displays them in a window.
"""

from __future__ import annotations

import sys
import time
import json
import socket
import threading
from typing import Any, Dict, List, Tuple

import cv2
from loguru import logger


def safe_int(v: str, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


class UDPSubscriber:
    def __init__(self, server_host: str, server_port: int, recv_port: int, source: str):
        self.server_addr = (server_host, server_port)
        self.recv_port = recv_port
        self.source = source
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.sock.bind(("0.0.0.0", self.recv_port))
        self.sock.settimeout(1.0)

        self.running = threading.Event()
        self.running.set()
        self.lock = threading.Lock()
        self.latest: Dict[str, Any] = {"detections": []}

    def start(self) -> None:
        # send START message
        start_msg = f"START|{self.source}|{self.recv_port}"
        self._send(start_msg)
        # start receiver thread
        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()
        # start heartbeat thread
        self.hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.hb_thread.start()

    def stop(self) -> None:
        self.running.clear()
        # send STOP
        stop_msg = f"STOP|{self.source}|{self.recv_port}"
        try:
            self._send(stop_msg)
        except Exception:
            pass
        try:
            self.recv_thread.join(timeout=1.0)
        except Exception:
            pass

    def _send(self, msg: str) -> None:
        self.sock.sendto(msg.encode("utf-8"), self.server_addr)

    def _heartbeat_loop(self) -> None:
        while self.running.is_set():
            try:
                hb_msg = f"HEARTBEAT|{self.source}|{self.recv_port}"
                self._send(hb_msg)
                logger.debug("[Viewer][HB] sent heartbeat")
            except Exception as e:
                logger.exception("Heartbeat send failed: {err}", err=e)
            time.sleep(3)

    def _recv_loop(self) -> None:
        while self.running.is_set():
            try:
                data, _ = self.sock.recvfrom(65536)
                try:
                    payload = json.loads(data.decode("utf-8"))
                except Exception:
                    logger.warning("Received non-json payload")
                    continue
                with self.lock:
                    self.latest = payload
            except socket.timeout:
                continue
            except Exception as e:
                logger.exception("UDP recv error: {err}", err=e)

    def get_latest(self) -> Dict[str, Any]:
        with self.lock:
            return self.latest.copy()


def draw_detections(frame, detections: List[Dict[str, Any]]) -> None:
    h, w = frame.shape[:2]
    for det in detections:
        try:
            cls = det.get("class", "obj")
            score = det.get("score", 0.0)
            loc = det.get("location")
            if not loc or len(loc) < 4:
                continue
            # location is list of four points (x,y)
            x1, y1 = map(int, loc[0])
            x2, y2 = map(int, loc[2])
            # clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            color = (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {score:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - t_size[1] - 4), (x1 + t_size[0] + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        except Exception:
            continue


def main(argv: List[str]) -> None:
    source = argv[1] if len(argv) > 1 else "0"
    server_host = argv[2] if len(argv) > 2 else "127.0.0.1"
    server_port = safe_int(argv[3], 55055) if len(argv) > 3 else 55055
    recv_port = safe_int(argv[4], 56060) if len(argv) > 4 else 56060

    subscriber = UDPSubscriber(server_host, server_port, recv_port, source)
    subscriber.start()

    # Open video source
    cap = None
    try:
        try:
            src = int(source)
            cap = cv2.VideoCapture(src)
        except Exception:
            src = str(source)
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap or not cap.isOpened():
            logger.error("Unable to open video source %s", source)
            return

        window = "inferencer-viewer"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed; retrying shortly")
                time.sleep(0.1)
                continue

            latest = subscriber.get_latest()
            detections = latest.get("detections", []) if isinstance(latest, dict) else []
            draw_detections(frame, detections)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        try:
            subscriber.stop()
        except Exception:
            pass
        if cap:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
