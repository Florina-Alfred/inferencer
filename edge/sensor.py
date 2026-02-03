#!/usr/bin/env python3
"""Simple sensor client: captures camera frames, base64-encodes, sends to server, logs responses."""
from __future__ import annotations

import argparse
import base64
import time
import queue
import threading

import cv2
import grpc
import numpy as np
from loguru import logger
import sys

# configure loguru
logger.remove()
logger.add(sys.stderr, level="INFO", enqueue=True)

import sys
import os
# ensure repo root on sys.path so package imports work when executed directly
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from edge.proto import detection_pb2 as pb
    from edge.proto import detection_pb2_grpc as pb_grpc
except Exception as e:  # pragma: no cover - helpful
    raise RuntimeError('gRPC stubs missing: run `python generate.py` in edge/ (and run from repo root)') from e


def jpeg_b64_from_frame(frame: np.ndarray, quality: int = 70) -> str:
    ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ret:
        raise RuntimeError('jpeg encoding failed')
    return base64.b64encode(buf.tobytes()).decode('ascii')


def capture_thread(cap, q: queue.Queue, stop_event: threading.Event, width: int, height: int, quality: int):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        small = cv2.resize(frame, (width, height))
        try:
            b64 = jpeg_b64_from_frame(small, quality)
        except Exception:
            continue
        req = pb.ImageRequest(source='cam0', image_b64=b64, width=width, height=height, timestamp_ms=int(time.time()*1000), seq=int(time.time()*1000))
        # drop previous if queue full
        try:
            q.put(req, timeout=0.1)
        except queue.Full:
            try:
                _ = q.get_nowait()
                q.put(req, timeout=0.1)
            except queue.Empty:
                pass
        time.sleep(0.03)


def request_generator(q: queue.Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            req = q.get(timeout=0.2)
        except queue.Empty:
            continue
        yield req


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', default=50051, type=int)
    p.add_argument('--width', default=320, type=int)
    p.add_argument('--height', default=240, type=int)
    p.add_argument('--quality', default=70, type=int)
    p.add_argument('--source', default=0)
    args = p.parse_args()

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        raise RuntimeError('failed to open capture source')

    q: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    t = threading.Thread(target=capture_thread, args=(cap, q, stop_event, args.width, args.height, args.quality), daemon=True)
    t.start()

    channel = grpc.insecure_channel(f"{args.host}:{args.port}", options=[('grpc.max_send_message_length', 50*1024*1024), ('grpc.max_receive_message_length', 50*1024*1024)])
    stub = pb_grpc.InferenceStub(channel)

    responses = stub.Stream(request_generator(q, stop_event))
    try:
        for resp in responses:
            # simple logging of detections via loguru
            try:
                found_count = len(resp.found)
            except Exception:
                found_count = 0
            logger.info("Received DetectionResponse source={} seq={} processing_ms={} found={}", resp.source, getattr(resp, 'seq', 0), getattr(resp, 'processing_ms', 0), found_count)
            for d in resp.found:
                try:
                    logger.info("Detection: class_name={} class_id={} score={} bbox={}..{}", d.class_name, d.class_id, d.score, (d.xmin, d.ymin), (d.xmax, d.ymax))
                except Exception:
                    logger.exception("Malformed detection message")
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cap.release()


if __name__ == '__main__':
    main()
