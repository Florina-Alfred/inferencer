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
        # keep a full-resolution copy for display
        try:
            # store full-res for overlay/display in the main thread
            last_frame_container['frame'] = frame.copy()
        except Exception:
            last_frame_container['frame'] = frame
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
    last_frame_container: dict = { 'frame': None }
    last_frame_lock = threading.Lock()
    # pass shared container and lock to capture thread via globals closure
    globals()['last_frame_container'] = last_frame_container
    globals()['last_frame_lock'] = last_frame_lock
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
            # overlay detections on the latest full-resolution frame and show
            with last_frame_lock:
                frame = last_frame_container.get('frame')
                if frame is None:
                    # nothing to display yet
                    continue
                display = frame.copy()
            full_h, full_w = display.shape[:2]
            for d in resp.found:
                try:
                    # Determine coordinate system and scale to full-res frame.
                    # Possible cases from server:
                    # - normalized coords in [0,1] -> multiply by full dims
                    # - small-frame pixel coords (based on args.width/args.height) -> scale by full/small
                    # - already full-frame pixel coords -> use directly
                    def scale_coord(x, axis_max, small_dim):
                        # x: incoming coordinate
                        try:
                            xf = float(x)
                        except Exception:
                            return 0
                        if 0.0 <= xf <= 1.0:
                            # normalized
                            return int(round(xf * axis_max))
                        # if within small dim, assume coords are in small-frame pixels
                        if 0 <= xf <= small_dim:
                            scale = axis_max / float(small_dim)
                            return int(round(xf * scale))
                        # otherwise assume already full pixels
                        return int(round(xf))

                    xmin = scale_coord(d.xmin, full_w, args.width)
                    ymin = scale_coord(d.ymin, full_h, args.height)
                    xmax = scale_coord(d.xmax, full_w, args.width)
                    ymax = scale_coord(d.ymax, full_h, args.height)
                    class_name = d.class_name if getattr(d, 'class_name', '') else str(d.class_id)
                    score = float(getattr(d, 'score', 0.0))
                    # draw box and label
                    cv2.rectangle(display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    label = f"{class_name}:{score:.2f}"
                    cv2.putText(display, label, (xmin, max(ymin - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    logger.info("Detection: class_name={} class_id={} score={} bbox={}..{}", class_name, d.class_id, score, (xmin, ymin), (xmax, ymax))
                except Exception:
                    logger.exception("Malformed detection message")
            # show display
            try:
                cv2.imshow('sensor', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                logger.exception('Failed to display frame')
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
