#!/usr/bin/env python3
"""
Minimal webcam client that streams downscaled JPEG frames (base64) to the gRPC server
and displays received detections overlaid on the camera feed.
"""
import argparse
import base64
import queue
import threading
import time

import cv2
import grpc
import numpy as np
import os
import sys

# Ensure repo root is on sys.path so `from edge.proto import ...` works even when this
# file is executed as `python edge/sensor.py` (which places `edge/` on sys.path[0]).
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import importlib
try:
    # Protect against protobuf runtime/gencode mismatches which raise at import time
    try:
        import google.protobuf.runtime_version as _pb_runtime
        _orig_validate = getattr(_pb_runtime, "ValidateProtobufRuntimeVersion", None)

        def _safe_validate(domain, major, minor, patch, suffix, filename):
            try:
                if _orig_validate is not None:
                    return _orig_validate(domain, major, minor, patch, suffix, filename)
            except Exception:
                import warnings

                warnings.warn(f"Ignored protobuf runtime/gencode mismatch during import")
                return None

        setattr(_pb_runtime, "ValidateProtobufRuntimeVersion", _safe_validate)
    except Exception:
        # ignore if protobuf runtime isn't present; let subsequent imports show useful errors
        pass

    # Import package-qualified proto module first, then alias bare name so generated
    # grpc files using `import infer_pb2` resolve correctly.
    infer_pb2 = importlib.import_module("edge.proto.infer_pb2")
    sys.modules.setdefault("infer_pb2", infer_pb2)
    infer_pb2_grpc = importlib.import_module("edge.proto.infer_pb2_grpc")
except Exception:
    raise RuntimeError("gRPC stubs not found. Run `cd edge && make gen` (requires grpcio-tools).")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", enqueue=True, backtrace=False, diagnose=False)


def jpeg_b64_from_frame(frame: np.ndarray, quality: int = 70) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ret, buf = cv2.imencode(".jpg", frame, encode_param)
    if not ret:
        raise RuntimeError("jpeg encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def capture_thread_fn(cap, req_q: queue.Queue, stop_event: threading.Event, width: int, height: int, quality: int, last_frame_container: dict, lock: threading.Lock):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        # store the full-resolution frame for display
        try:
            with lock:
                last_frame_container['frame'] = frame.copy()
        except Exception:
            # best-effort; don't fail sending if copy fails
            last_frame_container['frame'] = frame
        frame_small = cv2.resize(frame, (width, height))
        try:
            b64 = jpeg_b64_from_frame(frame_small, quality=quality)
        except Exception:
            continue
        # attach a per-frame sequence id for RTT correlation
        seq = int(time.time() * 1000)  # use ms timestamp as simple sequence id
        req = infer_pb2.ImageRequest(
            source="cam0",
            image_b64=b64,
            width=width,
            height=height,
            timestamp_ms=int(time.time() * 1000),
            seq=seq,
        )
        # keep only the most recent frame to reduce queueing
        try:
            req_q.put(req, timeout=0.1)
        except queue.Full:
            try:
                _ = req_q.get_nowait()
                req_q.put(req, timeout=0.1)
            except queue.Empty:
                pass
        # small sleep to bound frame rate
        time.sleep(0.03)


def request_generator(req_q: queue.Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            req = req_q.get(timeout=0.2)
        except queue.Empty:
            continue
        yield req


def overlay_detections(frame: np.ndarray, detections):
    h, w = frame.shape[:2]
    for det in detections:
        # det fields are protobuf message attributes; use getattr to avoid crashes
        try:
            xmin = float(getattr(det, 'xmin'))
            ymin = float(getattr(det, 'ymin'))
            xmax = float(getattr(det, 'xmax'))
            ymax = float(getattr(det, 'ymax'))
            cls_id = int(getattr(det, 'class_id'))
            score = float(getattr(det, 'score'))
            class_name = getattr(det, 'class_name', '')
            if class_name:
                label_text = f"{class_name}:{score:.2f}"
            else:
                label_text = f"{cls_id}:{score:.2f}"
        except Exception:
            # fallback: skip malformed detection
            continue
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--source", default=0, help="cv2 capture source (int) or path")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--quality", type=int, default=70)
    args = parser.parse_args()

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        raise RuntimeError("failed to open capture source")

    req_q = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    last_frame_container: dict = {'frame': None}
    last_frame_lock = threading.Lock()
    t = threading.Thread(
        target=capture_thread_fn,
        args=(cap, req_q, stop_event, args.width, args.height, args.quality, last_frame_container, last_frame_lock),
        daemon=True,
    )
    t.start()

    channel = grpc.insecure_channel(
        f"{args.host}:{args.port}",
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )
    stub = infer_pb2_grpc.InferenceStub(channel)

    # generator yields requests from the queue; responses come back as an iterator
    responses = stub.Stream(request_generator(req_q, stop_event), compression=grpc.Compression.Gzip)

    try:
        for resp in responses:
            # use the latest full-resolution frame captured by the capture thread
            with last_frame_lock:
                frame = last_frame_container.get('frame')
            if frame is None:
                # no frame yet
                continue
            if resp.detections:
                overlay_detections(frame, resp.detections)
            # show human-friendly timestamp + RTT if seq present
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(resp.timestamp_ms / 1000)) if getattr(resp, 'timestamp_ms', 0) else ''
            rtt_ms = None
            if getattr(resp, 'seq', 0):
                # calculate RTT using local wallclock (req.seq used as client-side send time in ms)
                rtt_ms = int(time.time() * 1000) - int(resp.seq)
            rtt_text = f" rtt:{rtt_ms}ms" if rtt_ms is not None else ''
            processing_text = f" proc:{getattr(resp, 'processing_ms', 0)}ms" if getattr(resp, 'processing_ms', None) is not None else ''
            # position overlay at top-left of the full-resolution display frame
            display_text = f"{ts}{rtt_text}{processing_text}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            # scale text with frame width, clamp to reasonable range
            font_scale = max(0.4, min(frame.shape[1] / 1280.0, frame.shape[0] / 720.0))
            thickness = max(1, int(round(font_scale * 2)))
            (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
            pad = 6
            x, y = 10, 10 + text_h
            # draw background rectangle for readability
            cv2.rectangle(frame, (x - pad, y - text_h - pad), (x + text_w + pad, y + pad), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, display_text, (x, y), font, font_scale, (255, 255, 0), thickness)
            # client-side logging of RTT and processing time
            if rtt_ms is not None:
                logger.info("frame seq={} rtt_ms={} processing_ms={} detections={}", getattr(resp, 'seq', 0), rtt_ms, getattr(resp, 'processing_ms', 0), len(resp.detections))
            cv2.imshow("sensor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
