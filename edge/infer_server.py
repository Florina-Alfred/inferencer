#!/usr/bin/env python3
"""
Minimal gRPC inference server.
Accepts a stream of base64-encoded JPEG images and streams back simple detections.
This implementation uses a placeholder detector that returns a center box for every frame.
"""
import argparse
import base64
import concurrent.futures
import logging
import time

import cv2
import numpy as np
import grpc
import os
import sys
import importlib

# When this script is executed as `python edge/infer_server.py` the module search
# path's first entry becomes the `edge/` directory which prevents importing the
# `edge` package. Ensure the repository root is on sys.path so package imports work
# regardless of invocation style.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    # Some generated proto stubs may have been produced with a different
    # protoc/gencode version than the installed protobuf runtime. The
    # generated `infer_pb2.py` calls
    # `google.protobuf.runtime_version.ValidateProtobufRuntimeVersion(...)`
    # at import time which raises a VersionError when versions mismatch.
    # To avoid forcing regeneration or editing generated files, patch the
    # validator to a safe wrapper that logs and ignores version mismatches.
    try:
        import google.protobuf.runtime_version as _pb_runtime
        _orig_validate = getattr(_pb_runtime, "ValidateProtobufRuntimeVersion", None)

        def _safe_validate(domain, major, minor, patch, suffix, filename):
            try:
                if _orig_validate is not None:
                    return _orig_validate(domain, major, minor, patch, suffix, filename)
            except Exception as _e:  # pragma: no cover - defensive
                import warnings

                warnings.warn(f"Ignored protobuf runtime/gencode mismatch: {_e}")
                return None

        setattr(_pb_runtime, "ValidateProtobufRuntimeVersion", _safe_validate)
    except Exception:
        # If protobuf runtime isn't available for some reason, continue and let
        # the real import error surface; we don't want to hide unrelated import failures.
        pass

    # Import package-qualified modules first
    infer_pb2 = importlib.import_module("edge.proto.infer_pb2")
    # Some generated grpc files use bare `import infer_pb2` which fails when
    # importing as a package. Add an alias in sys.modules so those bare imports
    # resolve to our package module without modifying generated files.
    sys.modules.setdefault("infer_pb2", infer_pb2)
    infer_pb2_grpc = importlib.import_module("edge.proto.infer_pb2_grpc")
except Exception as e:
    raise RuntimeError(
        "gRPC stubs not found in edge/proto. Run `cd edge && python generate_stubs.py`"
    ) from e

import onnxruntime as ort
from coco_classes import COCO_CLASSES
from utils import demo_postprocess, multiclass_nms
from loguru import logger

    # configure loguru: stdout + rotating file sink so logs are visible under uv and in files
logger.remove()
import sys as _sys
logger.add(_sys.stdout, level="INFO", colorize=True, enqueue=True)
logfile = os.path.join(repo_root, "edge", "edge_server.log")
logger.add(logfile, rotation="10 MB", retention="7 days", enqueue=True)

# route standard library logging into loguru so uv/run and other code that uses
# logging.info/debug still appear in our loguru output
class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple shim
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO)

# Simple session cache to avoid recreating ONNX sessions on every frame
_SESSION_CACHE = {}

MODEL_SHORT_TO_NAME = {
    "l": "yolox_l",
    "m": "yolox_m",
    "s": "yolox_s",
    "tiny": "yolox_tiny",
    "nano": "yolox_nano",
}


def get_session(model_name: str, device: str):
    provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    model_base = MODEL_SHORT_TO_NAME.get(model_name, "yolox_l")
    model_path = os.path.join(repo_root, "model", f"{model_base}.onnx")
    key = (model_path, provider)
    if key in _SESSION_CACHE:
        logger.debug("get_session: using cached session {} {}", model_path, provider)
        return _SESSION_CACHE[key], model_base
    # create session (fall back to CPU if provider unavailable)
    try:
        logger.info("get_session: creating session for {} provider={}", model_path, provider)
        sess = ort.InferenceSession(model_path, providers=[provider])
    except Exception:
        logger.exception("get_session: failed to init with provider {}, falling back to CPU", provider)
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _SESSION_CACHE[key] = sess
    return sess, model_base


def infer_frame(session, model_base: str, frame: np.ndarray, conf_threshold: float):
    """Run ONNX session on a single BGR frame and return list of detection records.

    Returns detections as list of dicts matching existing build_detection_records output.
    Coordinates are in absolute pixels (same as build_detection_records expects when given ratios).
    """
    # prepare sizes
    IMAGE_SIZE = 416 if model_base in ["yolox_tiny", "yolox_nano"] else 640
    input_size = (IMAGE_SIZE, IMAGE_SIZE)

    img = cv2.resize(frame, input_size).astype(np.float32)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)

    outputs = session.run(None, {session.get_inputs()[0].name: img})[0]
    logger.debug("infer_frame: raw outputs shape {}", getattr(outputs, 'shape', str(type(outputs))))
    predictions = demo_postprocess(outputs, input_size)[0]
    logger.debug("infer_frame: postprocessed predictions shape {}", getattr(predictions, 'shape', str(type(predictions))))
    if predictions.size == 0:
        logger.info("infer_frame: no predictions after postprocess")
        return []
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    final_boxes, final_scores, final_cls = multiclass_nms(boxes, scores, 0.45, conf_threshold)
    logger.debug(
        "infer_frame: final_boxes={} final_scores={} final_cls={}",
        0 if final_boxes is None else final_boxes.shape[0],
        0 if final_scores is None else (final_scores.shape[0] if hasattr(final_scores, 'shape') else 0),
        0 if final_cls is None else (final_cls.shape[0] if hasattr(final_cls, 'shape') else 0),
    )

    # final_boxes are in model input coordinates (IMAGE_SIZE). Convert to normalized coords (0..1)
    detections = []
    for box, score, cls in zip(final_boxes, final_scores, final_cls):
        x, y, w_box, h_box = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        xmin = x / IMAGE_SIZE
        ymin = y / IMAGE_SIZE
        xmax = (x + w_box) / IMAGE_SIZE
        ymax = (y + h_box) / IMAGE_SIZE
        # clamp
        xmin, ymin, xmax, ymax = max(0.0, xmin), max(0.0, ymin), min(1.0, xmax), min(1.0, ymax)
        class_name = COCO_CLASSES[int(cls)] if int(cls) < len(COCO_CLASSES) else str(int(cls))
        detections.append({
            "class_id": int(cls),
            # provide both keys for compatibility: 'class_name' is preferred
            "class_name": class_name,
            "class": class_name,
            "score": float(score),
            "location": [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
            "center": ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0),
        })
    return detections


class InferenceServicer(infer_pb2_grpc.InferenceServicer):
    def __init__(self, downscale_w: int = 320, downscale_h: int = 240, send_every_n: int = 1):
        self.downscale_w = downscale_w
        self.downscale_h = downscale_h
        self.send_every_n = max(1, int(send_every_n))

    def Stream(self, request_iterator, context):
        """Handle a bidirectional stream. Decode images and yield DetectionResponse."""
        frame_count = 0
        for req in request_iterator:
            frame_count += 1
            try:
                img_bytes = base64.b64decode(req.image_b64)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning("received invalid image, skipping")
                    continue

                # Optionally resize to target low-res size to reduce CPU for real detectors
                if req.width != self.downscale_w or req.height != self.downscale_h:
                    img = cv2.resize(img, (self.downscale_w, self.downscale_h))

                # By default respond every frame (send_every_n controls frequency)
                if (frame_count % self.send_every_n) != 0:
                    resp = infer_pb2.DetectionResponse(
                        source=req.source,
                        timestamp_ms=int(time.time() * 1000),
                        detections=[],
                    )
                    yield resp
                    continue

                # Run real inference if model available; otherwise fall back to a center box
                try:
                    logger.debug("getting session and running inference")
                    session, model_base = get_session(model_name="l", device="cuda")
                    detections = infer_frame(session, model_base, img, conf_threshold=0.3)
                    logger.info("frame {}: got {} detections", frame_count, len(detections))
                except Exception:
                    logger.exception("inference failed; using center-box fallback")
                    detections = [
                        {
                            "class": "fallback",
                            "score": 0.5,
                            # normalized coordinates (0..1) for fallback
                            "location": [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
                            "center": (0.5, 0.5),
                        }
                    ]

                # Convert detections into protobuf Detection messages (normalized coords)
                proto_dets = []
                h, w = img.shape[:2]
                for d in detections:
                    # location -> normalized bbox
                    loc = d.get("location")
                    tl = None
                    br = None
                    if isinstance(loc, (list, tuple)) and len(loc) >= 3:
                        tl = loc[0]
                        br = loc[2]
                    else:
                        # fallback center box
                        tl = (0.4, 0.4)
                        br = (0.6, 0.6)

                    def _to_float(x, default=0.0):
                        try:
                            return float(x)
                        except Exception:
                            return default

                    t_x = _to_float(tl[0], 0.4)
                    t_y = _to_float(tl[1], 0.4)
                    b_x = _to_float(br[0], 0.6)
                    b_y = _to_float(br[1], 0.6)

                    # If coordinates are already normalized use them; otherwise normalize using downscaled image size
                    if 0.0 <= t_x <= 1.0 and 0.0 <= t_y <= 1.0 and 0.0 <= b_x <= 1.0 and 0.0 <= b_y <= 1.0:
                        xmin_n, ymin_n, xmax_n, ymax_n = t_x, t_y, b_x, b_y
                    else:
                        xmin_n = t_x / float(w)
                        ymin_n = t_y / float(h)
                        xmax_n = b_x / float(w)
                        ymax_n = b_y / float(h)

                    # Determine class id and name deterministically
                    class_id_val = d.get("class_id")
                    class_name_val = d.get("class_name") if d.get("class_name") is not None else d.get("class")

                    resolved_id = None
                    if class_id_val is not None:
                        try:
                            resolved_id = int(class_id_val)
                        except Exception:
                            resolved_id = None

                    if resolved_id is None and isinstance(class_name_val, str) and class_name_val:
                        # try to map name to COCO index
                        try:
                            resolved_id = COCO_CLASSES.index(class_name_val)
                        except ValueError:
                            resolved_id = None

                    # choose what to send: prefer resolved numeric id when valid
                    send_id = int(resolved_id) if (isinstance(resolved_id, int) and 0 <= resolved_id < len(COCO_CLASSES)) else 0
                    send_name = class_name_val if (isinstance(class_name_val, str) and class_name_val) else (COCO_CLASSES[send_id] if 0 <= send_id < len(COCO_CLASSES) else "")

                    score = _to_float(d.get("score", 0.0), 0.0)

                    pd = infer_pb2.Detection(
                        class_id=send_id,
                        score=float(score),
                        xmin=float(xmin_n),
                        ymin=float(ymin_n),
                        xmax=float(xmax_n),
                        ymax=float(ymax_n),
                        class_name=str(send_name),
                    )
                    proto_dets.append(pd)

                # compute processing time (ms) and echo client seq
                processing_ms = int((time.time() * 1000) - req.timestamp_ms) if req.timestamp_ms else 0
                resp = infer_pb2.DetectionResponse(
                    source=req.source,
                    detections=proto_dets,
                    timestamp_ms=int(time.time() * 1000),
                    seq=getattr(req, 'seq', 0),
                    processing_ms=processing_ms,
                )
                yield resp
            except Exception:
                logger.exception("error handling request; continuing")
                continue


def serve(host: str, port: int, workers: int):
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=workers),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )
    infer_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(), server)
    bind = f"{host}:{port}"
    server.add_insecure_port(bind)
    server.start()
    logger.info("gRPC inference server started on {}", bind)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("shutting down server")
        server.stop(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    serve(args.host, args.port, args.workers)


if __name__ == "__main__":
    main()
