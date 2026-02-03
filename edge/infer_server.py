#!/usr/bin/env python3
"""gRPC inference server: receives base64 JPEGs, runs ONNX inference, returns detections.

This server expects generated stubs in `edge/proto` (run `python generate.py`).
"""
from __future__ import annotations

import argparse
import base64
import time
from concurrent import futures
import os

import grpc
import cv2
import numpy as np

try:
    from edge.proto import detection_pb2 as pb
    from edge.proto import detection_pb2_grpc as pb_grpc
except Exception as e:  # pragma: no cover - helpful error
    raise RuntimeError("gRPC stubs missing: run `python generate.py` in edge/") from e

from coco_classes import COCO_CLASSES
from utils import demo_postprocess, multiclass_nms, build_detection_records
import onnxruntime as ort


MODEL_SHORT_TO_NAME = {
    "l": "yolox_l",
    "m": "yolox_m",
    "s": "yolox_s",
    "tiny": "yolox_tiny",
    "nano": "yolox_nano",
}


def make_session(model_name: str, device: str):
    model_base = MODEL_SHORT_TO_NAME.get(model_name, "yolox_l")
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", f"{model_base}.onnx")
    provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    try:
        sess = ort.InferenceSession(model_path, providers=[provider])
    except Exception:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess, model_base


def infer_frame(session, model_base: str, frame: np.ndarray, conf_threshold: float = 0.5):
    # follow main.py preprocessing
    IMAGE_SIZE = 416 if model_base in ["yolox_tiny", "yolox_nano"] else 640
    input_size = (IMAGE_SIZE, IMAGE_SIZE)
    img = cv2.resize(frame, input_size).astype(np.float32)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    outputs = session.run(None, {session.get_inputs()[0].name: img})[0]
    preds = demo_postprocess(outputs, input_size)[0]
    if preds.size == 0:
        return []
    boxes = preds[:, :4]
    scores = preds[:, 4:5] * preds[:, 5:]
    final_boxes, final_scores, final_cls = multiclass_nms(boxes, scores, 0.45, conf_threshold)
    # build detection records in pixel coordinates similar to main.py
    h, w = frame.shape[:2]
    ratio_w, ratio_h = w / input_size[1], h / input_size[0]
    records = build_detection_records(final_boxes, final_scores, final_cls, COCO_CLASSES, ratio_w, ratio_h)
    return records


class InferenceServicer(pb_grpc.InferenceServicer):
    def __init__(self, model: str = "l", device: str = "cpu", conf: float = 0.5):
        self.session, self.model_base = make_session(model, device)
        self.conf = conf

    def Stream(self, request_iterator, context):
        for req in request_iterator:
            t0 = time.time()
            try:
                img_b64 = req.image_b64
                img_bytes = base64.b64decode(img_b64)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    # send empty response
                    resp = pb.DetectionResponse(source=req.source, timestamp_ms=int(time.time()*1000), seq=req.seq)
                    yield resp
                    continue
                records = infer_frame(self.session, self.model_base, img, self.conf)
                # convert records to proto
                proto_dets = []
                for r in records:
                    # r has 'location' top-left, top-right, bottom_right, bottom_left
                    tl = r['location'][0]
                    br = r['location'][2]
                    xmin, ymin = float(tl[0]), float(tl[1])
                    xmax, ymax = float(br[0]), float(br[1])
                    det = pb.Detection(class_id=COCO_CLASSES.index(r['class']) if r['class'] in COCO_CLASSES else 0,
                                       score=float(r.get('score', 0.0)), xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                                       class_name=str(r.get('class', '')))
                    proto_dets.append(det)
                processing_ms = int((time.time() - t0) * 1000)
                resp = pb.DetectionResponse(source=req.source, found=proto_dets, timestamp_ms=int(time.time()*1000), seq=req.seq, processing_ms=processing_ms)
                yield resp
            except Exception:
                # send empty response on error
                resp = pb.DetectionResponse(source=getattr(req, 'source', ''), timestamp_ms=int(time.time()*1000), seq=getattr(req, 'seq', 0))
                yield resp


def serve(host: str, port: int, model: str, device: str, conf: float, workers: int = 4):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    pb_grpc.add_InferenceServicer_to_server(InferenceServicer(model=model, device=device, conf=conf), server)
    bind = f"{host}:{port}"
    server.add_insecure_port(bind)
    server.start()
    print("gRPC server started on", bind)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', default=50051, type=int)
    p.add_argument('--model', default='l')
    p.add_argument('--device', default='cpu')
    p.add_argument('--conf', default=0.5, type=float)
    args = p.parse_args()
    serve(args.host, args.port, args.model, args.device, args.conf)


if __name__ == '__main__':
    main()
