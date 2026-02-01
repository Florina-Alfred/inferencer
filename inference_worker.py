import cv2
import json
import numpy as np
import onnxruntime as ort
import socket
from loguru import logger
from coco_classes import COCO_CLASSES
from utils import demo_postprocess, multiclass_nms, build_detection_records


def run_inference_stream(
    source,
    out_queue,
    model_name="l",
    confidence=0.75,
    device="cuda",
):
    """
    Perform inference on a camera/video stream and stream results to a UDP client.
    - source: camera index or stream url
    - out_queue: multiprocessing.Queue used to send (frame_count, message) tuples to the server
    - model_name: yolox short model name (l, m, s, tiny, nano)
    - confidence: confidence threshold for detection
    - device: 'cpu' or 'cuda'
    """
    PROVIDER_MAP = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
    }
    MODEL_SHORT_TO_NAME = {
        "l": "yolox_l",
        "m": "yolox_m",
        "s": "yolox_s",
        "tiny": "yolox_tiny",
        "nano": "yolox_nano",
    }
    PROVIDER = PROVIDER_MAP.get(device, "CPUExecutionProvider")
    model_base = MODEL_SHORT_TO_NAME.get(model_name, "yolox_l")
    MODEL = f"model/{model_base}.onnx"
    CONF_THRESHOLD = confidence
    IMAGE_SIZE = 416 if model_base in ["yolox_tiny", "yolox_nano"] else 640
    input_size = (IMAGE_SIZE, IMAGE_SIZE)

    logger.info("[Worker] Started run_inference_stream with source: {source}", source=source)
    try:
        session = ort.InferenceSession(MODEL, providers=[PROVIDER])
    except Exception as e:
        logger.exception("[Worker] Failed to create ONNX Runtime session with provider {provider}: {err}", provider=PROVIDER, err=e)
        logger.info("[Worker] Falling back to CPUExecutionProvider.")
        session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])

    # Video source handling
    cap = None
    try:
        src = int(source)
        cap = cv2.VideoCapture(src)
    except Exception:
        src = str(source)
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap or not cap.isOpened():
        logger.error("[Worker] Unable to open video source {source}", source=source)
        return

    logger.info("[Worker] Video source opened. Entering main loop.")
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("[Worker] Frame read failed. Exiting loop.")
                break
            frame_count += 1
            img = cv2.resize(frame, input_size).astype(np.float32)
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, 0)
            outputs = session.run(None, {session.get_inputs()[0].name: img})[0]
            predictions = demo_postprocess(outputs, input_size)[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]
            final_boxes, final_scores, final_cls = multiclass_nms(
                boxes, scores, 0.45, CONF_THRESHOLD
            )
            ratio_w, ratio_h = (
                frame.shape[1] / input_size[1],
                frame.shape[0] / input_size[0],
            )
            detections = build_detection_records(
                final_boxes, final_scores, final_cls, COCO_CLASSES, ratio_w, ratio_h
            )
            # Prepare message and hand it to the server via out_queue for broadcasting to clients
            MAX_UDP_SIZE = 65507
            payload = {"detections": detections}
            while True:
                message = json.dumps(payload).encode("utf-8")
                if len(message) <= MAX_UDP_SIZE or not payload["detections"]:
                    break
                payload["detections"] = sorted(payload["detections"], key=lambda x: x["score"], reverse=True)[:-1]
            if len(message) > MAX_UDP_SIZE:
                logger.warning("[Worker] Warning: Message truncated for frame {frame}", frame=frame_count)
                message = message[:MAX_UDP_SIZE]
            try:
                out_queue.put((frame_count, message))
                logger.debug("[Worker] Enqueued detection result for frame {frame}", frame=frame_count)
            except Exception as q_err:
                logger.exception("[Worker] Error enqueuing message: {err}", err=q_err)
            # time.sleep(0.04)  # Limit to ~25 FPS to prevent socket buffer overflow
    finally:
        if cap:
            cap.release()
        logger.info("[Worker] Capture closed. Signaling shutdown to server.")
        try:
            out_queue.put(None)
        except Exception:
            pass
