import cv2
import time
import json
import numpy as np
import onnxruntime as ort
import socket
from coco_classes import COCO_CLASSES
from utils import demo_postprocess, multiclass_nms, build_detection_records

def run_inference_stream(
    source,
    client_addr,
    client_port,
    model_name="l",
    confidence=0.75,
    device="cpu",
):
    """
    Perform inference on a camera/video stream and stream results to a UDP client.
    - source: camera index or stream url
    - client_addr: destination IP address (str)
    - client_port: destination port (int)
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

    print(f"[Worker] Started run_inference_stream with source: {source}, sending to {client_addr}:{client_port}")
    try:
        session = ort.InferenceSession(MODEL, providers=[PROVIDER])
    except Exception as e:
        print(f"[Worker] Failed to create ONNX Runtime session with provider {PROVIDER}: {e}")
        print("[Worker] Falling back to CPUExecutionProvider.")
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
        print(f"[Worker] Unable to open video source {source}")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("[Worker] Video source opened. Entering main loop.")
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Worker] Frame read failed. Exiting loop.")
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
            ratio_w, ratio_h = frame.shape[1] / input_size[1], frame.shape[0] / input_size[0]
            detections = build_detection_records(
                final_boxes, final_scores, final_cls, COCO_CLASSES, ratio_w, ratio_h
            )
            # Limit outgoing UDP message to 8192 bytes (prune detections if needed)
            payload = {"detections": detections}
            while True:
                message = json.dumps(payload).encode('utf-8')
                if len(message) <= 8192 or not payload["detections"]:
                    break
                # Drop the detection with lowest score (if multiple)
                payload["detections"] = sorted(payload["detections"], key=lambda x: x['score'], reverse=True)[:-1]
            if len(message) > 8192:
                print(f"[Worker] Warning: Message truncated for frame {frame_count}")
                message = message[:8192]
            # Send the results as a UDP datagram
            try:
                sock.sendto(message, (client_addr, client_port))
                print(f"[Worker] Sent detection result for frame {frame_count} to {client_addr}:{client_port}")
            except Exception as send_err:
                print(f"[Worker] Error sending to client: {send_err}")
            time.sleep(0.04)  # Limit to ~25 FPS to prevent socket buffer overflow
    finally:
        if cap:
            cap.release()
        sock.close()
        print("[Worker] Capture and socket closed.")

