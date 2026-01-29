import cv2
import argparse
import numpy as np
import onnxruntime as ort

from coco_classes import COCO_CLASSES
from utils import demo_postprocess, multiclass_nms

# Configuration
# IMAGE_SIZE = 416
IMAGE_SIZE = 640

# Provider mapping
PROVIDER_MAP = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


parser = argparse.ArgumentParser(description="YOLOX Runtime")
parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="cpu",
    choices=["cpu", "cuda"],
    help="Select compute device: cpu or cuda (default: cpu)",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="l",
    help="YOLOX model variant: l, m, s, tiny, or nano (default: l)",
)
args = parser.parse_args()

PROVIDER = PROVIDER_MAP.get(args.device, "CPUExecutionProvider")
MODEL_SHORT_TO_NAME = {
    "l": "yolox_l",
    "m": "yolox_m",
    "s": "yolox_s",
    "tiny": "yolox_tiny",
    "nano": "yolox_nano",
}
model_base = MODEL_SHORT_TO_NAME.get(args.model, "yolox_l")
MODEL = f"model/{model_base}.onnx"
CONF_THRESHOLD = 0.8

input_size = (IMAGE_SIZE, IMAGE_SIZE)

try:
    session = ort.InferenceSession(MODEL, providers=[PROVIDER])
except Exception as e:
    print(f"Failed to create ONNX Runtime session with provider {PROVIDER}: {e}")
    print("Falling back to CPUExecutionProvider.")
    session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Preprocess
    img = cv2.resize(frame, input_size).astype(np.float32)
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, 0)

    # 2. Inference
    outputs = session.run(None, {session.get_inputs()[0].name: img})[0]

    # 3. Post-process (Decoding + NMS)
    predictions = demo_postprocess(outputs, input_size)[0]
    boxes = predictions[:, :4]
    # Multiply objectness by class probabilities
    scores = predictions[:, 4:5] * predictions[:, 5:]

    # boxes already in [x0, y0, w, h] format for OpenCV NMS
    final_boxes, final_scores, final_cls = multiclass_nms(
        boxes, scores, 0.45, CONF_THRESHOLD
    )

    # 4. Visualization
    ratio_w, ratio_h = frame.shape[1] / input_size[1], frame.shape[0] / input_size[0]
    for box, score, cls in zip(final_boxes, final_scores, final_cls):
        x1, y1, w, h = (
            box[0] * ratio_w,
            box[1] * ratio_h,
            box[2] * ratio_w,
            box[3] * ratio_h,
        )
        cv2.rectangle(
            frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2
        )
        label = f"{COCO_CLASSES[cls]}: {score:.2f}"
        cv2.putText(
            frame,
            label,
            (int(x1), int(y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("YOLOX Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
