import cv2
import numpy as np
import onnxruntime as ort

# COCO Class names for visualization
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

def demo_postprocess(outputs, img_size, p6=False):
    """Decodes raw YOLOX outputs into bounding boxes and scores."""
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    # Generate grid coordinates for each stride level
    for stride in strides:
        hsize, wsize = img_size[0] // stride, img_size[1] // stride
        yv, xv = np.meshgrid(np.arange(hsize), np.arange(wsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, axis=1)
    expanded_strides = np.concatenate(expanded_strides, axis=1)

    # Decode boxes: center_x, center_y, width, height
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Applies Non-Max Suppression to filter overlapping boxes."""
    final_boxes, final_scores, final_cls = [], [], []
    for i in range(scores.shape[1]): # Iterate over classes
        cls_scores = scores[:, i]
        mask = cls_scores > score_thr
        if mask.any():
            cls_boxes = boxes[mask]
            cls_scores = cls_scores[mask]
            # Use OpenCV's efficient NMS implementation
            indices = cv2.dnn.NMSBoxes(cls_boxes.tolist(), cls_scores.tolist(), score_thr, nms_thr)
            if len(indices) > 0:
                for idx in indices.flatten():
                    final_boxes.append(cls_boxes[idx])
                    final_scores.append(cls_scores[idx])
                    final_cls.append(i)
    return final_boxes, final_scores, final_cls

# Configuration
input_size = (416, 416)
input_size = (640, 640)
# session = ort.InferenceSession("yolox_tiny.onnx", providers=['CPUExecutionProvider'])
# session = ort.InferenceSession("yolox_nano.onnx", providers=['CPUExecutionProvider'])
# session = ort.InferenceSession("yolox_s.onnx", providers=['CPUExecutionProvider'])
# session = ort.InferenceSession("yolox_m.onnx", providers=['CPUExecutionProvider'])
session = ort.InferenceSession("yolox_l.onnx", providers=['CPUExecutionProvider'])
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Preprocess
    img = cv2.resize(frame, input_size).astype(np.float32)
    img = img.transpose(2, 0, 1) # HWC to CHW
    img = np.expand_dims(img, 0)

    # 2. Inference
    outputs = session.run(None, {session.get_inputs()[0].name: img})[0]
    
    # 3. Post-process (Decoding + NMS)
    predictions = demo_postprocess(outputs, input_size)[0]
    boxes = predictions[:, :4]
    # Multiply objectness by class probabilities
    scores = predictions[:, 4:5] * predictions[:, 5:]
    
    # Convert [cx, cy, w, h] to [x1, y1, w, h] for OpenCV NMS
    boxes[:, 0:2] -= boxes[:, 2:4] / 2
    
    final_boxes, final_scores, final_cls = multiclass_nms(boxes, scores, 0.45, 0.3)

    # 4. Visualization
    ratio_w, ratio_h = frame.shape[1]/input_size[1], frame.shape[0]/input_size[0]
    for box, score, cls in zip(final_boxes, final_scores, final_cls):
        x1, y1, w, h = (box[0] * ratio_w, box[1] * ratio_h, box[2] * ratio_w, box[3] * ratio_h)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (0, 255, 0), 2)
        label = f"{COCO_CLASSES[cls]}: {score:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOX Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

