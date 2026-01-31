import numpy as np
import cv2


def demo_postprocess(outputs, img_size):
    grids, expanded_strides = [], []
    for stride in [8, 16, 32]:
        h, w = img_size[0] // stride, img_size[1] // stride
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full(grid.shape[:2] + (1,), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    outputs[..., 0:2] -= outputs[..., 2:4] / 2
    return outputs


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    out_boxes, out_scores, out_cls = [], [], []
    for i in range(scores.shape[1]):
        cls_scores = scores[:, i]
        mask = cls_scores > score_thr
        if mask.any():
            cls_boxes = boxes[mask]
            indices = cv2.dnn.NMSBoxes(
                cls_boxes.tolist(), cls_scores[mask].tolist(), score_thr, nms_thr
            )
            if len(indices) > 0:
                for idx in indices.flatten():
                    out_boxes.append(cls_boxes[idx])
                    out_scores.append(cls_scores[mask][idx])
                    out_cls.append(i)
    # Always return tuple, even if empty
    return (
        np.array(out_boxes)
        if len(out_boxes) > 0
        else np.empty((0, 4), dtype=np.float32),
        np.array(out_scores) if len(out_scores) > 0 else np.array([], dtype=np.float32),
        np.array(out_cls) if len(out_cls) > 0 else np.array([], dtype=np.float32),
    )


def build_detection_records(
    final_boxes, final_scores, final_cls, COCO_CLASSES, ratio_w, ratio_h
):
    results = []
    for box, score, cls in zip(final_boxes, final_scores, final_cls):
        x1, y1, w, h = (
            float(box[0]) * ratio_w,
            float(box[1]) * ratio_h,
            float(box[2]) * ratio_w,
            float(box[3]) * ratio_h,
        )
        top_left = (x1, y1)
        top_right = (x1 + w, y1)
        bottom_right = (x1 + w, y1 + h)
        bottom_left = (x1, y1 + h)
        center = (x1 + w / 2, y1 + h / 2)
        record = {
            "class": COCO_CLASSES[int(cls)],
            "score": round(float(score), 2),
            "location": [
                (float(round(top_left[0], 2)), float(round(top_left[1], 2))),
                (float(round(top_right[0], 2)), float(round(top_right[1], 2))),
                (float(round(bottom_right[0], 2)), float(round(bottom_right[1], 2))),
                (float(round(bottom_left[0], 2)), float(round(bottom_left[1], 2))),
            ],
            "center": (float(round(center[0], 2)), float(round(center[1], 2))),
        }
        results.append(record)
    return results
