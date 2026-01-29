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
    return out_boxes, out_scores, out_cls
