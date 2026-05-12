#!/usr/bin/env python
# encoding: utf-8
import numpy as np


def nms(boxes, scores, labels=None, score_threshold=0.5, iou_threshold=0.5):
    """
    Non-Maximum Suppression.

    Args:
        boxes: array-like (N, 4), format [y0, x0, y1, x1] or [x0, y0, x1, y1]
        scores: array-like (N,)
        labels: array-like (N,), optional
        score_threshold: discard boxes with score <= this value
        iou_threshold: suppress boxes with IoU > this value

    Returns:
        boxes_out, scores_out, labels_out as numpy arrays (labels_out is None if labels not given)
    """
    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    has_labels = labels is not None
    labels = np.array(labels) if has_labels else np.empty(len(scores), dtype=int)

    # Filter by score threshold
    keep = scores > score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if len(scores) == 0:
        empty = np.array([])
        return boxes, empty, labels if has_labels else None

    # Sort by score descending
    order = np.argsort(scores)[::-1]
    boxes, scores, labels = boxes[order], scores[order], labels[order]

    suppressed = np.zeros(len(scores), dtype=bool)
    selected = []

    for i in range(len(scores)):
        if suppressed[i]:
            continue
        selected.append(i)
        if i + 1 < len(scores):
            iou = _iou_vectorized(boxes[i], boxes[i + 1:])
            suppressed[i + 1:][iou > iou_threshold] = True

    idx = np.array(selected)
    return boxes[idx], scores[idx], labels[idx] if has_labels else None


def _iou_vectorized(box, boxes):
    """IoU of one box against an array of boxes."""
    inter_area = (
        np.maximum(0, np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])) *
        np.maximum(0, np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]))
    )
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = area_box + area_boxes - inter_area
    return np.where(union_area > 0, inter_area / union_area, 0.0)


if __name__ == '__main__':
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [3, 3, 5, 5]])
    scores = np.array([0.6, 0.7, 0.6])
    labels = np.array([0, 1, 2])
    boxes_out, scores_out, labels_out = nms(boxes, scores, labels, score_threshold=0.5, iou_threshold=0.5)
    print(boxes_out)
    print(scores_out)
    print(labels_out)
