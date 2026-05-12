# Python NMS

Non-Maximum Suppression (NMS) implemented in Python with NumPy.

## Usage

```python
from nms import nms

boxes_out, scores_out, labels_out = nms(boxes, scores, labels, score_threshold=0.5, iou_threshold=0.5)
```

## Parameters

| 参数 | 类型 | 说明 |
|------|------|------|
| `boxes` | array-like (N, 4) | 边界框，格式 `[y0, x0, y1, x1]` 或 `[x0, y0, x1, y1]` |
| `scores` | array-like (N,) | 每个框的置信度分数 |
| `labels` | array-like (N,), 可选 | 每个框的类别标签，不传则返回 `None` |
| `score_threshold` | float, 默认 0.5 | 低于此分数的框直接过滤 |
| `iou_threshold` | float, 默认 0.5 | IoU 超过此阈值的重叠框将被抑制 |

## Returns

| 返回值 | 说明 |
|--------|------|
| `boxes_out` | 经 NMS 筛选后的边界框 (numpy array) |
| `scores_out` | 对应的置信度分数 (numpy array) |
| `labels_out` | 对应的类别标签 (numpy array)，未传入 labels 时为 `None` |

## Example

```python
import numpy as np
from nms import nms

boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [3, 3, 5, 5]])
scores = np.array([0.6, 0.7, 0.6])
labels = np.array([0, 1, 2])

boxes_out, scores_out, labels_out = nms(boxes, scores, labels, score_threshold=0.5, iou_threshold=0.5)
print(boxes_out)   # [[ 1.  1. 10. 10.], [ 3.  3.  5.  5.]]
print(scores_out)  # [0.7 0.6]
print(labels_out)  # [1 2]
```
