"""Evaluation utilities for OCR models."""
from typing import List, Dict
from difflib import SequenceMatcher



def char_error_rate(pred: str, target: str) -> float:
    matcher = SequenceMatcher(None, pred, target)
    distance = sum(block.size for block in matcher.get_matching_blocks())
    return 1 - distance / max(len(pred), len(target), 1)


def word_error_rate(pred: str, target: str) -> float:
    return char_error_rate(pred.split(), target.split())


def compute_map(detections: List[Dict], targets: List[Dict]) -> float:
    """Compute mean average precision (mAP) over IoU thresholds.

    Parameters
    ----------
    detections : List[Dict]
        Predicted boxes with at least ``bbox`` and ``score``/``conf`` keys.
    targets : List[Dict]
        Ground truth boxes with ``bbox`` key.

    Returns
    -------
    float
        Mean average precision in the range ``0-1``.
    """

    def iou(box_a: List[float], box_b: List[float]) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(inter_x2 - inter_x1, 0)
        inter_h = max(inter_y2 - inter_y1, 0)
        inter = inter_w * inter_h
        area_a = max(xa2 - xa1, 0) * max(ya2 - ya1, 0)
        area_b = max(xb2 - xb1, 0) * max(yb2 - yb1, 0)
        union = area_a + area_b - inter
        if union == 0:
            return 0.0
        return inter / union

    def average_precision(preds: List[Dict], gts: List[Dict], thr: float) -> float:
        preds = sorted(preds, key=lambda x: x.get("score", x.get("conf", 0)), reverse=True)
        matched = set()
        tp, fp = [], []

        for p in preds:
            best_iou = 0.0
            best_gt = None
            for i, gt in enumerate(gts):
                if i in matched:
                    continue
                iou_val = iou(p["bbox"], gt["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt = i
            if best_iou >= thr and best_gt is not None:
                matched.add(best_gt)
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        if len(gts) == 0:
            return 0.0

        tp_cum, fp_cum = [], []
        total_tp, total_fp = 0, 0
        for t, f in zip(tp, fp):
            total_tp += t
            total_fp += f
            tp_cum.append(total_tp)
            fp_cum.append(total_fp)

        recalls = [t / len(gts) for t in tp_cum]
        precisions = [tp_cum[i] / (tp_cum[i] + fp_cum[i]) for i in range(len(tp_cum))]

        mrec = [0.0] + recalls + [1.0]
        mpre = [0.0] + precisions + [0.0]
        for i in range(len(mpre) - 1, 0, -1):
            if mpre[i - 1] < mpre[i]:
                mpre[i - 1] = mpre[i]
        ap = 0.0
        for i in range(len(mrec) - 1):
            if mrec[i + 1] != mrec[i]:
                ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
        return ap

    iou_thresholds = [round(x * 0.05 + 0.5, 2) for x in range(10)]
    ap_values = [average_precision(detections, targets, thr) for thr in iou_thresholds]
    return sum(ap_values) / len(ap_values) if ap_values else 0.0


__all__ = ["char_error_rate", "word_error_rate", "compute_map"]
