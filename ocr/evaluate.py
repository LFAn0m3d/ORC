"""Evaluation utilities for OCR models."""
from typing import List, Dict
from difflib import SequenceMatcher

def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute intersection over union for two boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(inter_x2 - inter_x1, 0)
    inter_h = max(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def char_error_rate(pred: str, target: str) -> float:
    matcher = SequenceMatcher(None, pred, target)
    distance = sum(block.size for block in matcher.get_matching_blocks())
    return 1 - distance / max(len(pred), len(target), 1)


def word_error_rate(pred: str, target: str) -> float:
    return char_error_rate(pred.split(), target.split())


def compute_map(detections: List[Dict], targets: List[Dict], iou_threshold: float = 0.5) -> float:
    """Compute a simplified mean average precision for detection results."""
    if len(detections) != len(targets):
        raise ValueError("detections and targets must have the same length")

    total_gt = 0
    correct = 0

    for det, tgt in zip(detections, targets):
        gt_boxes = [t["bbox"] for t in tgt]
        total_gt += len(gt_boxes)

        matched: List[int] = []
        for box in det.get("boxes", []):
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if idx in matched:
                    continue
                iou = _iou(box if isinstance(box, list) else box.tolist(), gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                correct += 1
                matched.append(best_idx)

    if total_gt == 0:
        return 0.0
    return correct / total_gt


__all__ = ["char_error_rate", "word_error_rate", "compute_map"]
