"""Evaluation utilities for OCR models."""
from typing import List, Dict
from difflib import SequenceMatcher
import torch


def char_error_rate(pred: str, target: str) -> float:
    matcher = SequenceMatcher(None, pred, target)
    distance = sum(block.size for block in matcher.get_matching_blocks())
    return 1 - distance / max(len(pred), len(target), 1)


def word_error_rate(pred: str, target: str) -> float:
    return char_error_rate(pred.split(), target.split())


def compute_map(detections: List[Dict], targets: List[Dict]) -> float:
    """Placeholder for mAP computation. Returns 0-1."""
    # Actual implementation would require IoU calculations across thresholds
    return 0.0


__all__ = ["char_error_rate", "word_error_rate", "compute_map"]
