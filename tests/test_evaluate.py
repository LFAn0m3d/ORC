import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ocr.evaluate import compute_map


def test_map_perfect_detection():
    preds = [{"bbox": [0, 0, 10, 10], "score": 0.9}]
    gts = [{"bbox": [0, 0, 10, 10]}]
    assert math.isclose(compute_map(preds, gts), 1.0, rel_tol=1e-6)


def test_map_partial_detection():
    preds = [
        {"bbox": [0, 0, 10, 10], "score": 0.9},
        {"bbox": [18, 18, 27, 27], "score": 0.7},
    ]
    gts = [
        {"bbox": [0, 0, 10, 10]},
        {"bbox": [20, 20, 30, 30]},
    ]
    ap = compute_map(preds, gts)
    assert 0 < ap < 1

