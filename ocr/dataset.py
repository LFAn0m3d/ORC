import json
from pathlib import Path
from typing import List, Tuple, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    """Dataset for OCR with bounding box annotations and text labels."""

    def __init__(self, root: str, annotation_file: str):
        root = Path(root)
        self.root = root
        self.annotations: List[Dict] = json.loads(Path(annotation_file).read_text())

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, List[Dict]]:
        record = self.annotations[idx]
        img_path = self.root / record["image"]
        image = Image.open(img_path).convert("RGB")
        bboxes = record.get("boxes", [])
        for b in bboxes:
            # ensure floats
            b["bbox"] = [float(x) for x in b["bbox"]]
        return image, bboxes
