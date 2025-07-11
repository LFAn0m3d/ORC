"""OCR processing pipeline with pre and post processing."""
from typing import Dict, List
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
import pytesseract
import re

from .model import OCRModel


class OCRPipeline:
    def __init__(self, model_path: str = "models/ocr_model.pt"):
        self.model = OCRModel()
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    @staticmethod
    def preprocess(image: Image.Image) -> Image.Image:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.bilateralFilter(img, 9, 75, 75)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    @staticmethod
    def postprocess(results: List[Dict]) -> List[Dict]:
        # placeholder for cleaning recognition results
        return results

    @staticmethod
    def _detect_language(text: str) -> str:
        if re.search(r"[\u0E00-\u0E7F]", text):
            return "th"
        return "en"

    def recognize_crop(self, crop: Image.Image) -> str:
        return pytesseract.image_to_string(crop, lang="tha+eng").strip()

    def __call__(self, image: Image.Image) -> List[Dict]:
        image = self.preprocess(image)
        tensor = [T.ToTensor()(image)]
        with torch.no_grad():
            det = self.model(tensor)[0]

        results = []
        boxes = det.get("boxes", [])
        scores = det.get("scores", [])
        for box, score in zip(boxes, scores):
            if float(score) < 0.5:
                continue
            box_list = box.tolist() if hasattr(box, "tolist") else list(box)
            crop = image.crop((box_list[0], box_list[1], box_list[2], box_list[3]))
            text = self.recognize_crop(crop)
            results.append({
                "text": text,
                "language": self._detect_language(text),
                "bbox": box_list,
                "conf": float(score),
            })

        return self.postprocess(results)
