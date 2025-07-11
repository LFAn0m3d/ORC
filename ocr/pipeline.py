"""OCR processing pipeline with pre and post processing."""
from typing import Dict, List
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T

from .model import OCRModel


class OCRPipeline:
    def __init__(self, model_path: str = "models/ocr_model.pt"):
        self.model = OCRModel()
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path))
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

    def __call__(self, image: Image.Image) -> List[Dict]:
        image = self.preprocess(image)
        tensor = [T.ToTensor()(image)]
        with torch.no_grad():
            detections = self.model(tensor)
        return self.postprocess(detections)
