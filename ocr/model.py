from typing import List
import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_align
import torchvision.transforms.functional as TF

class DetectionModel(nn.Module):
    """Simple wrapper around Faster R-CNN for text detection."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = nn.Linear(in_features, num_classes)

    def forward(self, images: List[torch.Tensor], targets=None):
        return self.model(images, targets)

VOCAB = "#0123456789abcdefghijklmnopqrstuvwxyz"

# Map characters to indices for recognition training
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}


class RecognitionModel(nn.Module):
    """Simple classifier for single-character recognition."""

    def __init__(self, num_classes: int = len(VOCAB)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv(images)
        features = features.view(features.size(0), -1)
        features = torch.relu(self.fc1(features))
        return self.fc2(features)

class OCRModel(nn.Module):
    """Combined detection and recognition model."""
    def __init__(self):
        super().__init__()
        self.detector = DetectionModel()
        self.recognizer = RecognitionModel()

    def forward(self, images: List[torch.Tensor]):
        detections = self.detector(images)
        # recognition step would require cropping detections and processing
        return detections


def recognition_targets_from_annotations(images: List[torch.Tensor], annotations: List[List[dict]], size: int = 32):
    """Generate cropped regions and target indices from annotations."""
    crops = []
    labels = []
    for img, ann in zip(images, annotations):
        for box in ann:
            x1, y1, x2, y2 = map(int, box["bbox"])
            if x2 <= x1 or y2 <= y1:
                continue
            crop = TF.resized_crop(img, y1, x1, y2 - y1, x2 - x1, (size, size))
            crops.append(crop)
            text = box.get("text", "").lower()
            char = text[0] if text else "#"
            labels.append(CHAR_TO_IDX.get(char, 0))
    if not crops:
        return None, None
    return torch.stack(crops), torch.tensor(labels, dtype=torch.long)
