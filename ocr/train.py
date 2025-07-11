"""Training loop for OCR model."""
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as T

from .dataset import OCRDataset
from .model import OCRModel


def collate(batch):
    images, targets = zip(*batch)
    images = [T.ToTensor()(img) for img in images]
    # targets should be dicts with boxes and text labels
    return images, targets


def train(data_root: str, annotation_file: str, epochs: int = 10, lr: float = 1e-4):
    dataset = OCRDataset(data_root, annotation_file)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)

    model = OCRModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for images, targets in loader:
            optimizer.zero_grad()
            outputs = model(images)
            # compute detection and recognition losses (placeholder)
            loss = sum(o.get('loss', 0) for o in outputs if isinstance(o, dict))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/ocr_model.pt")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train OCR model")
    p.add_argument("--data-root", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()

    train(args.data_root, args.annotations, epochs=args.epochs)
