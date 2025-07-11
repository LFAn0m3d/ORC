"""Training loop for OCR model."""
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as T

from .dataset import OCRDataset
from .model import OCRModel, recognition_targets_from_annotations


def collate(batch):
    images, targets = zip(*batch)
    images = [T.ToTensor()(img) for img in images]
    # targets should be dicts with boxes and text labels
    return images, targets


def train(data_root: str, annotation_file: str, epochs: int = 10, lr: float = 1e-4):
    dataset = OCRDataset(data_root, annotation_file)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)

    model = OCRModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for images, targets in loader:
            images = [img.to(device) for img in images]
            optimizer.zero_grad()

            # prepare detection targets
            det_targets = []
            for t in targets:
                boxes = torch.tensor([b["bbox"] for b in t], dtype=torch.float32, device=device)
                labels = torch.ones((len(boxes),), dtype=torch.int64, device=device)
                det_targets.append({"boxes": boxes, "labels": labels})

            det_loss_dict = model.detector(images, det_targets)
            det_loss = sum(det_loss_dict.values())

            # recognition targets and loss
            crops, rec_labels = recognition_targets_from_annotations(images, targets)
            if crops is not None:
                crops = crops.to(device)
                rec_labels = rec_labels.to(device)
                rec_logits = model.recognizer(crops)
                rec_loss = torch.nn.functional.cross_entropy(rec_logits, rec_labels)
            else:
                rec_loss = torch.tensor(0.0, device=device)

            loss = det_loss + rec_loss
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
