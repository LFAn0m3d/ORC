# Payment Slip OCR

This project provides a Python script for extracting data from payment slips, receipts and invoices. Images are preprocessed with OpenCV and then passed through Tesseract or EasyOCR to recognize text. The script parses the results to identify amounts, dates, reference numbers and other useful fields and stores them in JSON format.

## Required packages

Install the dependencies with pip:

```bash
pip install opencv-python pytesseract pillow numpy easyocr
```

## Usage

Run the training/processing script:

```bash
python traningOCR.py
```

The script processes the paths listed in the `image_paths` array within `traningOCR.py`. Modify that list to analyze your own files. Parsed data will be saved to JSON files such as `extracted_data_payment_slip.json`.

## Deep Learning OCR Pipeline

The `ocr` package contains an experimental pipeline for training and evaluating
a deep learning based OCR system. It is designed to handle both Thai and English
text and can be extended to additional languages.

### Dataset preparation

Annotations are expected in a JSON file with the following structure:

```json
[
  {
    "image": "path/to/image.jpg",
    "boxes": [
      {"bbox": [x1, y1, x2, y2], "text": "Example", "language": "en"}
    ]
  }
]
```

Images should cover a wide variety of scenes such as documents, signs and
packaging captured in different lighting conditions, angles and resolutions.

### Training

Run the training script with:

```bash
python -m ocr.train --data-root /path/to/images --annotations annotations.json
```

The model weights will be stored in `models/ocr_model.pt`.

### Evaluation

Utility functions in `ocr.evaluate` provide character error rate (CER), word
error rate (WER) and a placeholder for mAP. Evaluation should be performed on a
held‑out dataset to validate real‑world performance.

### API

A simple `FastAPI` server in `ocr/api.py` exposes the OCR pipeline. Launch it
with:

```bash
python -m ocr.api
```

It returns a JSON array of detected text boxes including the recognized text,
language, bounding box coordinates and confidence score.

