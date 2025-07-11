# OCR Payment Slip Processor

This repository contains a Python script for extracting information from
payment slip images. It relies on OpenCV and Tesseract by default with an
optional EasyOCR mode.

## Usage

Run the script with one or more image paths:

```bash
python traningOCR.py IMAGE1.jpg IMAGE2.png
```

Use EasyOCR instead of Tesseract by passing `--easyocr`:

```bash
python traningOCR.py IMAGE1.jpg --easyocr
```

The script will print the parsed data and save a JSON file for each image.
