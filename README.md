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

