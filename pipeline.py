import cv2
import pytesseract
import re
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    """Load image, convert to grayscale, and apply thresholding."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text(image):
    """Perform OCR using Tesseract."""
    pil_img = Image.fromarray(image)
    config = '--oem 3 --psm 6 -l tha+eng'
    return pytesseract.image_to_string(pil_img, config=config)

def parse_fields(text):
    """Extract structured fields using regex and postprocessing."""
    # Reference ID
    ref_match = re.search(r'\b\d{15,18}\b', text)
    # Account patterns
    accounts = re.findall(r'X{2,}-X-[X0-9A-Z]{4,}-\d', text)
    # Amount with fallback for บาท
    amount_match = re.search(r'(\d{1,3}(,\d{3})*\.\d{2})', text)
    # Date (Thai format recognition workaround)
    date_match = re.search(r'(0[1-9]|[12][0-9]|3[01])\s+[^a-zA-Z0-9\s]+\s+\d{4}', text)
    # Time
    time_match = re.search(r'(\d{2}:\d{2})', text)

    return {
        "reference_id": ref_match.group() if ref_match else None,
        "from_account": accounts[0] if len(accounts) > 0 else None,
        "to_account": accounts[1] if len(accounts) > 1 else None,
        "amount": amount_match.group(1) if amount_match else None,
        "date": date_match.group() if date_match else None,
        "time": time_match.group() if time_match else None
    }

# Run the full OCR pipeline on the uploaded slip
image_path = "/mnt/data/3b8bd289-74ec-4fa9-b3af-6eac88acbbe3.png"
processed_image = preprocess_image(image_path)
ocr_text = extract_text(processed_image)
fields = parse_fields(ocr_text)

fields
