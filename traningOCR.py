import cv2
import pytesseract
import numpy as np
import re
from PIL import Image
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse

class PaymentSlipOCR:
    def __init__(self):
        """
        Initialize OCR processor for payment slips
        Requires: pip install opencv-python pytesseract pillow numpy
        """
        self.extracted_data = {}
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def extract_text_tesseract(self, image_path: str) -> str:
        """
        Extract text using Tesseract OCR
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-/$#@() '
            
            # Extract text
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            
            return text.strip()
            
        except Exception as e:
            print(f"Error in Tesseract OCR: {e}")
            return ""
    
    def extract_text_easyocr(self, image_path: str) -> str:
        """
        Extract text using EasyOCR (alternative method)
        Requires: pip install easyocr
        """
        try:
            import easyocr
            
            # Initialize reader
            reader = easyocr.Reader(['en'])
            
            # Extract text
            results = reader.readtext(image_path)
            
            # Combine all detected text
            text = ' '.join([result[1] for result in results])
            
            return text.strip()
            
        except ImportError:
            print("EasyOCR not installed. Install with: pip install easyocr")
            return ""
        except Exception as e:
            print(f"Error in EasyOCR: {e}")
            return ""
    
    def parse_payment_data(self, text: str) -> Dict:
        """
        Parse payment-related data from extracted text
        """
        data = {
            'amounts': [],
            'dates': [],
            'account_numbers': [],
            'reference_numbers': [],
            'merchant_names': [],
            'transaction_types': [],
            'phone_numbers': [],
            'emails': []
        }
        
        # Amount patterns (various currency formats)
        amount_patterns = [
            r'[\$€£¥₹]?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*[\$€£¥₹]',
            r'(?:USD|EUR|GBP|INR|JPY)\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'(?:Amount|Total|Sum|Pay|Due)[:=\s]+[\$€£¥₹]?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            data['amounts'].extend(matches)
        
        # Date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2,4}'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            data['dates'].extend(matches)
        
        # Account number patterns
        account_patterns = [
            r'(?:Account|Acc|A/C)[:=\s]+(\d{8,20})',
            r'(?:Card|Credit|Debit)[:=\s]+(\d{4}\s*\d{4}\s*\d{4}\s*\d{4})',
            r'\b\d{10,20}\b'
        ]
        
        for pattern in account_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            data['account_numbers'].extend(matches)
        
        # Reference number patterns
        ref_patterns = [
            r'(?:Ref|Reference|Trans|Transaction)[:=\s]+([A-Z0-9]+)',
            r'(?:Receipt|Invoice)[:=\s]+([A-Z0-9]+)',
            r'\b[A-Z]{2,4}\d{6,12}\b'
        ]
        
        for pattern in ref_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            data['reference_numbers'].extend(matches)
        
        # Phone number patterns
        phone_patterns = [
            r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'(?:\+\d{1,3}[-.\s]?)?\d{10,15}'
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            data['phone_numbers'].extend(matches)
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        data['emails'] = re.findall(email_pattern, text)
        
        # Merchant/Company name patterns (common keywords)
        merchant_keywords = [
            r'(?:Pay to|Payee|Merchant|Company|Store)[:=\s]+([A-Za-z\s&.,]+)',
            r'(?:Bill from|From|Paid to)[:=\s]+([A-Za-z\s&.,]+)'
        ]
        
        for pattern in merchant_keywords:
            matches = re.findall(pattern, text, re.IGNORECASE)
            data['merchant_names'].extend(matches)
        
        # Transaction type patterns
        transaction_types = [
            'Payment', 'Transfer', 'Withdrawal', 'Deposit', 'Purchase', 
            'Refund', 'Credit', 'Debit', 'Bill Payment', 'Fee'
        ]
        
        for trans_type in transaction_types:
            if trans_type.lower() in text.lower():
                data['transaction_types'].append(trans_type)
        
        # Clean up duplicates
        for key in data:
            data[key] = list(set(data[key]))
        
        return data
    
    def process_slip(self, image_path: str, use_easyocr: bool = False) -> Dict:
        """
        Complete processing of payment slip
        """
        print(f"Processing payment slip: {image_path}")
        
        # Extract text using chosen OCR method
        if use_easyocr:
            raw_text = self.extract_text_easyocr(image_path)
        else:
            raw_text = self.extract_text_tesseract(image_path)
        
        if not raw_text:
            return {"error": "Could not extract text from image"}
        
        print(f"Extracted text:\n{raw_text}\n")
        
        # Parse payment data
        parsed_data = self.parse_payment_data(raw_text)
        
        # Combine results
        result = {
            'timestamp': datetime.now().isoformat(),
            'raw_text': raw_text,
            'parsed_data': parsed_data,
            'confidence_score': self.calculate_confidence_score(parsed_data)
        }
        
        return result
    
    def calculate_confidence_score(self, data: Dict) -> float:
        """
        Calculate confidence score based on extracted data quality
        """
        score = 0
        max_score = 8
        
        # Check if key fields are found
        if data['amounts']:
            score += 2
        if data['dates']:
            score += 2
        if data['account_numbers']:
            score += 1
        if data['reference_numbers']:
            score += 1
        if data['merchant_names']:
            score += 1
        if data['transaction_types']:
            score += 1
        
        return (score / max_score) * 100
    
    def save_results(self, results: Dict, output_file: str = "payment_data.json"):
        """
        Save extracted data to JSON file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def format_for_payment_processing(self, data: Dict) -> Dict:
        """
        Format data for payment processing systems
        """
        payment_data = {
            'transaction_id': data['parsed_data']['reference_numbers'][0] if data['parsed_data']['reference_numbers'] else None,
            'amount': data['parsed_data']['amounts'][0] if data['parsed_data']['amounts'] else None,
            'date': data['parsed_data']['dates'][0] if data['parsed_data']['dates'] else None,
            'account': data['parsed_data']['account_numbers'][0] if data['parsed_data']['account_numbers'] else None,
            'merchant': data['parsed_data']['merchant_names'][0] if data['parsed_data']['merchant_names'] else None,
            'transaction_type': data['parsed_data']['transaction_types'][0] if data['parsed_data']['transaction_types'] else None,
            'confidence': data['confidence_score'],
            'processed_at': data['timestamp']
        }
        
        return payment_data

# Example usage
def main():
    """Command line interface for processing payment slips."""
    parser = argparse.ArgumentParser(description="Process payment slip images")
    parser.add_argument(
        "image_paths",
        nargs="+",
        help="Paths to payment slip images to process"
    )
    parser.add_argument(
        "--easyocr",
        action="store_true",
        help="Use EasyOCR instead of Tesseract"
    )
    args = parser.parse_args()

    # Initialize OCR processor
    ocr_processor = PaymentSlipOCR()

    for image_path in args.image_paths:
        try:
            print(f"\n{'='*50}")
            print(f"Processing: {image_path}")
            print(f"{'='*50}")

            # Process the slip
            results = ocr_processor.process_slip(image_path, use_easyocr=args.easyocr)
            
            if "error" in results:
                print(f"Error: {results['error']}")
                continue
            
            # Display results
            print(f"Confidence Score: {results['confidence_score']:.1f}%")
            print(f"\nParsed Data:")
            for key, value in results['parsed_data'].items():
                if value:
                    print(f"  {key.title()}: {value}")
            
            # Format for payment processing
            payment_ready = ocr_processor.format_for_payment_processing(results)
            print(f"\nPayment Processing Format:")
            for key, value in payment_ready.items():
                print(f"  {key}: {value}")
            
            # Save results
            output_filename = f"extracted_data_{image_path.split('.')[0]}.json"
            ocr_processor.save_results(results, output_filename)
            
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    main()
