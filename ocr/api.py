"""Simple FastAPI server exposing OCR pipeline."""
from io import BytesIO
from typing import List

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
import uvicorn

from .pipeline import OCRPipeline

app = FastAPI()
ocr = OCRPipeline()

class TextBox(BaseModel):
    text: str
    language: str
    bounding_box: List[float]
    confidence: float

@app.post("/ocr", response_model=List[TextBox])
async def run_ocr(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(BytesIO(img_bytes))
    results = ocr(image)
    # results should be list of dicts with keys: text, language, bbox, conf
    return [TextBox(
        text=r.get("text", ""),
        language=r.get("language", ""),
        bounding_box=r.get("bbox", []),
        confidence=r.get("conf", 0.0)
    ) for r in results]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
