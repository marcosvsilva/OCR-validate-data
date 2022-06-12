import cv2
import numpy as np

from PIL import Image
from io import BytesIO

from fastapi import APIRouter, File, UploadFile

from classify.classify import Classify

# ---
#  GENERAL PROCESS
# ---

router = APIRouter(
    prefix="", tags=["ocr_validate"], responses={404: {"description": "Not found"}}
)


@router.post("/process")
async def process(image: UploadFile = File(...)):
    print(image.filename)
    
    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        raise Exception("Image format is invalid!")

    contents = await image.read()

    stream = BytesIO(contents)
    img0 = Image.open(stream).convert("RGB")
    img = np.asarray(img0)
    stream.close()

    classify = Classify()
    result = classify.classify(img)

    print(result)

    return result
