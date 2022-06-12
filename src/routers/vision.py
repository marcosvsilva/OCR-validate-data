import cv2

from PIL import Image
from io import BytesIO

from fastapi import APIRouter, File, UploadFile

# ---
#  PRE-PROCESS IMAGES
# ---

router = APIRouter(
    prefix="/vision", tags=["VISION"], responses={404: {"description": "Not found"}}
)


@router.post("/preprocess-image")
async def pre_process(image: UploadFile = File(...)):
    print(image.filename)
    contents = await image.read()

    stream = BytesIO(contents)
    image = Image.open(stream).convert("RGBA")
    stream.close()

    image.show()
    
    return "end"
