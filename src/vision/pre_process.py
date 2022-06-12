import cv2
import os

class PreprocessImage:
    def __init__(self):
        self.new_shape = int(os.getenv("VISION_OCR_PREPROCESSED_IMG_SIZE", "640"))

    def preprocess(self, img0):
        img = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        return img
