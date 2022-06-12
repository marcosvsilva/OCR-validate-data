import os

import onnxruntime

ROOT_PATH = os.getcwd()


class Model:
    def __init__(self):
        session = os.getenv("VISION_OCR_DETECT_YOLO_SESSION", "yolo.onnx")
        self.session_path = os.path.join(
            ROOT_PATH, "classify", "model", session
        )

        if not os.path.exists(self.session_path):
            raise Exception("Session file not found!")

    def session(self):
        return onnxruntime.InferenceSession(self.session_path)
