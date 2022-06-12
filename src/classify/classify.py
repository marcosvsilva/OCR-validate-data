from ast import Mod
import os
import numpy as np
import cv2
import torch
from routers.vision import pre_process

from vision.pre_process import PreprocessImage
from classify.utils import PointsDetection, letterbox, non_max_suppression, scale_coords
from classify.model import Model
from classify.segmentation import Segmentation


class Classify:
    def __init__(self):
        session = Model()
        self.segmentation = Segmentation()
        self.preprocess = PreprocessImage()

        self.new_shape = int(os.getenv("VISION_OCR_PREPROCESSED_IMG_SIZE", "640"))
        self.stride = int(os.getenv("VISION_OCR_DETECT_YOLO_STRIDE", "32"))
        self.conf_tree = float(os.getenv("VISION_OCR_DETECT_YOLO_CONF_THREES", "0.25"))
        self.iou_three = float(os.getenv("VISION_OCR_DETECT_YOLO_IOU_THREES", "0.45"))
        self.max_det = int(os.getenv("VISION_OCR_DETECT_YOLO_MAX_DET", "100"))

        self.width = int(os.getenv("VISION_OCR_PREPROCESSED_IMG_SIZE", "640"))
        self.height = int(os.getenv("VISION_OCR_PREPROCESSED_IMG_SIZE", "640"))
        
        self.session = session.session()


    def classify(self, img0):
        if img0.shape is None:
            raise Exception("Image is none!")

        # img = self.preprocess.preprocess(np.copy(img0))
        img = np.copy(img0)

        points = self.detect(img)

        images_result = []
        for point in points:
            crop_size_1 = 130
            crop_size_2 = 130

            img_crop_pad = img0[
                point.c1[1] - crop_size_1 : point.c2[1] + crop_size_1,
                point.c1[0] - crop_size_2 : point.c2[0] + crop_size_2,
                ::1,
            ]

            img_crop_pad = cv2.resize(
                img_crop_pad, (self.width, self.height), interpolation=cv2.INTER_AREA
            )


            images_result.append(
                {point.label: self.segmentation.segmentation(point.label, img_crop_pad)}
            )

        return img_crop_pad


    def detect(self, img0):
        img = letterbox(
            img0, new_shape=self.new_shape, stride=self.stride, auto=False
        )[0]

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = img.astype("float32")
        img = img / 255.0

        if len(img.shape) == 3:
            img = img[None]

        pred = torch.tensor(
            self.session.run(
                [self.session.get_outputs()[0].name],
                {self.session.get_inputs()[0].name: img},
            )
        )

        pred = non_max_suppression(
            pred, self.conf_tree, self.iou_three, None, False, max_det=self.max_det
        )

        names = ["rg-frente", "rg-verso", "cnh"]
        results = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, cls in reversed(det):
                    centroid = int(cls)
                    label = names[centroid]
                    centroid1, centroid2 = (int(xyxy[0]), int(xyxy[1])), (
                        int(xyxy[2]),
                        int(xyxy[3]),
                    )

                    detection = PointsDetection(label, centroid1, centroid2)
                    results.append(detection)

        return results
