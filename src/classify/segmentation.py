import os

import cv2
import numpy as np

# from classify.transform import four_point_transform


class Segmentation:
    def __init__(self):
        self.width = int(os.getenv("VISION_OCR_PREPROCESSED_IMG_SIZE", "640"))
        self.height = int(os.getenv("VISION_OCR_PREPROCESSED_IMG_SIZE", "640"))

    def segmentation(self, type_document, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(gray, 75, 200)

        mask = cv2.GaussianBlur(edged, (5, 5), 11)
        bg = np.ones([edged.shape[0], edged.shape[1], 1], dtype="uint8") * 255

        smoothed = np.ones(edged.shape, dtype="uint8")

        for r in range(edged.shape[0]):
            for c in range(edged.shape[1]):
                smoothed[r][c] = int(
                    bg[r][c] * (mask[r][c]) + edged[r][c] * (1 - mask[r][c])
                )

        p1, p2, p3, p4 = self.get_points(smoothed)

        color = (255, 255, 255)
        img_marker = cv2.circle(np.copy(image), p1, 4, color, 2)
        img_marker = cv2.circle(img_marker, p2, 4, color, 2)
        img_marker = cv2.circle(img_marker, p3, 4, color, 2)
        img_marker = cv2.circle(img_marker, p4, 4, color, 2)

        points = np.array([p1, p2, p3, p4])

        list_images_for_ocr = self.segmentation_parts(image, type_document)

        return list_images_for_ocr

    def get_points(self, image):
        i, j = image.shape
        pad = 10

        m = max(image.shape)
        p1, p2, p3, p4 = (0, 0), (m, m), (0, 0), (m, m)
        for k in range(pad, i - pad):
            for c in range(pad, j - pad):
                if image[k, c] > 0:
                    if p1[0] < c:
                        p1 = (c, k)
                    elif p2[0] > c:
                        p2 = (c, k)
                    elif p3[1] < k:
                        p3 = (c, k)
                    elif p4[1] > k:
                        p4 = (c, k)

        return p1, p2, p3, p4

    def segmentation_parts(self, image, type_document):
        list_images = []
        if type_document == "cnh":
            cnh_nome_c1 = (140, 130)
            cnh_nome_c2 = (540, 150)

            nome = image[
                cnh_nome_c1[1] : cnh_nome_c2[1],
                cnh_nome_c1[0] : cnh_nome_c2[0], ::1
            ]
            list_images.append({"nome": nome})

            cnh_id_c1 = (320, 210)
            cnh_id_c2 = (450, 230)

            cpf = image[
                cnh_id_c1[1] : cnh_id_c2[1],
                cnh_id_c1[0] : cnh_id_c2[0], ::1    
            ]
            list_images.append({"cpf": cpf})

            cnh_filiacao_c1 = (320, 240)
            cnh_filiacao_c2 = (560, 335)

            filiacao = image[
                cnh_filiacao_c1[1] : cnh_filiacao_c2[1],
                cnh_filiacao_c1[0] : cnh_filiacao_c2[0],
                ::1,
            ]
            list_images.append({"filiacao": filiacao})

            cnh_nregistro_c1 = (130, 400)
            cnh_nregistro_c2 = (300, 420)

            nregistro = image[
                cnh_nregistro_c1[1] : cnh_nregistro_c2[1],
                cnh_nregistro_c1[0] : cnh_nregistro_c2[0],
                ::1,
            ]
            list_images.append({"nregistro": nregistro})

        return list_images
