from PIL import Image
import cv2
import numpy as np


def mask2img(img, mask, rd):
    if rd == "Cropped":
        image = img.convert('RGBA')
        image = np.array(image, dtype=np.uint8)
        mask = mask.convert("L").resize(img.size)
        mask = np.array(mask, dtype=np.uint8)
        image = cv2.bitwise_and(image, image, mask=mask)
    elif rd == "Highlighted":
        image = np.array(img.convert("RGB"), dtype=np.uint8)
        m_image = image.copy()
        mask = np.array(mask.convert("RGB").resize(img.size), dtype=np.uint8)
        m_image = np.where(mask.astype(int),
        np.array([0,255,0], dtype='uint8'),
        m_image)
        m_image = m_image.astype(np.uint8)
        image = cv2.addWeighted(image, 0.3, m_image, 0.7, 0)
    return Image.fromarray(image)
