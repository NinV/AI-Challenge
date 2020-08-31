import cv2
import numpy as np


def histogram(img, box, bins=8):
    x_min, y_min, x_max, y_max = np.round(box).astype(np.int)
    patch = img[y_min:y_max, x_min: x_max]
    hist = cv2.calcHist([patch], [0, 1, 2], None, [bins, bins, bins], [0, 256]*3)
    hist = hist.flatten()
    return hist