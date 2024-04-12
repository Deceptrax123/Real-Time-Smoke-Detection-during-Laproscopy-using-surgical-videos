import sys
import cv2
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
from math import pi


def cropToCentreAdaptive(img: cv2.Mat, threshold: float = 0.85, step_size: int = 32, starting_size: int = 32) -> cv2.Mat:
    img = cv2.resize(img, (512, 512))
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_bw, 75, 255, cv2.THRESH_BINARY)
    finalSize = 400
    c = 512//2, 512//2
    lastSize = starting_size
    for size in range(starting_size, 512, step_size):
        maskA = np.ones((512, 512, 1), dtype=np.uint8)*255
        maskB = np.zeros((512, 512, 1), dtype=np.uint8)
        cv2.circle(maskA, (256, 256), size//2, (0), -1)
        cv2.circle(maskB, (256, 256), (size+step_size)//2, (255), -1)
        combined_mask = cv2.bitwise_and(maskA, maskB)
        mask = cv2.bitwise_and(img, img, mask=combined_mask)
        img_bw_box = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh_box = cv2.threshold(img_bw_box, 2, 255, cv2.THRESH_BINARY)
        area = pi*((size+step_size)//2)**2 - pi*(size//2)**2
        ratio_black = cv2.countNonZero(thresh_box)/(area)

        if (ratio_black < threshold):
            finalSize = lastSize
            break
        lastSize = size
    return cv2.resize(cropToCentre(img, (finalSize, finalSize)), (512, 512))


def cropToCentre(img: cv2.Mat, dim: Tuple[int] = (512, 512)) -> cv2.Mat:
    # process crop width and height for max available dimension
    width, height = img.shape[1], img.shape[0]
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img
