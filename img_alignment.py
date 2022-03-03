import cv2
import numpy as np

def img2MTB(img):
    """_summary_

    Args:
        OPENCV_IMG: photograph

    Returns:
        OPENCV_IMG: an img thresholding with medium value
        OPENCV_IMG: a mask within medium +- 10
    """    
    
    img_h, img_w = img.shape[0], img.shape[1]
    mtb_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color_count = [0] * 256
    for row in range(img_h):
        for col in range(img_w):
            color_count[mtb_img[row][col]] += 1
    
    mid_th = int(img_h * img_w / 2)
    mid_count = 0
    mid_index = 0
    for i in range(256):
        if (mid_count < mid_th):
            mid_count += color_count[i]
        else:
            mid_index = i
            break
    _, img_th = cv2.threshold(mtb_img, mid_index, 255, cv2.THRESH_BINARY)
    img_mask = cv2.inRange(mtb_img, mid_index - 10, mid_index + 10)
    return img_th, img_mask