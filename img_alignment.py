import math
import cv2
import numpy as np
from tqdm import tqdm

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
    img_mask = 255 - cv2.inRange(mtb_img, mid_index - 10, mid_index + 10)
    #cv2.imshow('mask', img_mask)
    #cv2.waitKey()
    return img_th, img_mask

def try_align(img_contents, depth=4):
    # Use first image as standard
    std_img_content = None
    print("Image alignment begins.")
    progress = tqdm(total=len(img_contents))
    for img_content in img_contents:
        if std_img_content is None:
            std_img_content = img_content
            height, width = std_img_content['MTBImg'][0].shape
            depth = max(1, int(math.log2(min(height, width)) - 4))
            continue
        
        img_content['offset'] = {"x": 0, "y": 0}

        # Copy std image so I can crop it without impacting the origin
        ori_std_mtb_img = std_img_content['MTBImg'][0].copy()
        ori_std_mtb_mask = std_img_content['MTBImg'][1].copy()
        ori_mtb_img = img_content['MTBImg'][0].copy()
        for i in range(depth):
            height, width = ori_std_mtb_img.shape
            height = int(height / pow(2, depth - i - 1))
            width = int(width / pow(2, depth - i - 1))
            resized_std_img = cv2.resize(ori_std_mtb_img, (width, height))
            resized_std_mask = cv2.resize(ori_std_mtb_mask, (width, height))
            resized_img = cv2.resize(ori_mtb_img, (width, height))

            best_pix_count = 9999999999999
            best_offset = [0, 0]
            for j in range(-1, 2, 1):
                for k in range(-1, 2, 1):
                    x0, x1, y0, y1 = max(0, k), min(width, width + k), max(0, j), min(height, height + j)
                    std_mtb_img = resized_std_img[y0: y1, x0:x1]
                    std_mtb_mask = resized_std_mask[y0: y1, x0:x1]
                    mtb_img = resized_img[max(0, -j):min(height, height - j), max(0, -k):min(width, width - k)]
                    result = cv2.bitwise_xor(std_mtb_img, mtb_img, mask=std_mtb_mask)
                    result_pix_count = np.sum(result == 255)
                    if result_pix_count < best_pix_count:
                        best_pix_count = result_pix_count
                        best_offset = [k, j]
            if (best_offset[0] != 0 or best_offset[1] != 0):
                # Offset stored
                img_content['offset']["x"] += best_offset[0] * pow(2, depth - i - 1)
                img_content['offset']["y"] += best_offset[1] * pow(2, depth - i - 1)
                ori_height, ori_width = ori_std_mtb_img.shape
                x0, x1, y0, y1 = max(0, -best_offset[0]), min(ori_width, ori_width - best_offset[0]),\
                                    max(0, -best_offset[1]), min(ori_height, ori_height - best_offset[1])
                ori_std_mtb_img = ori_std_mtb_img[y0:y1, x0:x1]
                ori_std_mtb_mask = ori_std_mtb_mask[y0:y1, x0:x1]
                ori_mtb_img = ori_mtb_img[y0:y1, x0:x1]
            # print(best_offset, best_pix_count)
        # print("RES:", ori_mtb_img.shape, img_content["offset"])
        progress.update(1)
    progress.close()

def crop_imgs(img_contents):
    minX, maxX, minY, maxY = 0, 0, 0, 0
    for img_content in img_contents:
        offset_x, offset_y = img_content['offset']["x"], img_content['offset']["y"]
        minX = min(offset_x, minX)
        maxX = max(offset_x, maxX)
        minY = min(offset_y, minY)
        maxY = max(offset_y, maxY)
    width = img_content['data'].shape[1] - abs(minX) - abs(maxX)
    height = img_content['data'].shape[0] - abs(minY) - abs(maxY)
    
    for img_content in img_contents:
        offset_x, offset_y = img_content['offset']["x"], img_content['offset']["y"]
        img_content['alignedImg'] = img_content['data'][maxY - offset_y:height + maxY - offset_y, maxX - offset_x:width + maxX - offset_x]

def test(img_contents):
    av_img = None
    av_img_no_align = None
    av_img_mtb = None
    for i, img_content in enumerate(img_contents):
        if av_img is None:
            av_img = img_content['alignedImg']
            continue
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        av_img = cv2.addWeighted(img_content['alignedImg'], alpha, av_img, beta, 0.0)
    
    for i, img_content in enumerate(img_contents):
        if av_img_no_align is None:
            av_img_no_align = img_content['data']
            continue
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        av_img_no_align = cv2.addWeighted(img_content['data'], alpha, av_img_no_align, beta, 0.0)
    
    for i, img_content in enumerate(img_contents):
        if av_img_mtb is None:
            av_img_mtb = img_content['MTBImg'][0]
            continue
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        av_img_mtb = cv2.addWeighted(img_content['MTBImg'][0], alpha, av_img_mtb, beta, 0.0)

    cv2.imshow("res", av_img)
    cv2.imshow("ori", av_img_no_align)
    cv2.imshow("mtb", av_img_mtb)
    cv2.imwrite('D://ori_img.png', av_img_no_align)
    cv2.imwrite('D://aligned_img.png', av_img)
    cv2.waitKey()