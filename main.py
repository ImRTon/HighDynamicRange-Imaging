import argparse
import math
from distutils.util import strtobool
from typing import List
import cv2
import numpy as np
import os
import img_alignment as imgAlignUtils
import hdr_utils
from PIL import Image
from PIL.ExifTags import TAGS
import exifread
import matplotlib.pyplot as plt
from tqdm import tqdm
# import ToneMapping

def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('-i', '--input_dir', default='./imgs', type=str, help='Folder of input images.')
    parser.add_argument('-a', '--align_img', default='True', type=str, help='Whether to align img or not.')
    parser.add_argument('-s', '--sample_method', default='uniform', type=str, help='The way to sample points [uniform / random]')
    return parser

def imgImportFromPil(img_path: str):
    pil_img = Image.open(img_path).convert("RGB")
    cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    exif = pil_img.getexif()
    return cv_img


if __name__ == '__main__':
    img_contents = []
    '''
    [
        {
            "filepath": FILEPATH,
            "data": OPENCV_IMG,
            "MTBImg": (MTB_OPENCV_IMG, MASK_OPENCV_IMG),
            "offset": {"x": int, "y": int}, # 1 means shift left or top, -1 means shift right or down
            "alignedImg": OPENCV_IMG,
            "brightness": INT
        }
    ]
    '''

    parser = get_parser()
    args = parser.parse_args()
    av_brightness = 0
    for file in os.listdir(args.input_dir):
        file_lower = file.lower()
        if file_lower.endswith('.jpg') or file_lower.endswith('.png'):
            img_filePath = os.path.join(args.input_dir, file_lower)
            img = imgImportFromPil(img_filePath)
            img_mean = cv2.mean(img)
            exif = {}
            # Exif read
            with open(img_filePath, 'rb') as file:
                tags = exifread.process_file(file, details=False)
                # for key, val in tags.items():
                #     print(key, val)
                exif['exposure_time'] = eval(str(tags['EXIF ExposureTime']))
                # exif['iso'] = int(str(tags['EXIF ISOSpeedRatings']))
                # exif['focal_len'] = int(str(tags['EXIF FNumber']))
            img_contents.append({
                'filepath': img_filePath,
                'data': img,
                'MTBImg': imgAlignUtils.img2MTB(img),
                "offset": {"x": 0, "y": 0},
                'brightness': img_mean[2] * 0.299 + img_mean[1] * 0.587 + img_mean[0] * 0.114,
                'exif': exif
            })
            '''
                exif
                {
                    "exposure_time": float
                }
            '''

            if not strtobool(args.align_img):
                img_contents[-1]['alignedImg'] = img
            av_brightness += img_mean[2] * 0.299 + img_mean[1] * 0.587 + img_mean[0] * 0.114
    
    if strtobool(args.align_img):
        # Sort images by its brightness
        sorted(img_contents, key = lambda s: s['brightness'])
        av_brightness = av_brightness / len(img_contents)
        for i in range(len(img_contents) - 1, 0, -1):
            if img_contents[i]['brightness'] > av_brightness:
                mid_img_content = img_contents.pop(i)
                img_contents.insert(0, mid_img_content)
                break
        
        imgAlignUtils.try_align(img_contents)

        # Offset print
        print("Offsets:")
        for img_content in img_contents:
            print(img_content["offset"])
        
        imgAlignUtils.crop_imgs(img_contents)

    # imgAlignUtils.test(img_contents)

    ## HDR
    sample_pixel_vals = [[], [], []] # BGR
    pixel_vals = [[], [], []] # BGR
    exposures = []
    lamba = 30.0
    weightings = []

    hdr_utils.set_hdr_parameters(img_contents, pixel_vals, sample_pixel_vals, exposures, weightings, args.sample_method)

    rg, lE = hdr_utils.g_solver(sample_pixel_vals[2], exposures, lamba, weightings)
    gg, _ = hdr_utils.g_solver(sample_pixel_vals[1], exposures, lamba, weightings)
    bg, _ = hdr_utils.g_solver(sample_pixel_vals[0], exposures, lamba, weightings)

    plt.figure(figsize=(10,10))
    plt.plot(rg,range(256), 'r')
    plt.plot(gg,range(256), 'g')
    plt.plot(bg,range(256), 'b')
    plt.ylabel('Z (Energy)')
    plt.xlabel('log2(X)')
    plt.show()

    img_shape = img_contents[0]['alignedImg'].shape

    hdr_img = hdr_utils.get_radiance_map(img_contents, [bg, gg, rg], exposures, weightings, img_shape)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(np.log(cv2.cvtColor(hdr_img, cv2.COLOR_BGR2GRAY)), cmap='jet')
    plt.colorbar()
    plt.show()

    # ToneMapping.toneMapping_Reinhard(hdr_img, 0.5)

