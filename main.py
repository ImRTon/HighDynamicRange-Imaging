import argparse
import cv2
import numpy as np
import os
import img_alignment as imgAlignUtils
from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('-i', '--input_dir', default='./imgs', type=str, help='Folder of input images.')
    return parser

def imgImportFromPil(img_path: str):
    pil_img = Image.open(img_path).convert("RGB")
    cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return cv_img

if __name__ == '__main__':
    img_contents = []
    '''
    [
        {
            "filepath": FILEPATH,
            "data": OPENCV_IMG
        }
    ]
    '''

    parser = get_parser()
    args = parser.parse_args()
    print(args.input_dir)

    for file in os.listdir(args.input_dir):
        file_lower = file.lower()
        if file_lower.endswith('.jpg') or file_lower.endswith('.png'):
            img_filePath = os.path.join(args.input_dir, file_lower)
            img_contents.append({
                'filepath': img_filePath,
                'data': imgImportFromPil(img_filePath)
            })

    imgAlignUtils.img2MTB(img_contents[0]['data'])

