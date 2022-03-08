import argparse
import cv2
import numpy as np
import os
import img_alignment as imgAlignUtils
from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('-i', '--input_dir', default='./test_imgs', type=str, help='Folder of input images.')
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
            "data": OPENCV_IMG,
            "MTBImg": (MTB_OPENCV_IMG, MASK_OPENCV_IMG),
            "offset": {"x": int, "y": int},
            "alignedImg" : OPENCV_IMG
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
            img = imgImportFromPil(img_filePath)
            img_contents.append({
                'filepath': img_filePath,
                'data': img,
                'MTBImg': imgAlignUtils.img2MTB(img)
            })
    imgAlignUtils.try_align(img_contents)

    for img_content in img_contents:
        print(img_content["offset"])
    
    imgAlignUtils.crop_imgs(img_contents)
    
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
    
    # Test block begin
    ###############################
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
    ###############################
    # Test block end

