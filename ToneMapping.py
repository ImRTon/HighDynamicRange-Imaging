import cv2
import numpy as np
import math
from tqdm import tqdm

def toneMapping_Reinhard(radianceMap, alpha):
    height = radianceMap.shape[0]
    width = radianceMap.shape[1]
    Lw = np.zeros((height, width))
    LwBar = 0.0
    # Global operator
    print("Compute LwBar")
    progress = tqdm(total=height + 1)
    for row in range(height):
        for col in range(width):
            # Convert from rgb to luminance
            [r, g, b] = radianceMap[row][col]
            Lw[row][col] = r * 0.2125 + g * 0.7154 + b * 0.0721
            LwBar += math.log(Lw[row][col] + 0.000001)
        progress.update(1)
    
    # Compute LwBar
    LwBar = math.exp(LwBar / (height * width))
    progress.update(1)
    progress.close()

    # Compute Lm
    Lm = Lw * alpha / LwBar
    
    # Compute Ld
    # Find maximum luminance from all pixels
    sortedLw = np.sort(Lw.flatten())
    Lwhite = sortedLw[sortedLw.shape[0] - 5000]
    print(Lwhite)
    Lwhite2 = Lwhite * Lwhite
    Ld = (Lm * (1 + Lm / Lwhite2)) / (1 + Lm)

    return Lw, Ld

# ref: https://github.com/brunopop/hdr/blob/master/HDR/Tonemap.cpp
# mapping from luminance to RGB
def mapToRGB(Lw, Ld, radianceMap):
    height = radianceMap.shape[0]
    width = radianceMap.shape[1]
    result = np.zeros((height, width, 3))

    print("Compute RGB mapping")
    progress = tqdm(total=height)
    for row in range(height):
        for col in range(width):
            result[row][col] = Ld[row][col] * radianceMap[row][col] / Lw[row][col] * 255
        progress.update(1)
    progress.close()

    return result