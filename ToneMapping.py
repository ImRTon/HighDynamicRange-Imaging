import cv2
import numpy as np
import math

def toneMapping_Reinhard(radianceMap, alpha):
    height = radianceMap.shape[0]
    width = radianceMap.shape[1]
    Lw = np.zeros((height, width))
    LwBar = 0.0
    # Global operator
    for row in range(height):
        for col in range(width):
            # Convert from rgb to luminance
            [r, g, b] = radianceMap[row][col]
            Lw[row][col] = r * 0.2125 + g * 0.7154 + b * 0.0721
            LwBar += math.log(Lw[row][col] + 0.000001)
    
    # Compute LwBar
    LwBar = math.exp(LwBar / (height * width))
    print(LwBar)

    # Compute Lm
    Lm = Lw * alpha / LwBar
    #print(Lm)
    
    # Compute Ld
    # Find maximum luminance from all pixels
    Lwhite = np.max(Lm)
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

    for row in range(height):
        for col in range(width):
            result[row][col] = Ld[row][col] * radianceMap[row][col] / Lw[row][col]

    return result

# Load testing hdr file
img = cv2.imread("./imgs/memorial.hdr", flags=cv2.IMREAD_ANYDEPTH)
img = np.array(img)

result = np.ndarray(shape=img.shape)
result2 = np.ndarray(shape=img.shape)

# Test different alpha value
Lw1, Ld1 = toneMapping_Reinhard(img, alpha=0.7)
Lw2, Ld2 = toneMapping_Reinhard(img, alpha=0.1)

result = mapToRGB(Lw1, Ld1, img)
result2 = mapToRGB(Lw2, Ld2, img)

cv2.imshow('HDR', img)
cv2.imshow('LDR_0.7', result)
cv2.imshow('LDR_0.3', result2)
cv2.waitKey(0)