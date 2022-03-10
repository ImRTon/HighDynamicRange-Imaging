import cv2
import numpy as np
import math

def toneMapping_Reinhard(radianceMap, channel, alpha):
    height = radianceMap.shape[0]
    width = radianceMap.shape[1]
    Lw = np.zeros((height, width))
    LwBar = 0.0
    # Global operator
    for row in range(height):
        for col in range(width):
            # Convert from rgb to luminance
            Lw[row][col] = radianceMap[row][col][channel]
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

    return Ld

# Load testing hdr file
img = cv2.imread("./imgs/memorial.hdr", flags=cv2.IMREAD_ANYDEPTH)
img = np.array(img)

result = np.ndarray(shape=img.shape)
result2 = np.ndarray(shape=img.shape)

# Test different alpha value
for channel in range(0,3):
    result[:,:,channel] = toneMapping_Reinhard(img, channel, alpha=0.7)
    result2[:,:,channel] = toneMapping_Reinhard(img, channel, alpha=0.1)

cv2.imshow('HDR', img)
cv2.imshow('LDR_0.7', result)
cv2.imshow('LDR_0.3', result2)
cv2.waitKey(0)