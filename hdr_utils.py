from random import random
from typing import List
import numpy as np
import cv2
import math

def set_hdr_parameters(img_contents, pixel_vals: List, sample_pixel_vals: List, exposures: List, weightings: List, sample_method: str):
    for i in range(128):
        weightings.append(i)
    for i in range(127, -1, -1):
        weightings.append(i)
    # weightings = [int(127.5 - abs(z - 127.5)) for z in range(256)]
    for img_content in img_contents:
        for i in range(3):
            img_channel = img_content['alignedImg'][:, :, i]
            pixel_vals[i].append(img_channel.flatten())
            if sample_method.lower() == "random":
                sample_pixel_vals[i].append(np.random.choice(img_channel.flatten(), size=100, replace=False))
            else:
                sample_pixel_vals[i].append(cv2.resize(img_channel, (10, 10)).flatten())
        exposures.append(math.log2(img_content['exif']['exposure_time']))

def g_solver(Z: List, B: List, lamba, weighting):
    n = 256
    A = np.zeros(shape=(np.size(Z, 0) * np.size(Z, 1) + n + 1, n + np.size(Z, 1)), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)

    k = 1; 
    # Include the data-fitting equations
    for i in range(np.size(Z, 1)):
        for j in range(np.size(Z, 0)):
            z = Z[j][i]
            wij = weighting[z]
            A[k][z] = wij
            A[k][n + i] = - wij
            b[k] = wij * B[j]
            k += 1

    # fix the curve by setting its middle value to 0
    A[k][128] = 1
    k += 1

    # Include the smoothness equations
    for i in range(n - 1):
        A[k][i]=lamba * weighting[i+1]
        A[k][i+1] = -2 * lamba * weighting[i+1]
        A[k][i+2] = lamba * weighting[i+1]
        k += 1

    # Solve the system using SVD
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[:n]
    logE = x[n:]

    return g, logE

def get_radiance_map(imgs_flat: List, response_curves: List, exposures: List, weightings: List, img_shape):

    hdr = np.zeros(img_shape, dtype=np.float32)
    for i in range(3):
        pass
