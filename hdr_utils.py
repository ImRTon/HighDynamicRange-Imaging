import enum
from random import random
from typing import List
import numpy as np
import cv2
import math
from tqdm import tqdm
import gc

def set_hdr_parameters(img_contents, pixel_vals: List, sample_pixel_vals: List, exposures: List, weightings: List, sample_method: str):
    for i in range(128):
        weightings.append(max(0.001, i))
    for i in range(127, -1, -1):
        weightings.append(max(0.001, i))
    # weightings = [int(127.5 - abs(z - 127.5)) for z in range(256)]
    for img_content in img_contents:
        for i in range(3):
            img_channel = img_content['alignedImg'][:, :, i]
            pixel_vals[i].append(img_channel.astype(int).flatten())
            if sample_method.lower() == "random":
                sample_pixel_vals[i].append(np.random.choice(img_channel.flatten(), size=100, replace=False))
            else:
                sample_pixel_vals[i].append(cv2.resize(img_channel, (10, 10)).flatten())
        exposures.append(math.log(img_content['exif']['exposure_time']))

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

def get_radiance_map(img_contents, response_curves: List, exposures: List, weightings: List, img_shape):
    print("Counting radiance map")
    hdr = np.zeros(img_shape, dtype=np.float32)
    progress = tqdm(total=img_shape[0])
    for row in range(img_shape[0]):
        for col in range(img_shape[1]):
            for channel in range(img_shape[2]):
                denominator = 0
                numerator = 0
                for i, img_content in enumerate(img_contents):
                    img = img_content['alignedImg']
                    Z = img[row][col][channel]
                    numerator += weightings[Z] * (response_curves[channel][Z] -exposures[i])
                    denominator += weightings[Z]
                hdr[row][col][channel] = pow(2, numerator / denominator)
            # hdr[row][col] = math.exp(numerator / denominator)
            # hdr[row][col] = pow(2, numerator / denominator)
        progress.update(1)
    progress.close()

    return hdr

def get_radiance_map_np(pixel_vals, response_curves, exposures, weightings, img_shape):
    hdr = np.zeros((img_shape[0] * img_shape[1], img_shape[2]), dtype=np.float32)
    progress = tqdm(total=len(pixel_vals) * len(pixel_vals[0]))
    for i, channel in enumerate(pixel_vals):
        denominator = np.zeros((img_shape[0] * img_shape[1]), dtype=np.float32)
        numerator = np.zeros((img_shape[0] * img_shape[1]), dtype=np.float32)
        for j, img_channel in enumerate(channel):
            numerator += weightings[img_channel] * (response_curves[i][img_channel].flatten() - exposures[j])
            denominator += weightings[img_channel]
            progress.update(1)
        hdr[:, i] = np.exp(numerator / denominator)
    progress.close()
    hdr = hdr.reshape(img_shape)
    return hdr

