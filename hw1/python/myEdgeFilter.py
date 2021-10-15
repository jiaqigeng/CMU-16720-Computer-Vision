import cv2
import numpy as np
import numpy.matlib as npm
import math
from GaussianKernel import Gauss2D
from myImageFilterX import myImageFilterX


def nms(Io, Img1):
    Img1_pad = np.pad(Img1, ((1, 1), (1, 1)), 'constant')
    Img2 = np.zeros(Img1.shape)
    m, n = Io.shape
    for i in range(0, m):
        for j in range(0, n):
            k, l = i+1, j+1
            if (-22.5 <= Io[i, j] < 22.5) or (-180. <= Io[i, j] < -157.5) or (157.5 < Io[i, j] < 180):
                p1, p2 = Img1_pad[k, l+1], Img1_pad[k, l-1]
            elif (22.5 <= Io[i, j] < 67.5) or (-157.5 <= Io[i, j] < -112.5):
                p1, p2 = Img1_pad[k+1, l-1], Img1_pad[k-1, l+1]
            elif (67.5 <= Io[i, j] < 112.5) or (-112.5 <= Io[i, j] < -67.5):
                p1, p2 = Img1_pad[k+1, l], Img1_pad[k-1, l]
            else:
                p1, p2 = Img1_pad[k-1, l-1], Img1_pad[k+1, l+1]

            if p1 > Img1[i, j] or p2 > Img1[i, j]:
                Img2[i, j] = 0.
            else:
                Img2[i, j] = Img1[i, j]

    return Img2


def myEdgeFilter(img0, sigma):
    # Your implemention
    img0 = img0.astype(np.float32)
    kernel_size = 2 * math.ceil(3*sigma)+1
    smoothed = myImageFilterX(img0, Gauss2D((kernel_size, kernel_size), sigma))
    sobel_x = (1/8.) * np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = (1/8.) * np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = myImageFilterX(smoothed, sobel_x)
    Iy = myImageFilterX(smoothed, sobel_y)

    Io = np.arctan2(Iy, Ix) * 180 / np.pi
    Img1 = np.sqrt(Ix ** 2 + Iy ** 2)
    Img1 = nms(Io, Img1)
    return Img1, Io, Ix, Iy
