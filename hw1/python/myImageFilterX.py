import cv2
import numpy as np
import scipy
import scipy.signal


def vectorize(n, k, l):
    vec = np.zeros((k, l), dtype=int)
    base = np.arange(l)
    for i in range(k):
        vec[i, :] = base + i*n

    return vec.flatten()


def find_start_idx(m, n, k, l):
    idx_array = np.reshape(np.arange(m*n), (m, n))
    return idx_array[:-(k-1), :-(l-1)].flatten()


def myImageFilterX(img0, hfilt):
    # Your implemention
    img0 = img0.astype(np.float32)

    # padding
    height, width = img0.shape
    k, l = hfilt.shape
    pad_size_k, pad_size_l = int(k / 2.), int(l / 2.)
    img0_pad = np.pad(img0, ((pad_size_k, pad_size_k), (pad_size_l, pad_size_l)), 'edge')
    m, n = img0_pad.shape

    vec = vectorize(n, k, l)
    start_idx = np.reshape(find_start_idx(m, n, k, l), (-1, 1))
    vectorized = (img0_pad.flatten())[(start_idx + vec).flatten()]
    vectorized = np.reshape(vectorized, (-1, k*l))
    hfilt_flip = np.flip(hfilt, axis=0)
    hfilt_flip = np.flip(hfilt_flip, axis=1)
    img1 = np.dot(vectorized, hfilt_flip.flatten()).reshape((height, width))

    return img1
