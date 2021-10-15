import numpy as np


def nms(H):
    H_pad = np.pad(H, ((1, 1), (1, 1)), 'constant')
    nms_H = np.zeros(H.shape)
    for i in range(0, H.shape[0]):
        for j in range(0, H.shape[1]):
            k, l = i+1, j+1
            if H[i, j] > H_pad[k, l-1] and H[i, j] > H_pad[k, l+1] and \
               H[i, j] > H_pad[k-1, l-1] and H[i, j] > H_pad[k-1, l] and H[i, j] > H_pad[k-1, l+1] and \
               H[i, j] > H_pad[k+1, l-1] and H[i, j] > H_pad[k+1, l] and H[i, j] > H_pad[k+1, l+1]:
                nms_H[i, j] = H[i, j]
            else:
                nms_H[i, j] = 0

    return nms_H


def myHoughLines(H, nLines):
    # Your implemention
    rhos, thetas = np.unravel_index(np.argsort(nms(H).ravel()), H.shape)
    rhos, thetas = rhos[-nLines:], thetas[-nLines:]
    return rhos, thetas
