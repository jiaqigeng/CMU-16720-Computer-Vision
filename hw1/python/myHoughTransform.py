import cv2
import numpy as np


def myHoughTransform(InnputImage, rho_resolution, theta_resolution):
    # Your implemention
    m, n = InnputImage.shape
    d = np.ceil(np.sqrt(m ** 2 + n ** 2))
    rhos = np.linspace(-d, d, int(2*d / rho_resolution), dtype=np.int)
    thetas = np.linspace(0, 180, int(180 / theta_resolution), dtype=np.int)
    y_idx, x_idx = np.nonzero(InnputImage)
    H = np.zeros((rhos.shape[0], thetas.shape[0]))

    for j in range(thetas.shape[0]):
        theta = thetas[j] * np.pi / 180.
        rho = np.round(((x_idx * np.cos(theta) + y_idx * np.sin(theta)) + d) / rho_resolution).astype(np.int)+1
        for r in rho:
            H[r, j] += 1

    return H, rhos, thetas
