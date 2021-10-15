import glob
import os.path as osp
import numpy as np
import cv2


def myHoughLineSegments(img_in, edgeimage, peakRho, peakTheta, rhosscale, thetasscale):
    # Your implemention
    original = np.copy(img_in)
    img_out = np.copy(img_in)

    height, width = img_in.shape[:2]
    for i in range(peakTheta.shape[0]):
        theta, rho = thetasscale[peakTheta[i]], rhosscale[peakRho[i]]
        theta = theta * np.pi / 180.

        if 0 < rho < width and abs(theta - 0) < 0.0000001 or abs(theta - np.pi) < 0.0000001:
            cv2.line(img_out, (abs(rho), 0), (abs(rho), height - 1), (0, 255, 0), 3)
        elif abs(theta - np.pi / 2) < 0.0000001 and 0 < rho < height:
            cv2.line(img_out, (0, int(rho)), (width - 1, int(rho)), (0, 255, 0), 3)
        else:
            m = -np.cos(theta) / np.sin(theta)
            b = rho / np.sin(theta)

            pts = set()
            # y == 0
            x = -b / m
            if 0 <= x < width:
                pts.add((int(x), 0))

            # x == 0
            y = b
            if 0 <= y < height:
                pts.add((0, int(y)))

            # y == height-1
            x = (height - 1 - b) / m
            if 0 <= x < width:
                pts.add((int(x), int(height-1)))

            # x == width-1
            y = m * (width - 1) + b
            if 0 <= y < height:
                pts.add((int(width-1), int(y)))

            pts = list(pts)
            if pts[0][0] < pts[1][0]:
                cv2.line(img_out, pts[0], pts[1], (0, 255, 0), 3)
            else:
                cv2.line(img_out, pts[1], pts[0], (0, 255, 0), 3)

        img_out[edgeimage == 0] = original[edgeimage == 0]

    return img_out
