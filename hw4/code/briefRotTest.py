import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import BRIEF


def main():
    np.seterr(divide='ignore', invalid='ignore')
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    degrees = []
    num_matches = []

    for i in range(36):
        M = cv2.getRotationMatrix2D((im1.shape[1]/2, im1.shape[0]/2), 10*i, 1)
        im2 = cv2.warpAffine(im1, M, (im1.shape[1], im1.shape[0]))

        locs1, desc1 = BRIEF.briefLite(im1)
        locs2, desc2 = BRIEF.briefLite(im2)
        matches = BRIEF.briefMatch(desc1, desc2)
        degrees.append(10*i)
        num_matches.append(matches.shape[0])

    plt.bar(degrees, num_matches, width=5)
    plt.show()


if __name__ == '__main__':
    main()
