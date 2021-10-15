import numpy as np
import cv2
from BRIEF import briefLite, briefMatch, plotMatches
import matplotlib.pyplot as plt


def computeH(p1, p2):
    """
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    """
    assert p1.shape[1] == p2.shape[1]
    assert p1.shape[0] == 2
    #############################
    # TO DO ...
    n = p1.shape[1]
    A = np.zeros((2*n, 9))

    U, V = p2[0, :].reshape(-1, 1), p2[1, :].reshape(-1, 1)
    X, Y = p1[0, :].reshape(-1, 1), p1[1, :].reshape(-1, 1)

    A[::2] = np.hstack((-U, -V, -np.ones((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), U * X, V * X, X))
    A[1::2] = np.hstack((np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), -U, -V, -np.ones((n, 1)), U * Y, V * Y, Y))

    eigen_vals, eigen_vecs = np.linalg.eigh(np.dot(A.T, A))
    H2to1 = eigen_vecs[:, 0].reshape((3, 3))
    return H2to1 / H2to1[2, 2]


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    """
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    """
    ###########################
    # TO DO ...
    p1_original, p2_original = locs1[matches[:, 0], :2].T, locs2[matches[:, 1], :2].T
    best_num_inliers = 0
    best_inliers_p1, best_inliers_p2 = None, None

    for i in range(num_iter):
        idx = np.random.choice(np.arange(p1_original.shape[1]), 4)
        p1 = p1_original[:, idx]
        p2 = p2_original[:, idx]
        H = computeH(p1, p2)

        p1_homo = np.vstack((p1_original, np.ones(p1_original.shape[1])))
        p2_homo = np.vstack((p2_original, np.ones(p2_original.shape[1])))
        p2_transform = np.dot(H, p2_homo)

        p2_transform[0, :] = p2_transform[0, :] / p2_transform[2, :]
        p2_transform[1, :] = p2_transform[1, :] / p2_transform[2, :]

        dist = np.sqrt((p2_transform[0, :] - p1_homo[0, :]) ** 2 + (p2_transform[1, :] - p1_homo[1, :]) ** 2)
        num_inliers = np.where(dist <= tol)[0].shape[0]

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers_p1 = p1_original[:, np.where(dist < tol)[0]]
            best_inliers_p2 = p2_original[:, np.where(dist < tol)[0]]

    bestH = computeH(best_inliers_p1, best_inliers_p2)
    print(best_num_inliers)
    return bestH


def compositeH(H, template, img):
    """
    Returns final warped harry potter image. 
    INPUTS
        H - homography 
        template - desk image
        img - harry potter image
    OUTPUTS
        final_img - harry potter on book cover image  
    """
    # TODO
    warp_img = cv2.warpPerspective(img, H, (template.shape[1], template.shape[0]))
    warp_img_mask = warp_img > 0

    final_img = template * np.logical_not(warp_img_mask)
    final_img = final_img + warp_img
    return final_img


if __name__ == "__main__":
    im1 = cv2.imread("../data/model_chickenbroth.jpg")
    im2 = cv2.imread("../data/chickenbroth_01.jpg")
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
