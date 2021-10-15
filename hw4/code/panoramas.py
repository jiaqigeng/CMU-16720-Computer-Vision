import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite, briefMatch, plotMatches
import matplotlib.pyplot as plt


def imageStitching(im1, im2, H2to1):
    """
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    """
    #######################################
    im1_pano = np.zeros((im1.shape[0] + 80, im1.shape[1] + 750, 3), dtype=np.uint8)
    im1_pano[: im1.shape[0], : im1.shape[1], : im1.shape[2]] = im1
    im1_pano_mask = im1_pano > 0

    # TODO ...
    # warp im2 onto pano
    pano_im = cv2.warpPerspective(im2, H2to1, (im1_pano.shape[1], im1_pano.shape[0]))
    pano_im_mask = pano_im > 0

    # TODO
    # dealing with the center where images meet.
    im_full = pano_im.astype(np.uint32) + im1_pano.astype(np.uint32)

    im_R = im_full * np.logical_not(im1_pano_mask)
    im_L = im_full * np.logical_not(pano_im_mask)

    # TODO produce im center, mix of pano_im and im1_pano
    im_center = (im_full / 2).astype(np.uint8) * np.logical_and(im1_pano_mask, pano_im_mask)
    return im_R + im_L + im_center


def imageStitching_noClip(im1, im2, H2to1):
    """
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    """
    ######################################
    # TO DO ...
    s = 1
    tx = 0
    # clip
    # establish corners
    # create new corners

    im1_corners = np.array([(0, 0), (0, im1.shape[0]), (im1.shape[1], 0), (im1.shape[1], im1.shape[0])]).T
    im2_corners = np.array([(0, 0), (0, im2.shape[0]), (im2.shape[1], 0), (im2.shape[1], im2.shape[0])]).T

    im2_corners = np.vstack((im2_corners, np.ones(4)))
    im2_corners_transformed = np.dot(H2to1, im2_corners)

    im2_corners_transformed[0, :] = im2_corners_transformed[0, :] / im2_corners_transformed[2, :]
    im2_corners_transformed[1, :] = im2_corners_transformed[1, :] / im2_corners_transformed[2, :]

    h_min = min(im2_corners_transformed[1, :].min(), im1_corners[1, :].min())
    h_max = max(im2_corners_transformed[1, :].max(), im1_corners[1, :].max())
    w_min = min(im2_corners_transformed[0, :].min(), im1_corners[0, :].min())
    w_max = max(im2_corners_transformed[0, :].max(), im1_corners[0, :].max())

    # calculate the correct height and scale
    # canvas_width = int(im1.shape[1] + 750)
    # aspect_ratio = (h_max-h_min) / (w_max-w_min)
    # canvas_height = int(canvas_width * aspect_ratio)
    # s = canvas_width / (w_max-w_min)

    # change s directly
    s = 1
    canvas_width, canvas_height = int((w_max-w_min) * s), int((h_max-h_min) * s)

    tx = -w_min * s
    ty = -h_min * s

    # you actually dont need to use M_scale for the pittsburgh city stitching.
    M_scale = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    M_translate = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    M = np.dot(M_translate, M_scale)

    # TODO fill in the arguments
    pano_im2 = cv2.warpPerspective(im2, np.matmul(M, H2to1), (canvas_width, canvas_height)).astype(np.uint8)
    pano_im1 = cv2.warpPerspective(im1, M, (canvas_width, canvas_height)).astype(np.uint8)

    im1_pano_mask = pano_im1 > 0
    im2_pano_mask = pano_im2 > 0

    # TODO
    # should be same line as what you implemented in line 32, in imagestitching
    pano_im_full = pano_im1.astype(np.uint32) + pano_im2.astype(np.uint32)

    im_R = pano_im_full * np.logical_not(im1_pano_mask)
    im_L = pano_im_full * np.logical_not(im2_pano_mask)
    # should be same line as what you implemented in line 39, in imagestitching
    im_center = (pano_im_full / 2).astype(np.uint8) * np.logical_and(im1_pano_mask, im2_pano_mask)
    return im_center + im_R + im_L


def generatePanorama(im1, im2):
    H2to1 = np.load("bestH.npy")
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im


if __name__ == "__main__":
    # 7.1
    # im1 = cv2.imread("../data/incline_L.png")
    # im2 = cv2.imread("../data/incline_R.png")
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    # plotMatches(im1, im2, matches, locs1, locs2)
    # H2to1 = ransacH(matches, locs1, locs2, num_iter=10000, tol=1)

    # TODO
    # save bestH.npy
    # np.save("../results/q7_1.npy", H2to1)
    # pano_im = imageStitching(im1, im2, H2to1).astype(np.uint8)
    # cv2.imwrite("../results/7_1.jpg", pano_im)

    # 7.3 incline
    # im1 = cv2.imread("../data/incline_L.png")
    # im2 = cv2.imread("../data/incline_R.png")
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    # plotMatches(im1, im2, matches, locs1, locs2)
    # H2to1 = ransacH(matches, locs1, locs2, num_iter=10000, tol=1)
    # np.save("bestH.npy", H2to1)
    # pano_im = generatePanorama(im1, im2).astype(np.uint8)
    # cv2.imwrite("../results/7_3_incline.jpg", pano_im)

    # 7.3 hi
    im1_original = cv2.imread("../data/hi_L.jpg")
    im2_original = cv2.imread("../data/hi_R.jpg")
    im1 = cv2.resize(im1_original, (int(im1_original.shape[1] / 10), int(im1_original.shape[0] / 10)))
    im2 = cv2.resize(im2_original, (int(im2_original.shape[1] / 10), int(im2_original.shape[0] / 10)))
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    locs1[:, :2] = locs1[:, :2] * 10
    locs2[:, :2] = locs2[:, :2] * 10
    plotMatches(im1_original, im2_original, matches, locs1, locs2)

    H2to1 = ransacH(matches, locs1, locs2, num_iter=50000, tol=20)
    np.save("bestH.npy", H2to1)
    pano_im = generatePanorama(im1_original, im2_original).astype(np.uint8)
    cv2.imwrite("../results/7_3_hi.jpg", pano_im)
