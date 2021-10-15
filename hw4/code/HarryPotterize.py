import numpy as np
import cv2
import skimage.io
from BRIEF import briefLite, briefMatch, plotMatches
from planarH import ransacH, compositeH


# warp harry potter onto cv desk image
# save final image as final_image
# TODO
im1 = cv2.imread("../data/pf_desk.jpg")
im2 = cv2.imread("../data/pf_scan_scaled.jpg")
im3 = cv2.imread("../data/hp_cover.jpg")
locs1, desc1 = briefLite(im1)
locs2, desc2 = briefLite(im2)
matches = briefMatch(desc1, desc2)
print(matches.shape)
im3 = cv2.resize(im3, (im2.shape[1], im2.shape[0]), interpolation=cv2.INTER_AREA)
H = ransacH(matches, locs1, locs2, num_iter=700, tol=20)
print(H)
final = compositeH(H, im1, im3)
plotMatches(im1, im2, matches, locs1, locs2)
cv2.imwrite("../results/6_1_iter_700_tol_20.jpg", final)
