# ##################################################################### #
# 16720B: Computer Vision Homework 5
# Carnegie Mellon University
# Oct. 26, 2020
# ##################################################################### #

import numpy as np
from submission import triangulate, eightpoint, epipolarCorrespondence
import matplotlib.pyplot as plt
from findM2 import test_M2_solution

'''
Q3.7:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

data1 = np.load("../data/templeCoords.npz")
x1s, y1s = data1['x1'], data1['y1']
intrinsics = np.load('../data/intrinsics.npz')

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

M = 640
F = eightpoint(data['pts1'], data['pts2'], M)

x1_y1_s = []
x2_y2_s = []
for i in range(x1s.shape[0]):
    x1, y1 = x1s[i].item(), y1s[i].item()
    x2, y2 = epipolarCorrespondence(im1, im2, F, x1, y1)
    x1_y1_s.append([x1, y1])
    x2_y2_s.append([x2, y2])

x1_y1_s = np.array(x1_y1_s)
x2_y2_s = np.array(x2_y2_s)

M1 = np.hstack((np.identity(3), np.zeros((3, 1))))
K1 = intrinsics['K1']
C1 = np.dot(K1, M1)

M2, C2, P = test_M2_solution(x1_y1_s, x2_y2_s, intrinsics)
np.savez("q3_4_2.npz", F=F, M1=M1, M2=M2, C1=C1, C2=C2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = P[:, 0]
y = P[:, 1]
z = P[:, 2]

ax.scatter(x, y, z, c='r', marker='.')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
