"""
Driver main function file
"""


import matplotlib.pyplot as plt
import numpy as np
import submission as sub
from helper import displayEpipolarF, epipolarMatchGUI
from helper import camera2
from findM2 import test_M2_solution


data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

N = data['pts1'].shape[0]
M = 640

# 3.2.1
F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
np.savez("q3_2_1.npz", F=F8, m=M)
assert F8.shape == (3, 3), 'eightpoint returns 3x3 matrix'
# displayEpipolarF(im1, im2, F8)


# 3.3.1
intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']
E = sub.essentialMatrix(F8, K1, K2)
# print(E)

# 3.4.1
x2, y2 = sub.epipolarCorrespondence(im1, im2, F8, data['pts1'][0, 0], data['pts1'][0, 1])
assert np.isscalar(x2) & np.isscalar(y2), 'epipolarCoorespondence returns x & y coordinates'
# epipolarMatchGUI(im1, im2, F8)

# 3.5.1
data_noisy = np.load('../data/some_corresp_noisy.npz')
F, inliers = sub.ransacF(data_noisy['pts1'], data_noisy['pts2'], M)
M2, C2, P = test_M2_solution(data_noisy['pts1'][inliers[:, 0]], data_noisy['pts2'][inliers[:, 0]], intrinsics)
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
assert F.shape == (3, 3), 'ransacF returns 3x3 matrix'


# 3.5.3
data = np.load('../data/some_corresp_noisy.npz')
pts1 = data['pts1']
pts2 = data['pts2']
intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']
M1 = np.hstack((np.identity(3), np.zeros((3, 1))))
C1 = np.dot(K1, M1)

F, inliers = sub.ransacF(pts1, pts2, 640)
M2_init, C2_init, P_init = test_M2_solution(pts1[inliers[:, 0]], pts2[inliers[:, 0]], intrinsics)
M2, P2 = sub.bundleAdjustment(K1, M1, pts1[inliers[:, 0]], K2, M2_init, pts2[inliers[:, 0]], P_init)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = P2[:, 0]
y = P2[:, 1]
z = P2[:, 2]

_, err1 = sub.triangulate(C1, pts1[inliers[:, 0]], C2_init, pts2[inliers[:, 0]])
_, err2 = sub.triangulate(C1, pts1[inliers[:, 0]], np.dot(K2, M2), pts2[inliers[:, 0]])
print(err1, err2)

ax.scatter(x, y, z, c='r', marker='.')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# 4.1
sub.renderNDotLSphere((0, 0, 0), 5000, np.array([1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]), 5, (3000, 2500))
sub.renderNDotLSphere((0, 0, 0), 5000, np.array([1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)]), 5, (3000, 2500))
sub.renderNDotLSphere((0, 0, 0), 5000, np.array([-1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)]), 5, (3000, 2500))

# 4.2
I, L, s = sub.loadData("../data/")
B = sub.estimatePseudonormalsCalibrated(I, L)
albedos, normals = sub.estimateAlbedosNormals(B)
sub.displayAlbedosNormals(albedos, normals, s)
