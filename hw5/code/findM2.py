# ##################################################################### #
# 16720B: Computer Vision Homework 5
# Carnegie Mellon University
# Oct. 26, 2020
# ##################################################################### #

import numpy as np
from submission import triangulate, eightpoint, essentialMatrix
from helper import camera2


'''
Q3.3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_5.npz
'''


def test_M2_solution(pts1, pts2, intrinsics):
	'''
	Estimate all possible M2 and return the correct M2 and 3D points P
	:param pred_pts1:
	:param pred_pts2:
	:param intrinsics:
	:return: M2, the extrinsics of camera 2
			 C2, the 3x4 camera matrix
			 P, 3D points after triangulation (Nx3)
	'''

	F = eightpoint(pts1, pts2, 640)
	K1, K2 = intrinsics['K1'], intrinsics['K2']
	E = essentialMatrix(F, K1, K2)
	M2_all = camera2(E)

	C1 = np.dot(K1, np.hstack((np.identity(3), np.zeros((3, 1)))))

	best_err = 1000000
	best_M2, best_C2, best_P = None, None, None
	for i in range(4):
		M2 = M2_all[:, :, i]
		C2 = np.dot(K2, M2)
		P, err = triangulate(C1, pts1, C2, pts2)
		if err < best_err and (P[:, -1] > 0).all():
			best_err = err
			best_M2, best_C2, best_P = M2, C2, P

	M2, C2, P = best_M2, best_C2, best_P
	return M2, C2, P


if __name__ == '__main__':
	data = np.load('../data/some_corresp.npz')
	pts1 = data['pts1']
	pts2 = data['pts2']
	intrinsics = np.load('../data/intrinsics.npz')
	M2, C2, P = test_M2_solution(pts1, pts2, intrinsics)
	np.savez('q3_3_3.npz', M2=M2, C2=C2, P=P)
