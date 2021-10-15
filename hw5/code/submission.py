# ##################################################################### #
# 16720B: Computer Vision Homework 5
# Carnegie Mellon University
# Oct. 26, 2020
# ##################################################################### #


# Insert your package here
from skimage.color import rgb2xyz
from scipy.sparse import kron as spkron
from scipy.sparse import eye as speye
from scipy.sparse.linalg import lsqr as splsqr
import pdb
from utils import integrateFrankot
import numpy as np
from helper import refineF
import math
import scipy.optimize
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from utils import lRGB2XYZ
import os


'''
Q3.2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''


def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1_normed, pts2_normed = pts1 / M, pts2 / M

    m = pts1.shape[0]

    A = np.hstack(
        ((pts1_normed[:, 0] * pts2_normed[:, 0]).reshape((m, 1)),
         (pts1_normed[:, 0] * pts2_normed[:, 1]).reshape((m, 1)), pts1_normed[:, 0].reshape((m, 1)),
         (pts1_normed[:, 1] * pts2_normed[:, 0]).reshape((m, 1)),
         (pts1_normed[:, 1] * pts2_normed[:, 1]).reshape((m, 1)), pts1_normed[:, 1].reshape((m, 1)),
         pts2_normed[:, 0].reshape((m, 1)), pts2_normed[:, 1].reshape((m, 1)), np.ones((m, 1))
         ))

    eigen_vals, eigen_vecs = np.linalg.eigh(np.dot(A.T, A))
    F = eigen_vecs[:, 0].reshape((3, 3)).T
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(np.dot(U, np.diag(S)), V)
    # F = F / F[2, 2]
    F = refineF(F, pts1_normed, pts2_normed)
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    return np.dot(np.dot(T.T, F), T)


'''
Q3.2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''


def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pass


'''
Q3.3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''


def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    return np.dot(np.dot(K2.T, F), K1)


'''
Q3.3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''


def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    X = np.zeros((pts1.shape[0], 4))
    for i in range(pts1.shape[0]):
        A = np.zeros((4, 4))
        A[0, :] = pts1[i, 1] * C1[2, :] - C1[1, :]
        A[1, :] = C1[0, :] - pts1[i, 0] * C1[2:, ]
        A[2, :] = pts2[i, 1] * C2[2, :] - C2[1, :]
        A[3, :] = C2[0, :] - pts2[i, 0] * C2[2:, ]

        eigen_vals, eigen_vecs = np.linalg.eigh(np.dot(A.T, A))
        X[i, :] = eigen_vecs[:, 0]

    P = np.zeros((pts1.shape[0], 3))

    P[:, 0] = X[:, 0] / X[:, 3]
    P[:, 1] = X[:, 1] / X[:, 3]
    P[:, 2] = X[:, 2] / X[:, 3]

    proj1 = np.dot(C1, np.vstack((P.T, np.ones((1, pts1.shape[0])))))
    proj2 = np.dot(C2, np.vstack((P.T, np.ones((1, pts1.shape[0])))))

    proj1[0, :] = proj1[0, :] / proj1[2, :]
    proj1[1, :] = proj1[1, :] / proj1[2, :]
    proj2[0, :] = proj2[0, :] / proj2[2, :]
    proj2[1, :] = proj2[1, :] / proj2[2, :]

    proj1, proj2 = proj1[:2, :].T, proj2[:2, :].T

    err = np.sum((pts1 - proj1) ** 2 + (pts2 - proj2) ** 2)
    return P, err


'''
Q3.4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''


def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    gaussian_filter = 1/16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    # x = np.array([x1, y1, 1]).reshape((-1, 1))
    x = np.array([x1, y1, 1])
    # a, b, c = np.dot(F, x)
    l = np.dot(F, x)
    s = np.sqrt(l[0] ** 2 + l[1] ** 2)
    l = l / s
    a, b, c = l
    # a, b, c = a.item(), b.item(), c.item()

    y2_all = np.arange(im2.shape[0])
    x2_all = (-c - b * y2_all) / a

    x2_on_line = x2_all[np.where((x2_all < im2.shape[1]) & (x2_all >= 0))].astype(np.int32)
    y2_on_line = y2_all[np.where((x2_all < im2.shape[1]) & (x2_all >= 0))]

    pts_on_line = np.hstack((x2_on_line.reshape((-1, 1)), (y2_on_line.reshape((-1, 1)))))

    best_dist = 1000000
    best_pair = None, None

    im2_pad = np.pad(im2, ((1, 1), (1, 1), (0, 0)))
    window1 = im1[y1-1:y1+2, x1-1:x1+2, :]

    for pt in pts_on_line:
        x2, y2 = pt
        x2, y2 = x2+1, y2+1

        window2 = im2_pad[y2-1:y2+2, x2-1:x2+2, :]
        dist = np.sum(np.sum((window1 - window2) ** 2, axis=2) * gaussian_filter)

        if dist < best_dist:
            best_dist = dist
            best_pair = x2, y2

    return best_pair


'''
Q3.5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''


def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    n = pts1.shape[0]

    best_F = None
    best_inliers = None
    best_num_inliers = 0

    for i in range(200):
        random_idx = np.random.choice(n, size=8, replace=False)
        pts1_selected, pts2_selected = pts1[random_idx], pts2[random_idx]
        F = eightpoint(pts1_selected, pts2_selected, M)

        x1s = np.hstack((pts1, np.ones((n, 1)))).T
        x2s = np.hstack((pts2, np.ones((n, 1)))).T

        l = np.dot(x2s.T, F)
        l = l / np.sqrt(np.sum(l[:, :2] ** 2, axis=1)).reshape((-1, 1))

        # result = np.diag(np.dot(np.dot(x2s.T, F), x1s))
        result = np.diag(np.dot(l, x1s))
        inliers_index = np.where(np.abs(result) < 1)[0]

        inliers = np.full((n, 1), False)
        inliers[inliers_index] = True
        num_inliers = inliers_index.shape[0]

        if num_inliers > best_num_inliers:
            best_F = F
            best_inliers = inliers
            best_num_inliers = num_inliers

    print(best_num_inliers)

    return best_F, best_inliers


'''
Q3.5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''


def rodrigues(r):
    # Replace pass by your implementation
    theta = np.sqrt(np.sum(r ** 2))
    I = np.identity(3)
    if theta == 0:
        return I
    else:
        u = r / theta
        ux = np.array([[0, -u[2, 0], u[1, 0]], [u[2, 0], 0, -u[0, 0]], [-u[1, 0], u[0, 0], 0]])
        R = I * np.cos(theta) + (1 - np.cos(theta)) * np.dot(u, u.T) + ux * np.sin(theta)
        return R


'''
Q3.5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''


def arctan_2(y, x):
    assert(x != 0 or y != 0)

    if x > 0:
        return np.arctan(y / x)
    elif x < 0:
        return np.pi + np.arctan(y / x)
    elif x == 0 and y > 0:
        return np.pi / 2
    else:
        return -np.pi / 2


def invRodrigues(R):
    # Replace pass by your implementation
    A = (R - R.T) / 2
    I = np.identity(3)

    p = np.array([A[2, 1], A[0, 2], A[1, 0]])
    s = np.sqrt(np.sum(p ** 2))
    c = (np.sum(np.diag(R)) - 1) / 2

    if s == 0 and c == 1:
        return np.zeros((3, 1))
    elif s == 0 and c == -1:
        v = R + I
        for i in range(v.shape[1]):
            if v[:, i] != np.zeros((3, )):
                v = v[:, i]
                break

        u = v / np.sqrt(np.sum(v ** 2))
        u_pi = np.pi * u
        if np.sqrt(np.sum(u_pi ** 2)) == np.pi and ((u_pi[0] == 0 and u_pi[1] == 0 and u_pi[2] < 0) or
                                                    (u_pi[0] == 0 and u_pi[1] < 0) or
                                                    (u_pi[0] < 0)):
            return -u_pi.reshape((3, 1))
        else:
            return u_pi.reshape((3, 1))
    else:
        u = p.T / s
        theta = arctan_2(s, c)
        return np.dot(u, theta).reshape((3, 1))


'''
Q3.5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original 
            and estimated projections
'''


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    P = x[:-6].reshape((-1, 3))
    P = np.hstack((P, np.ones((P.shape[0], 1)))).T

    r2, t2 = x[-6:-3].reshape((3, 1)), x[-3:].reshape((3, 1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    C1 = np.dot(K1, M1)
    C2 = np.dot(K2, M2)

    p1_hat = np.dot(C1, P)
    p1_hat = (p1_hat[:2, :] / p1_hat[2, :]).T
    p2_hat = np.dot(C2, P)
    p2_hat = (p2_hat[:2, :] / p2_hat[2, :]).T
    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]), (p2 - p2_hat).reshape([-1])])
    return residuals.reshape((-1, 1))


'''
Q3.5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    R2 = M2_init[:, :3]
    r2 = invRodrigues(R2).reshape((3, ))
    t2 = M2_init[:, -1].reshape((3, ))
    x_init = np.concatenate([P_init.reshape([-1]), r2, t2])
    objective = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x).reshape((-1,))
    x_star = scipy.optimize.leastsq(objective, x_init)[0]

    P2 = x_star[:-6].reshape((-1, 3))
    r2, t2 = x_star[-6:-3].reshape((3, 1)), x_star[-3:].reshape((3, 1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))
    return M2, P2


def renderNDotLSphere(center, rad, light, pxSize, res):
    """
    Q4.1

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    # Replace pass by your implementation
    h, w = res[0] * pxSize, res[1] * pxSize
    h_vals = np.arange(-int(h/2), int(h/2))
    w_vals = np.arange(-int(w/2), int(w/2))

    xx, yy = np.meshgrid(w_vals, h_vals)
    mask = np.sqrt(xx ** 2 + yy ** 2) < rad
    normals = np.zeros((h, w, 3))
    normals[:, :, 0][mask] = 2 * xx[mask]
    normals[:, :, 1][mask] = 2 * yy[mask]
    normals[:, :, 2][mask] = 2 * np.sqrt(rad * rad - xx[mask] ** 2 + yy[mask] ** 2)

    normals_flat = normals.reshape((h*w, 3))

    result = np.dot(normals_flat, light).reshape((h, w))
    result[result < 0] = 0
    plt.imshow(result, cmap='gray')
    plt.show()
    return result


def loadData(path="../data/"):
    """
    Q4.2.1

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    # Replace pass by your implementation
    I = []
    s = None
    for file in os.listdir(path):
        if file.endswith(".tif"):
            image = cv2.imread(os.path.join(path, file))
            s = image.shape[:2]
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_XYZ = lRGB2XYZ(image_RGB)
            Y_channel = (image_XYZ[:, :, 1]).flatten()
            I.append(Y_channel)

    I = np.array(I)
    L = np.load("../data/sources.npy").T
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):
    """
    Q4.2.2

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    # Replace pass by your implementation
    B = np.linalg.lstsq(L.T, I, rcond=None)[0]
    return B


def estimateAlbedosNormals(B):
    '''
    Q4.2.3

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    # Replace pass by your implementation
    albedos = np.sqrt(np.sum(B ** 2, axis=0))
    normals = B / albedos.reshape((1, -1))
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):
    """
    Q4.2.4

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    # Replace pass by your implementation
    plt.imshow(albedos.reshape(s), cmap="gray")
    plt.show()
    normals_ = (normals.T + 1) / 2.
    plt.imshow(normals_.reshape((s[0], s[1], 3)), cmap="rainbow")
    plt.show()
    return albedos.reshape(s), normals_.reshape((s[0], s[1], 3))


def estimateShape(normals, s):
    """
    Q4.3.1

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    # Replace pass by your implementation
    pass


def plotSurface(surface):
    """
    Q4.3.1

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    # Replace pass by your implementation
    pass
