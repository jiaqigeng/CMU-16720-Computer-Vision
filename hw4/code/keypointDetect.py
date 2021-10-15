import numpy as np
import cv2
import matplotlib.pyplot as plt


def createGaussianPyramid(im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4]):
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max() > 10:
        im = np.float32(im) / 255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0 * k ** i
        im_pyramid.append(cv2.GaussianBlur(im, (0, 0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(
        im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    cv2.imshow("Pyramid of image", im_pyramid)
    cv2.waitKey(0)  # press any key to exit
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1, 0, 1, 2, 3, 4]):
    """
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    """
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    h, w, num_levels = gaussian_pyramid.shape
    DoG_pyramid = np.zeros((h, w, num_levels-1))
    for i in range(1, num_levels):
        DoG_pyramid[:, :, i-1] = (gaussian_pyramid[:, :, i] - gaussian_pyramid[:, :, i-1])

    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    """
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    """
    principal_curvature = None

    gxx = []
    gyy = []
    gxy = []
    gyx = []

    for l in range(DoG_pyramid.shape[2]):
        # Computing 1st order derivatives
        gx = cv2.Sobel(
            DoG_pyramid[:, :, l],
            cv2.CV_64F,
            1,
            0,
            ksize=3,
            borderType=cv2.BORDER_CONSTANT,
        )
        gy = cv2.Sobel(
            DoG_pyramid[:, :, l],
            cv2.CV_64F,
            0,
            1,
            ksize=3,
            borderType=cv2.BORDER_CONSTANT,
        )

        # Computing 2nd order derivatives
        gxx.append(
            cv2.Sobel(gx, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
        )
        gxy.append(
            cv2.Sobel(gx, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
        )
        gyx.append(
            cv2.Sobel(gy, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
        )
        gyy.append(
            cv2.Sobel(gy, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
        )

    gxx = np.stack(gxx, axis=-1)
    gxy = np.stack(gxy, axis=-1)
    gyx = np.stack(gyx, axis=-1)
    gyy = np.stack(gyy, axis=-1)

    principal_curvature = np.divide(
        np.square(np.add(gxx, gyy)), (np.multiply(gxx, gyy) - np.multiply(gxy, gyx))
    )

    return principal_curvature


def getLocalExtrema(
    DoG_pyramid, DoG_levels, principal_curvature, th_contrast=0.03, th_r=12
):
    """
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    """
    locsDoG = None
    ##############
    #  TO DO ...
    # Compute locsDoG here
    h, w, num_level = DoG_pyramid.shape
    DoG_pyramid = np.abs(DoG_pyramid)
    padded_DoG = np.pad(DoG_pyramid, ((1,), (1,), (1,)), constant_values=0)
    padded_DoG = np.rollaxis(padded_DoG, 2, 0)
    new_num_level, new_h, new_w = padded_DoG.shape

    flat_DoG = padded_DoG.flatten()
    flat_DoG_indices = np.arange(flat_DoG.shape[0])
    flat_DoG_indices = flat_DoG_indices.reshape((new_num_level, new_h, new_w))[1:-1, 1:-1, 1:-1]
    flat_DoG_indices = flat_DoG_indices.reshape(-1, 1)

    indices = np.array([0, -new_w-1, -new_w, -new_w+1, -1, 1, new_w-1, new_w, new_w+1, -new_h*new_w, new_h*new_w])
    vectorized_indices = flat_DoG_indices + indices
    max_idx = np.argmax(flat_DoG[vectorized_indices], axis=1)
    idx = np.where(max_idx == 0)[0]
    layers, y, x = idx // (h*w), (idx % (h*w)) // w, (idx % (h*w)) % w
    locsDoG = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), layers.reshape(-1, 1)))
    locsDoG = locsDoG[np.where(DoG_pyramid[locsDoG[:, 1], locsDoG[:, 0], locsDoG[:, 2]] > th_contrast)[0]]
    locsDoG = locsDoG[np.where(np.abs(principal_curvature[locsDoG[:, 1], locsDoG[:, 0], locsDoG[:, 2]]) < th_r)[0]]
    return locsDoG


def DoGdetector(
    im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4], th_contrast=0.03, th_r=12
):
    """
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    """
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im, sigma0=sigma0, k=k, levels=levels)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1, 0, 1, 2, 3, 4]
    im = cv2.imread("../data/model_chickenbroth.jpg")
    im_pyr = createGaussianPyramid(im)
    # displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    # displayPyramid(DoG_pyr)
    # breakpoint()
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    im = plt.imread("../data/model_chickenbroth.jpg")
    plt.imshow(im.mean(axis=2), cmap='gray')
    plt.scatter(x=locsDoG[:, 0], y=locsDoG[:, 1], c='lime', s=10)
    plt.show()
