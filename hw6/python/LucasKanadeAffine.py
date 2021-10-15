import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanadeAffine(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros((6, 1))
    x1, y1, x2, y2 = rect
    h, w = It.shape

    # put your implementation here
    rbs = RectBivariateSpline(np.arange(h), np.arange(w), It)
    rbs1 = RectBivariateSpline(np.arange(h), np.arange(w), It1)

    rect_xs = np.linspace(x1, x2, int(x2-x1))
    rect_ys = np.linspace(y1, y2, int(y2-y1))
    rect_xx, rect_yy = np.meshgrid(rect_xs, rect_ys)

    template = rbs.ev(rect_yy, rect_xx)

    delta_p = np.array([[10000], [10000], [10000], [10000], [10000], [10000]])
    counter = 0

    while np.sqrt(np.sum(delta_p ** 2)) >= threshold and counter <= maxIters:
        rect_xx_ = (1 + p[0, 0]) * rect_xx + p[1, 0] * rect_yy + p[2, 0]
        rect_yy_ = p[3, 0] * rect_xx + (1 + p[4, 0]) * rect_yy + p[5, 0]

        warped = rbs1.ev(rect_yy_, rect_xx_)
        error = template - warped

        x_flatten, y_flatten = rect_xx_.reshape((-1, 1)), \
                               rect_yy_.reshape((-1, 1))

        jacobian = np.zeros((x_flatten.shape[0] * 2, 6))
        jacobian[::2] = np.hstack((x_flatten,
                                   np.zeros((x_flatten.shape[0], 1)),
                                   y_flatten,
                                   np.zeros((x_flatten.shape[0], 1)),
                                   np.ones((x_flatten.shape[0], 1)),
                                   np.zeros((x_flatten.shape[0], 1))))

        jacobian[1::2] = np.hstack((np.zeros((x_flatten.shape[0], 1)),
                                    x_flatten,
                                    np.zeros((x_flatten.shape[0], 1)),
                                    y_flatten,
                                    np.zeros((x_flatten.shape[0], 1)),
                                    np.ones((x_flatten.shape[0], 1))))

        Ix, Iy = rbs1.ev(rect_yy_, rect_xx_, 0, 1), \
                 rbs1.ev(rect_yy_, rect_xx_, 1, 0)
        Ix, Iy = Ix.reshape((-1, 1)), Iy.reshape((-1, 1))
        I = np.hstack((Ix, Iy))

        A = np.zeros((x_flatten.shape[0], 6))
        for i in range(x_flatten.shape[0]):
            A[i, :] = np.dot(I[i, :],
                             jacobian[2*i: 2*i+2, :]).reshape((1, -1))

        H = np.dot(A.T, A)
        delta_p = np.dot(np.dot(np.linalg.inv(H), A.T),
                         error.reshape((-1, 1)))

        p[0] += delta_p[0, 0]
        p[1] += delta_p[2, 0]
        p[2] += delta_p[4, 0]
        p[3] += delta_p[1, 0]
        p[4] += delta_p[3, 0]
        p[5] += delta_p[5, 0]
        counter += 1

    M = np.array([[1.0 + p[0], p[1], p[2]],
                  [p[3], 1.0 + p[4], p[5]]]).reshape(2, 3)

    return M
