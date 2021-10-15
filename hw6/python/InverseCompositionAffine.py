import numpy as np
from scipy.interpolate import RectBivariateSpline


def InverseCompositionAffine(It, It1, rect):
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

    Tx, Ty = rbs.ev(rect_yy, rect_xx, 0, 1), rbs.ev(rect_yy, rect_xx, 1, 0)
    Tx, Ty = Tx.reshape((-1, 1)), Ty.reshape((-1, 1))
    T_grad = np.hstack((Tx, Ty))

    x_flatten, y_flatten = rect_xx.reshape((-1, 1)), rect_yy.reshape((-1, 1))
    jacobian = np.zeros((x_flatten.shape[0] * 2, 6))
    jacobian[::2] = np.hstack((x_flatten, np.zeros((x_flatten.shape[0], 1)),
                               y_flatten, np.zeros((x_flatten.shape[0], 1)),
                               np.ones((x_flatten.shape[0], 1)),
                               np.zeros((x_flatten.shape[0], 1))))

    jacobian[1::2] = np.hstack((np.zeros((x_flatten.shape[0], 1)), x_flatten,
                                np.zeros((x_flatten.shape[0], 1)), y_flatten,
                                np.zeros((x_flatten.shape[0], 1)),
                                np.ones((x_flatten.shape[0], 1))))

    J = np.zeros((x_flatten.shape[0], 6))
    for i in range(x_flatten.shape[0]):
        J[i, :] = np.dot(T_grad[i, :],
                         jacobian[2 * i: 2 * i + 2, :]).reshape((1, -1))

    H = np.dot(J.T, J)

    counter = 0
    delta_p = np.array([[10000], [10000], [10000], [10000], [10000], [10000]])

    W = np.eye(3)

    while np.sqrt(np.sum(delta_p ** 2)) >= threshold and counter <= maxIters:
        rect_xx_ = W[0, 0] * rect_xx + W[0, 1] * rect_yy + W[0, 2]
        rect_yy_ = W[1, 0] * rect_xx + W[1, 1] * rect_yy + W[1, 2]

        warped = rbs1.ev(rect_yy_, rect_xx_)

        error = warped - template
        delta_p = np.dot(np.dot(np.linalg.inv(H), J.T),
                         error.reshape((-1, 1)))

        original_W = W
        W = np.array([[1.0 + delta_p[0, 0], delta_p[2, 0], delta_p[4, 0]],
                      [delta_p[1, 0], 1.0 + delta_p[3, 0], delta_p[5, 0]],
                      [0.0, 0.0, 1.0]]).reshape(3, 3)

        W = np.dot(original_W, np.linalg.inv(W))
        counter += 1

    return W[:2, :]
