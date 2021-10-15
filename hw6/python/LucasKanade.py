import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   p: movement vector dx, dy

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)
    x1, y1, x2, y2 = rect
    h, w = It.shape

    rbs = RectBivariateSpline(np.arange(h), np.arange(w), It)
    rbs1 = RectBivariateSpline(np.arange(h), np.arange(w), It1)

    rect_xs = np.linspace(x1, x2, int(x2-x1))
    rect_ys = np.linspace(y1, y2, int(y2-y1))
    rect_xx, rect_yy = np.meshgrid(rect_xs, rect_ys)
    template = rbs.ev(rect_yy, rect_xx)

    delta_p = np.array([[10000], [10000]])
    counter = 0

    while np.sqrt(np.sum(delta_p ** 2)) >= threshold and \
            counter <= maxIters:

        rect_xx_, rect_yy_ = rect_xx + p[0], rect_yy + p[1]
        warped = rbs1.ev(rect_yy_, rect_xx_)
        error = template - warped

        Ix, Iy = rbs1.ev(rect_yy_, rect_xx_, 0, 1), \
                 rbs1.ev(rect_yy_, rect_xx_, 1, 0)
        Ix, Iy = Ix.reshape((-1, 1)), Iy.reshape((-1, 1))
        I = np.hstack((Ix, Iy))

        jacobian = np.eye(2)
        A = np.dot(I, jacobian)
        H = np.dot(A.T, A)

        delta_p = np.dot(np.dot(np.linalg.inv(H), A.T),
                         error.reshape((-1, 1)))
        p[0] += delta_p[0, 0]
        p[1] += delta_p[1, 0]
        counter += 1

    return p
