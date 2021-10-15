import numpy as np
import scipy.ndimage
import os
import skimage.transform


def extract_deep_feature(x, vgg16_weights):
    '''
    Extracts deep features from the given VGG-16 weights.

    [input]
    * x: numpy.ndarray of shape (H, W, 3)
    * vgg16_weights: list of shape (L, 3)

    [output]
    * feat: numpy.ndarray of shape (K)
    '''

    feat = x

    layer_counter = 0
    for i in range(len(vgg16_weights)):
        layer_type = vgg16_weights[i][0]

        if layer_type == 'conv2d':
            w, b = vgg16_weights[i][1], vgg16_weights[i][2]
            feat = multichannel_conv2d(feat, weight=w, bias=b)
        elif layer_type == 'relu':
            feat = relu(feat)
        elif layer_type == 'maxpool2d':
            feat = max_pool2d(feat, vgg16_weights[i][1])
        else:
            layer_counter += 1
            if layer_counter == 1:
                feat = np.moveaxis(feat, -1, 0).flatten()

            w, b = vgg16_weights[i][1], vgg16_weights[i][2]
            feat = linear(feat, w, b)
            if layer_counter == 2:
                break

    return feat


def multichannel_conv2d(x, weight, bias):
    '''
    Performs multi-channel 2D convolution.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * weight: numpy.ndarray of shape (output_dim, input_dim, kernel_size, kernel_size)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * feat: numpy.ndarray of shape (H, W, output_dim)
    '''

    output_dim = weight.shape[0]
    input_dim = x.shape[2]

    feat = np.zeros((x.shape[0], x.shape[1], output_dim))

    for i in range(output_dim):
        for j in range(input_dim):
            feat[:, :, i] += scipy.ndimage.convolve(x[:, :, j], np.flip(weight[i, j, :]), mode='constant')

    feat += bias
    return feat


def relu(x):
    '''
    Rectified linear unit.

    [input]
    * x: numpy.ndarray

    [output]
    * y: numpy.ndarray
    '''

    y = np.copy(x)
    y[y < 0] = 0
    return y


def max_pool2d(x, size):
    '''
    2D max pooling operation.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * size: pooling receptive field

    [output]
    * y: numpy.ndarray of shape (H/size, W/size, input_dim)
    '''

    h, w, _ = x.shape

    new_h, new_w = int(h / size) * size, int(w / size) * size
    x_truncated = x[:new_h, :new_w, :]
    x_reshaped = np.reshape(x_truncated, (int(h / size), size, int(w / size), size, -1))
    y = np.amax(x_reshaped, axis=(1, 3))
    return y


def linear(x, W, b):
    '''
    Fully-connected layer.

    [input]
    * x: numpy.ndarray of shape (input_dim)
    * weight: numpy.ndarray of shape (output_dim,input_dim)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * y: numpy.ndarray of shape (output_dim)
    '''

    return np.dot(W, x) + b
