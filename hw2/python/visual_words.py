import numpy as np
import multiprocessing
import scipy.ndimage
import skimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os
import time
import matplotlib.pyplot as plt
import util
import cv2
import scipy.signal
from skimage import io
from tempfile import TemporaryFile


def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''

    # ----- TODO -----
    image = np.asarray(image, dtype=np.float32)
    if len(image.shape) < 3:
        image = cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
        image = np.dstack((image, image, image))
    else:
        image[:, :, 0] = cv2.normalize(image[:, :, 0], None, 0, 1, cv2.NORM_MINMAX)
        image[:, :, 1] = cv2.normalize(image[:, :, 1], None, 0, 1, cv2.NORM_MINMAX)
        image[:, :, 2] = cv2.normalize(image[:, :, 2], None, 0, 1, cv2.NORM_MINMAX)

    image = skimage.color.rgb2lab(image)

    scales = [1., 2., 4., 8., 8. * np.sqrt(2)]
    responses = None

    for s in scales:
        if responses is None:
            responses = scipy.ndimage.gaussian_filter(image, sigma=(s, s, 0))
        else:
            responses = np.concatenate((responses, scipy.ndimage.gaussian_filter(image, sigma=(s, s, 0))), axis=2)

        c1 = scipy.ndimage.gaussian_laplace(image[:, :, 0], sigma=s)
        c2 = scipy.ndimage.gaussian_laplace(image[:, :, 1], sigma=s)
        c3 = scipy.ndimage.gaussian_laplace(image[:, :, 2], sigma=s)
        log = np.dstack((c1, c2, c3))
        responses = np.concatenate((responses, log), axis=2)

        responses = np.concatenate((responses,
                                    scipy.ndimage.gaussian_filter(image, sigma=(s, s, 0), order=(0, 1, 0))), axis=2)
        responses = np.concatenate((responses,
                                    scipy.ndimage.gaussian_filter(image, sigma=(s, s, 0), order=(1, 0, 0))), axis=2)

    return responses


def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    '''

    # ----- TODO -----
    filter_responses = extract_filter_responses(image)
    filter_responses = np.reshape(filter_responses, (image.shape[0] * image.shape[1], -1))
    wordmap = np.argmin(scipy.spatial.distance.cdist(filter_responses, dictionary), axis=1).reshape((image.shape[0],
                                                                                                     image.shape[1]))
    return wordmap


def get_harris_points(image, alpha, k=0.05):
    '''
    Compute points of interest using the Harris corner detector

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)
    * alpha: number of points of interest desired
    * k: senstivity factor 

    [output]
    * points_of_interest: numpy.ndarray of shape (alpha, 2) that contains interest points
    '''

    # ----- TODO -----
    image_gray = skimage.color.rgb2gray(image)
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = sobel_x * sobel_x
    Iyy = sobel_y * sobel_y
    Ixy = sobel_x * sobel_y

    Sxx = scipy.signal.convolve2d(Ixx, np.ones((3, 3)), boundary='fill', mode='same')
    Syy = scipy.signal.convolve2d(Iyy, np.ones((3, 3)), boundary='fill', mode='same')
    Sxy = scipy.signal.convolve2d(Ixy, np.ones((3, 3)), boundary='fill', mode='same')

    R = (Sxx * Syy - Sxy * Sxy) - k * (Sxx + Syy) ** 2

    x_indices, y_indices = np.unravel_index(np.argsort(R.ravel()), R.shape)
    x_indices, y_indices = x_indices[-alpha:], y_indices[-alpha:]

    poi = np.zeros((alpha, 2))
    poi[:, 0] = x_indices
    poi[:, 1] = y_indices
    return poi


def compute_dictionary_one_image(args):
    '''
    Extracts alpha samples of the dictionary entries from an image. Use the 
    harris corner detector implmented from previous question to extract 
    the point of interests. This should be a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''

    i, alpha, image_path = args
    # ----- TODO -----
    image = io.imread(image_path)

    filter_responses = extract_filter_responses(image)
    poi = get_harris_points(image, alpha=alpha)
    poi = poi.astype(np.int)
    if not os.path.exists("./temp_files"):
        os.makedirs("./temp_files")

    np.save(os.path.join("./temp_files", str(i).zfill(8)), filter_responses[poi[:, 0], poi[:, 1], :])


def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''

    train_data = np.load("../data/train_data.npz")
    # ----- TODO -----
    img_idx = 0
    alpha = 250
    k = 200
    F = 20

    args = []
    for image_path in train_data['files']:
        image_path = "../data/" + image_path
        args.append((img_idx, alpha, image_path))
        img_idx += 1

    with multiprocessing.Pool(num_workers) as p:
        p.map(compute_dictionary_one_image, args)

    filter_responses_all = []
    for npy_file in os.listdir("./temp_files"):
        if npy_file.endswith(".npy"):
            npy_path = os.path.join("./temp_files", npy_file)
            response = np.load(npy_path)
            filter_responses_all.append(response)

    filter_responses_all = np.stack(filter_responses_all, axis=1)
    filter_responses_all = filter_responses_all.reshape((alpha * img_idx), 3*F)
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(filter_responses_all)
    dictionary = kmeans.cluster_centers_
    np.save("dictionary", dictionary)
    os.system("rm -r ./temp_files")
