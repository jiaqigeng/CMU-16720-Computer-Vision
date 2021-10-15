import numpy as np
import skimage
import multiprocessing
import threading
import queue
import os,time
import math
import visual_words
from skimage import io
import matplotlib.pyplot as plt
import scipy.signal
import random


def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K, 3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")

    # ----- TODO -----
    dict_size = dictionary.shape[0]
    SPM_layer_num = 3
    args = []
    for i in range(train_data['files'].shape[0]):
        image_path = train_data['files'][i]
        image_path = "../data/" + image_path
        args.append((i, image_path, dictionary, SPM_layer_num, dict_size))

    with multiprocessing.Pool(num_workers) as p:
        p.map(saving_output, args)

    features_all = []
    file_names_list = os.listdir("./temp_files_features")
    file_names_list.sort()

    for npy_file in file_names_list:
        if npy_file.endswith(".npy"):
            npy_path = os.path.join("./temp_files_features", npy_file)
            features = np.load(npy_path)
            features_all.append(features)

    features_all = np.array(features_all)

    np.savez("trained_system", features=features_all, labels=train_data['labels'],
             dictionary=dictionary, SPM_layer_num=SPM_layer_num)
    os.system("rm -r ./temp_files_features")


def saving_output(args):
    image_idx, image_path, dictionary, SPM_layer_num, dict_size = args
    features = get_image_feature(image_path, dictionary, SPM_layer_num, dict_size)
    if not os.path.exists("./temp_files_features"):
        os.makedirs("./temp_files_features")

    np.save(os.path.join("./temp_files_features", str(image_idx).zfill(8)), features)


def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")

    # ----- TODO -----
    train_features = trained_system['features']
    dictionary = trained_system['dictionary']
    train_labels = trained_system['labels']
    SPM_layer_num = int(trained_system['SPM_layer_num'])
    dict_size = dictionary.shape[0]

    confusion_matrix = np.zeros((8, 8))

    args = []
    for i in range(test_data['files'].shape[0]):
        image_path = "../data/" + test_data['files'][i]
        args.append((i, image_path, dictionary, SPM_layer_num, dict_size, train_features))

    with multiprocessing.Pool(num_workers) as p:
        p.map(compute_confusion_matrix, args)

    file_names_list = os.listdir("./temp_files_indices")
    file_names_list.sort()

    matched_train_indices = []
    for npy_file in file_names_list:
        if npy_file.endswith(".npy"):
            npy_path = os.path.join("./temp_files_indices", npy_file)
            train_idx = np.load(npy_path)
            matched_train_indices.append(train_idx)

    for i in range(len(matched_train_indices)):
        test_label = test_data['labels'][i]
        predicted_label = train_labels[matched_train_indices[i]]
        confusion_matrix[test_label, predicted_label] += 1

    os.system("rm -r ./temp_files_indices")
    return confusion_matrix, np.trace(confusion_matrix) / np.sum(confusion_matrix)


def compute_confusion_matrix(args):
    image_idx, image_path, dictionary, SPM_layer_num, dict_size, train_features = args
    feature = get_image_feature(image_path, dictionary, SPM_layer_num, dict_size)
    train_image_index = np.argmax(distance_to_set(feature, train_features))

    if not os.path.exists("./temp_files_indices"):
        os.makedirs("./temp_files_indices")

    np.save(os.path.join("./temp_files_indices", str(image_idx).zfill(8)), train_image_index)


def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    # ----- TODO -----
    image = io.imread(file_path)

    # random horizontal flip
    # need to add a parameter is_train to this function
    '''
    if is_train and random.random() < 0.5:
        image = np.fliplr(image)
    '''

    image = image.astype('float') / 255

    wordmap = visual_words.get_visual_words(image, dictionary)
    hist_all = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
    return hist_all


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N, K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    # ----- TODO -----
    word_hist_flat = np.array([word_hist, ] * histograms.shape[0]).flatten()
    hist_flat = histograms.flatten()
    stacked_matrix = np.vstack((word_hist_flat, hist_flat))
    sim = np.amin(stacked_matrix, axis=0).reshape(histograms.shape)
    sim = np.sum(sim, axis=1)
    return sim


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    hist, bins = np.histogram(wordmap.flatten(), bins=dict_size, range=(0, dict_size), density=True)
    # plt.bar(bins[:-1], hist, width=1)
    # plt.show()
    return hist


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    *  layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    # ----- TODO -----
    layer_num = layer_num - 1
    h, w = wordmap.shape
    new_h, new_w = int(h/(2**layer_num)) * (2**layer_num), int(w/(2**layer_num)) * (2**layer_num)
    wordmap_truncated = wordmap[:new_h, :new_w]

    row_split = np.split(wordmap_truncated, (2**layer_num), axis=0)

    sub_maps = []
    for row_sub_map in row_split:
        sub_maps.append(np.split(row_sub_map, (2**layer_num), axis=1))

    sub_maps = np.array(sub_maps)

    finest_layer = []
    for i in range(sub_maps.shape[0]):
        for j in range(sub_maps.shape[1]):
            sub_map = sub_maps[i, j]
            hist = get_feature_from_wordmap(sub_map, dict_size)
            finest_layer.append(hist)

    finest_layer = np.array(finest_layer).reshape((2**layer_num, 2**layer_num, -1))
    finest_layer_h, finest_layer_w = finest_layer.shape[0], finest_layer.shape[1]

    layers = [finest_layer.reshape((-1, ))]
    for l in range(1, layer_num + 1):
        hist_layer = []
        for i in range(0, finest_layer_h, 2**l):
            for j in range(0, finest_layer_w, 2**l):
                hist_block = np.sum(finest_layer[i:i+2**l, j:j+2**l], axis=(0, 1))
                hist_layer.append(hist_block)

        hist_layer = np.array(hist_layer).reshape((-1, ))
        layers.append(hist_layer)

    hist_all = None
    for layer_idx in range(len(layers)):
        i = len(layers)-layer_idx-1
        layer = layers[i]

        if layer_idx == 0:
            hist_all = layer * (2 ** (-layer_num))
        elif layer_idx == 1:
            hist_all = np.append(hist_all, layer * (2 ** (-layer_num)))
        else:
            hist_all = np.append(hist_all, layer * (2 ** (layer_idx-layer_num-1)))

    hist_all = hist_all * (4 ** (-layer_num))
    return hist_all
