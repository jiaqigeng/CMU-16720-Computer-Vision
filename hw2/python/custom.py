import numpy as np
import skimage
import multiprocessing
import threading
import queue
import os, time
import math
import visual_words
import visual_recog
from skimage import io
import matplotlib.pyplot as plt
import scipy.signal
import random
import util
import sklearn.cluster


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

    # Setting alpha to 250, k to 200
    alpha = 250
    k = 200
    F = 20

    args = []
    for image_path in train_data['files']:
        image_path = "../data/" + image_path
        args.append((img_idx, alpha, image_path))
        img_idx += 1

    with multiprocessing.Pool(num_workers) as p:
        p.map(visual_words.compute_dictionary_one_image, args)

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
    np.save("dictionary_tuned", dictionary)
    os.system("rm -r ./temp_files")


def custom_build_recognition_system(num_workers=2):
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

    # Please uncomment the line of code below if you want to use the retrained dicitionary
    # which might cause the model accuracy to decrease
    # See clarification in the PDF
    # dictionary = np.load("dictionary_tuned.npy")

    # ----- TODO -----
    dict_size = dictionary.shape[0]

    # setting the spatial layer number to 4
    SPM_layer_num = 4

    args = []
    for i in range(train_data['files'].shape[0]):
        image_path = train_data['files'][i]
        image_path = "../data/" + image_path
        args.append((i, image_path, dictionary, SPM_layer_num, dict_size))

    with multiprocessing.Pool(num_workers) as p:
        p.map(visual_recog.saving_output, args)

    features_all = []
    file_names_list = os.listdir("./temp_files_features")
    file_names_list.sort()

    for npy_file in file_names_list:
        if npy_file.endswith(".npy"):
            npy_path = os.path.join("./temp_files_features", npy_file)
            features = np.load(npy_path)
            features_all.append(features)

    features_all = np.array(features_all)

    np.savez("trained_system_tuned", features=features_all, labels=train_data['labels'],
             dictionary=dictionary, SPM_layer_num=SPM_layer_num)
    os.system("rm -r ./temp_files_features")


def custom_evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system_tuned.npz")

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
        p.map(visual_recog.compute_confusion_matrix, args)

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


def main():
    num_cores = util.get_num_CPU()

    # Please uncomment the line of code below if you want to use the retrained dicitionary
    # which might cause the model accuracy to decrease
    # See clarification in the PDF
    # compute_dictionary(num_workers=num_cores)

    custom_build_recognition_system(num_workers=num_cores)
    conf, accuracy = custom_evaluate_recognition_system(num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())


if __name__ == '__main__':
    main()
