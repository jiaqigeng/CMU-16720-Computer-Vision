import numpy as np
import multiprocessing
import threading
import queue
import os
import time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
from skimage import io
import scipy.spatial
import cv2


torch.set_num_threads(1)  # without this line, pytroch forward pass will hang with multiprocessing


def evaluate_deep_extractor(img, vgg16):
    '''
    Evaluates the deep feature extractor for a single image.

    [input]
    * image: numpy.ndarray of shape (H,W,3)
    * vgg16: prebuilt VGG-16 network.

    [output]
    * diff: difference between the two feature extractor's result
    '''

    vgg16_weights = util.get_VGG16_weights()
    img_torch = preprocess_image(img)

    feat = network_layers.extract_deep_feature(np.transpose(img_torch.numpy(), (1, 2, 0)), vgg16_weights)

    with torch.no_grad():
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())

    return np.sum(np.abs(vgg_feat_feat.numpy() - feat))


def build_recognition_system(vgg16, num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, K)
    * labels: numpy.ndarray of shape (N)
    '''

    train_data = np.load("../data/train_data.npz")

    # ----- TODO -----
    args = []
    for i in range(train_data['files'].shape[0]):
        image_path = train_data['files'][i]
        image_path = "../data/" + image_path
        args.append((i, image_path, vgg16))

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

    np.savez("trained_system_deep", features=features_all, labels=train_data['labels'])
    os.system("rm -r ./temp_files_features")


def evaluate_recognition_system(vgg16, num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system_deep.npz")

    # ----- TODO -----
    train_features = trained_system['features']
    train_labels = trained_system['labels']

    confusion_matrix = np.zeros((8, 8))

    args = []
    for i in range(test_data['files'].shape[0]):
        image_path = "../data/" + test_data['files'][i]
        args.append((i, image_path, vgg16, train_features))

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
    image_idx, image_path, vgg16, train_features = args
    feature = get_image_feature(args[:3])
    train_image_index = np.argmax(distance_to_set(feature, train_features))

    if not os.path.exists("./temp_files_indices"):
        os.makedirs("./temp_files_indices")

    np.save(os.path.join("./temp_files_indices", str(image_idx).zfill(8)), train_image_index)


def preprocess_image(image):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H, W, 3)

    [output]
    * image_processed: torch.Tensor of shape (3, H, W)
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

    image = skimage.transform.resize(image, (224, 224))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image - mean) / std

    image_processed = np.moveaxis(image, -1, 0)
    image_processed = torch.tensor(image_processed)
    return image_processed


def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.
    
    [output]
    * feat: evaluated deep feature
    '''

    i, image_path, vgg16 = args

    # ----- TODO -----
    image = io.imread(image_path)
    img_torch = preprocess_image(image)

    with torch.no_grad():
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())

    return vgg_feat_feat.numpy()


def saving_output(args):
    feat = get_image_feature(args)
    image_idx = args[0]

    if not os.path.exists("./temp_files_features"):
        os.makedirs("./temp_files_features")

    np.save(os.path.join("./temp_files_features", str(image_idx).zfill(8)), feat)


def distance_to_set(feature, train_features):
    '''
    Compute distance between a deep feature with all training image deep features.

    [input]
    * feature: numpy.ndarray of shape (K)
    * train_features: numpy.ndarray of shape (N, K)

    [output]
    * dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    return -scipy.spatial.distance.cdist(feature.reshape((1, -1)), train_features, 'euclidean')
