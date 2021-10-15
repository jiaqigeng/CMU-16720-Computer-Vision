import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.transform

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
import scipy.io


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images', img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox

        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    bboxes = sorted(bboxes, key=lambda x: x[0])
    prev_y1, prev_x1, prev_y2, prev_x2 = bboxes[0]
    rows = [[bboxes[0]]]
    for i in range(1, len(bboxes)):
        y1, x1, y2, x2 = bboxes[i]
        if y1 > prev_y2:
            rows.append([bboxes[i]])
        else:
            rows[-1].append(bboxes[i])

        prev_y1, prev_x1, prev_y2, prev_x2 = y1, x1, y2, x2

    for i in range(0, len(rows)):
        rows[i] = sorted(rows[i], key=lambda x: x[1])
    ##########################

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    crop_rows = []
    for row in rows:
        crops = []
        for bbox in row:
            y1, x1, y2, x2 = bbox
            side = max(y2-y1, x2-x1)
            center_y, center_x = (y2+y1) / 2, (x2+x1) / 2
            crop = bw[int(center_y-side/2): int(center_y+side/2), int(center_x-side/2):int(center_x+side/2)] * (-1.)
            crop = skimage.transform.resize(crop, (24, 24))
            crop = np.pad(crop, ((4, 4), (4, 4)), constant_values=crop.max())
            crop = crop.T.flatten()
            crops.append(crop)

        crops = np.array(crops)
        crops = (crops - np.mean(crops, axis=1).reshape(-1, 1)) / np.std(crops, axis=1).reshape(-1, 1)
        crop_rows.append(crops)
    ##########################

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string

    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle', 'rb'))
    ##########################
    print(img)
    for crops in crop_rows:
        h1 = forward(crops, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        output = letters[np.argmax(probs, axis=1)]
        output = ''.join(output)
        print(output)
    print()
    ##########################
