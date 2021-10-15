import numpy as np
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    image = skimage.color.rgb2gray(image)
    thresh = threshold_otsu(image)
    bw = closing(image <= thresh, square(10))
    cleared = clear_border(bw)
    label_image = label(cleared)

    for region in regionprops(label_image):
        if region.area >= 100:
            bboxes.append(region.bbox)

    ##########################
    return bboxes, bw
