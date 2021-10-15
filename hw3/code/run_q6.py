import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA
##########################
##### your code here #####
##########################

# rebuild a low-rank version
lrank = None
##########################
##### your code here #####
##########################

# rebuild it
recon = None
##########################
##### your code here #####
##########################

# build valid dataset
recon_valid = None
##########################
##### your code here #####
##########################

# visualize the comparison and compute PSNR
##########################
##### your code here #####
##########################
