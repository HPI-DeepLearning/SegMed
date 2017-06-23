"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(image_path, image_size, input_c_dim, output_c_dim, flip=True):
    input_img = imread(image_path)
    images = np.split(input_img, input_c_dim + output_c_dim, axis=1)
    
    offset = 16
    hypersize = image_size + offset
    
    h1 = int(np.ceil(np.random.uniform(1e-2, offset)))
    w1 = int(np.ceil(np.random.uniform(1e-2, offset)))
    ran = np.random.random()
    
    conv = []
    for image in images:
        tmp = scipy.misc.imresize(image, [hypersize, hypersize], interp='nearest')
        image = tmp[h1:h1+image_size, w1:w1+image_size]
        image = image/127.5 - 1.
        if flip and ran > 0.5:
            image = np.fliplr(image)
        conv.append(image)
    
    return np.stack(conv, axis=2)

# -----------------------------

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


