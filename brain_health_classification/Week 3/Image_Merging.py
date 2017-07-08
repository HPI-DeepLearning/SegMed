
# coding: utf-8

# Pix2Pix expects merged JPEG training images that consist of the 256x256 output image and the 256x256 input image (you can find an example below). In this notebook we will create these training images that consist of the brain scans and associated tumor segmentations.

# ![](http://i.imgur.com/afSM6qP.jpg)

# In[1]:

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

WIDTH, HEIGHT = 240, 240
DATA_DIR = os.path.join("..", "data")

# Ground truth
HGG_GT_TXT = '/mnt/naruto/Seminar-Med-Seg2017/BRATS2017/list/HGG_GT.txt'
LGG_GT_TXT = '/mnt/naruto/Seminar-Med-Seg2017/BRATS2017/list/LGG-GT.txt'

# 
HGG_IMG_TXT = '/mnt/naruto/Seminar-Med-Seg2017/BRATS2017/list/HGG-img.txt'
LGG_IMG_TXT = '/mnt/naruto/Seminar-Med-Seg2017/BRATS2017/list/LGG-img.txt'


# In[2]:

def get_lines(path):
    csv_file = os.path.abspath(path)
    with open(csv_file) as f:
        data_file = f.read().splitlines()
    return [x for x in data_file if len(x) > 0]


# In[ ]:

lgg_set = list(get_lines(LGG_IMG_TXT))
hgg_set = list(get_lines(HGG_IMG_TXT))
lgg_gt_set = list(get_lines(LGG_GT_TXT))
hgg_gt_set = list(get_lines(HGG_GT_TXT))

print('{} lgg images'.format(len(lgg_set)))
print('{} hgg images'.format(len(hgg_set)))
print('{} lgg gt images'.format(len(lgg_gt_set)))
print('{} hgg gt images'.format(len(hgg_gt_set)))


# In[ ]:

print("cleaning ...")
lgg_set_cleaned = [f for f in lgg_set if f.replace('_flair', '_seg') in lgg_gt_set]
hgg_set_cleaned = [f for f in hgg_set if f.replace('_flair', '_seg') in hgg_gt_set]
print("finished cleaning")


# In[ ]:

def merge_horizontally(path):
    images = map(Image.open, [path.replace('_flair', '_seg'), path])
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(path.replace('_flair', '_combined'))


# In[ ]:

for path in tqdm(lgg_set_cleaned):
    merge_horizontally(path)
for path in tqdm(hgg_set_cleaned):
    merge_horizontally(path)

