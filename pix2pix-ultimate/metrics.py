import os
from glob import glob
from utils import *


# [batch_size, width, height, dimensions(7 or 10)]

image_size = 256
input_c_dim = 3
output_c_dim = 7
sample_file = 'test-x/test_0118.png'

input_img = imread(sample_file)
print(input_img)

# sample = load_data(sample_file, image_size, input_c_dim, output_c_dim)
# print(sample.shape)
