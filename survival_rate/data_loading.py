import numpy as np
from PIL import Image


def load_tumors(file_path):
    """Given the path to a PNG file containing 3+4+3 images, extract the 3 tumor predictions, convert them to grayscale
    and put them in one matrix of size 128x128x3"""
    IMAGES_COUNT = 10
    TUMOR_IMAGES_COUNT = 3

    img = Image.open(file_path).convert('LA')
    IMAGES_WIDTH = int(img.width / IMAGES_COUNT)
    IMAGES_HEIGHT = img.height

    tumors = np.empty((IMAGES_WIDTH, IMAGES_HEIGHT, TUMOR_IMAGES_COUNT))
    img_array = np.asarray(img, dtype=np.uint8)
    for i in range(TUMOR_IMAGES_COUNT):
        tumor = img_array[:IMAGES_HEIGHT, i * IMAGES_WIDTH:(i + 1) * IMAGES_WIDTH, 0]
        tumors[:, :, i] = tumor

    return tumors
