from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.layers.recurrent import LSTM
#from keras.datasets import brain
import pdb
import numpy as np
from numpy import *
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import cv2
import argparse
import math
import csv
import random
import os
from sklearn.cross_validation import train_test_split
from skimage import data


def classification_model():
	model = Sequential()
	model.add(Embedding(max_features, 256))
	model.add(LSTM(256, 128, activation='sigmoid', inner_activation='hard_sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(128, 1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='rmsprop')

	model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
	score = model.evaluate(X_test, Y_test, batch_size=16)



def load_imgs(path, grayscale=False, target_size=None):
    image = Image.open(path)
    print "Creating numpy representation of image %s " % file
    resize = image.resize((250,250), Image.NEAREST) 
    resize.load()
    data = np.asarray( resize, dtype="uint8")
    print(data.shape)
    master_dataset.append(data)
    train_set = master_dataset, np.asarray(master_labels)
    valid_set = valid_data, np.asarray(valid_labels)
    test_set = test_data, np.asarray(test_labels) 
    dataset = [train_set, valid_set, test_set]

    print("Creating pickle file")
    f = gzip.open('brain.pkl.gz', 'wb')
    cPickle.dump(dataset, f, protocol=2)
    f.close()
    return f

def getData(X_path, y_path):
    pdb.set_trace()
    while 1: 
        with open(X_path, "rb") as csv1:
            reader1 = csv.reader(csv1, delimiter=',')
        with open(y_path, "rb") as csv2:
            reader2 = csv.reader(csv2, delimiter=',')
            for row in zip(reader1, reader2):
                yield (np.array(row[0], dtype=np.float), np.array(row[1], dtype=np.float))
                csv1.close()
                csv2.close()


def train(BATCH_SIZE):
    print 'here'
    pdb.set_trace()
    f = open('/data/Workspace/keras/data/train.txt')
    for line in f:
            x, y = process_line(line)
            image = load_images(x)
            yield (img, y)
            pdb.set_trace()
    #f.close()
    
    print 'Creating numpy representation of image'
    resize = image.resize((300,300), Image.NEAREST) 
    resize.load()
    data = np.asarray( resize, astype="uint8" )
    print(data.shape)
    master_dataset.append(data)  
    train_set = master_dataset, np.asarray(master_labels)
    valid_set = valid_data, np.asarray(valid_labels)
    test_set = test_data, np.asarray(test_labels)
    dataset = [train_set, valid_set, test_set]
    print("Creating pickle file")
    f = gzip.open('MRbrain.pkl.gz', 'wb')
    cPickle.dump(dataset, f, protocol=2)
    f.close()
    data = cPickle.load(f)
    train = data[0]
    valid = data[1]
    test = data[2]

    train_x, train_y = data[0]
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    classification = classification_model()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
