import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import utils
from tensorflow.python.client import device_lib
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD, Adadelta, Nadam
from matplotlib import pyplot as plt
device_lib.list_local_devices()
WIDTH, HEIGHT = 240, 240
SOURCE_FILE = os.path.join('..', 'data', 'week3', 'week3_just_tumor_{}.csv')
SOURCE_FILES = [SOURCE_FILE.format(i) for i in range(25)]
chunks = []
for i, file_ in enumerate(SOURCE_FILES):
    print('Processing chunk file #{}'.format(i))
    # chunksize=100 -> returns TextFileReader for iteration
    print('from', i*999, 'to', i*999 + 998)
    chunk_df = pd.read_csv(file_, dtype=np.uint8, nrows=999)
    chunks.append(chunk_df)
data = pd.concat(chunks)
chunks = None
print("Preprocess data")

def preprocess_dataset(dataset):
    # Pop labels and transform them to vectors
    y = dataset.pop('label')
    y = y.values.reshape((-1, 1))
    # Reshape the features for CNN
    # X = dataset.as_matrix().reshape(dataset.shape[0], 1, WIDTH, HEIGHT).astype(np.float32)
    X = dataset.as_matrix().reshape(dataset.shape[0], 1, WIDTH, HEIGHT).astype(np.float32)
    # Norm datax
    X /= 255
    # Convert labels to categorical values
    y = keras.utils.to_categorical(y, 2)
    return X, y

def get_shuffled_splitted_data():
    # Shuffle and split data into: 70% train, 20% test, 10% validation
    train, test, val = np.split(data.sample(frac=1), [int(.7*len(data)), int(.85*len(data))])    
    # Extract labels, normalize, preprocess for keras
    X_train, y_train = preprocess_dataset(train)
    X_test, y_test = preprocess_dataset(test)
    X_val, y_val = preprocess_dataset(val)
    return X_train, y_train, X_test, y_test, X_val, y_val

def setup_simple_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(7, 7), input_shape=(1, WIDTH, HEIGHT)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(1, WIDTH, HEIGHT)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(7, 7)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Nadam(),
                  metrics=['accuracy'])
    return model

# Collect new dataset containing sagittal images including scull and tumor
X_train, y_train, X_test, y_test, X_val, y_val = get_shuffled_splitted_data()
print("Start training")
def setup_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(11, 11), padding='same', input_shape=(1, WIDTH, HEIGHT)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(7, 7)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))

    
    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

with tf.device('/gpu:0'):
    K.set_image_dim_ordering('th')
    batch_size = 36
    num_classes = 2
    epochs = 20
    # For storing the validation loss values
    history = keras.callbacks.History()
    # Train model
    model = setup_simple_model()
    model_results = model.fit(X_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=2,
                              validation_data=(X_test, y_test),
                              callbacks=[history])
    # Evaluate model on validation set
    print('\nValidate model on {} unknown validation samples:'.format(X_val.shape[0]))
    val_score = model.evaluate(X_val, y_val, verbose=0)
    print('Val loss:', val_score[0])
    print('Val accuracy:', val_score[1])

utils.plot_history(model_results)
plt.savefig('alexnet_model_history.png')

# Always same results
np.random.seed(1)
y_val_pred = model.predict(X_val, batch_size=32, verbose=0)
y_val_pred = np.round(y_val_pred).astype(int)
is_lgg = y_val.argmax(axis=1) == 0
utils.plot_predicted_samples(4, X_val[is_lgg], y_val[is_lgg], y_val_pred[is_lgg], 'Validation set - LGG', (WIDTH, HEIGHT))
plt.savefig('alexnet_samples_lgg.png')
utils.plot_predicted_samples(4, X_val[is_lgg == False], y_val[is_lgg == False], y_val_pred[is_lgg == False], 'Validation set - HGG', (WIDTH, HEIGHT))
plt.savefig('alexnet_samples_hgg.png')
utils.plot_model(model, 'skull_classification_model_alexnet.png', show=False)


