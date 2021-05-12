#!/usr/bin/env python3
"""
Current Objectives:
[X] 0. Practice autoencoder: using the MNIST dataset (digits hand-writing)
    [X] a. Get the modelling up and running
    [X] b. Try google colab
    [X] c. Setup OpenCL for AMD card
    [X] d. Setp up intel CPU optimization: using venv: conda_venv_intel

NOTE: Autoencoder is neither supervised learning, nor unsupervised learning. 
    It is "self-supervsied learning"
NOTE: This dense version WILL NOT run well without GPU acceleration. Do it on Google Colab
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
# ------ import modules ------
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.python.compiler.mlcompute import mlcompute
from tqdm import tqdm

tf.compat.v1.disable_eager_execution()
mlcompute.set_mlc_device(device_name="any")


# ------ custom functions ------

# ----- model constuction ------
"""
1.
The model we are building here only has three layers:
Input -> Encoded -> Decoded

2. 
Here we use the double parentheses syntax (example, transfer learning):
    model = VGG19(weights='imagenet',include_top=False)
    model.trainable=False
    layer1 = Flatten(name='flat')(model)
    layer2 = Dense(512, activation='relu', name='fc1')(layer1)
    layer3 = Dense(512, activation='relu', name='fc2')(layer2)
    layer4 = Dense(10, activation='softmax', name='predictions')(layer3)

This is the same as (example):
    model = VGG19(weights='imagenet',include_top=False)
    model.trainable=False
    model.add( Flatten(name='flat'))
    model.add( Dense(512, activation='relu', name='fc1'))
    model.add( Dense(512, activation='relu', name='fc2'))
    model.add( Dense(10, activation='softmax', name='predictions'))

NOTE: we can also subclass Model class to constuct the same thing
"""
# - autoencoder (complete model) -
encoding_dim = 32  # compressed size

# Below: input tensor object.
"""
shape: input dimensions. 
    Shape=(784, ) means a single sample is a 784-dimension vector
    There is also a "batch size" parameter. When not set, the batch size is one.
    NOTE: Batch size and number of batches are two different things.
        Batch size: total number of training samples present in a single batch
        Number of batches: number of batches

Where is 784 from? MINST data dimensions reshaping: 60000, 28, 28 -> 60000, 28*28 = 60000, 784
    Therefore, the key is data vectorization (contatenation etc). 

By having a shape=784, each batch is a single sample (image). 

When we fit the model, we will need to set a batch size (how many batches per run)
    It is also noted that an "epoch" is the run of all the batches required to go through all the samples
"""
input_img = Input(shape=(784, ))

# Chain 1, encoding layer: from 784 to 32
encoded = Dense(encoding_dim, activation='relu')(input_img)

# Chain 2, encoding layer: from 32 to 784
# NOTE: 0,1 is the output format, simoid
decoded = Dense(784, activation='sigmoid')(encoded)

# - Final model -
# This is a tf.keras Model class
# Syntax: Model(input, output), in which output is 0-1, i.e. sigmoid
# Sigmoid: all the input data will be rescaled into [0, 1]
autoencoder = Model(input_img, decoded)

# - Seperate encoder model -
encoder = Model(input_img, encoded)

# - Seperate decoder model -
# input has the same dimension of the encoded data
encoded_input = Input(shape=(encoding_dim, ))
# retrieve the last layer of the autoencoder model, which is the decoding layer
# NOTE (-1 explanation):
#       >>> tst = [1,2,3,4,5]
#       >>> tst[-1]
#       5
decoder_layer = autoencoder.layers[-1]
# NOTE: put a pin here.
decoder = Model(encoded_input, decoder_layer(encoded_input))

# ------ train the model ------
# - compile/configure the models -
# NOTE: adam is a extension of adadelta. adam is more robust, but has a learning rate to be tuned
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# - load the MINST data -
# NOTE (full data): (x_train, y_train), (x_test, y_test) = mnist.load_data()
# Below we use only the X as teh data label is note requried for the current autoencoder practice
# All data objects from this data loader are numpy array
(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape  # (60000, 28, 28)
x_test.shape  # (10000, 28, 28)

# data rescaling to 0-1: min (0)-max(255) normalization
# x = (x-0) / (255-0)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# data reshape: 60000, 28, 28 -> 60000, 28*28 = 60000, 784
# this step concatnates/flattern the 28*28 array into a dim 784 vector
len(x_train)  # 60000
x_train.shape[1:]  # 28, 28
"""
test array shape: 2, 3, 4
    tst_array = np.array([[[0,1,2,3],
    [0,1,2,3],
    [0,1,2,3]],
    [[4,5,6,7],
    [4,5,6,7],
    [4,5,6,7]]])

test array reshape: 2, 3*4 = 2, 12
    tst_array.reshape(len(tst_array), np.prod(tst_array.shape[1:]))

we get:
    array([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7]])
"""
# NOTE: nparray.shape[1:]: product of the dim numbers excluding the first one
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

# - fit the model -
# callbacaks
earlystop = EarlyStopping(monitor='val_loss', patience=5)
callbacks = [earlystop]

# ftting
"""
input, output = x_train,, x_train: because the input and outpout are the same for autoencoder 
batch_size: we don't set the number of batches. Instead, we set batch_size, which will determine the number of batches
    given the total sample number. 
"""
autoencoder.fit(x_train, x_train,
                epochs=50, batch_size=256,
                callbacks=callbacks,
                shuffle=True, validation_data=(x_test, x_test))

# ------ display resutls ------
# - predict -
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)  # get the encoded image
decoded_imgs = decoder.predict(encoded_imgs)  # decode the encoded image

# - visulization -
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
