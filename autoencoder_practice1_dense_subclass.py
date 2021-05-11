#!/usr/bin/env python3
"""
NOTE: Use the FUNCTIONAL API. This subclassing API is just for fun. 

SEE BELOW:
Saving the model to HDF5 format requires the model to be a Functional model or a Sequential model. 
It does not work for subclassed models, because such models are defined via the body of a Python method, 
which isn't safely serializable. Consider saving to the Tensorflow SavedModel format (by setting save_format="tf") 
or using `save_weights`.

Current Objectives:
[X] 0. Practice autoencoder: using the MNIST dataset (digits hand-writing)
    [X] a. Construct the models with subclassing keras.Model and keras.Layer

NOTE: Autoencoder is neither supervised learning, nor unsupervised learning. 
    It is "self-supervsied learning"
NOTE: This dense version WILL NOT run well without GPU acceleration. Do it on Google Colab

NOTE: Use conda_venv_intel for this one

"""
# ------ import modules ------
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from tensorflow.keras.datasets import mnist


# ----- model constuction ------
class Autoencoder(Model):
    def __init__(self, intermediate_dim, original_dim, name='autoencoder', **kwargs):
        super(Autoencoder, self).__init__(name=name, **kwargs)
        self.encoder = Dense(intermediate_dim, activation='relu')
        self.decoder = Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


# ------ data prep ------
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
encoding_dim = 32  # compressed size
original_dim = 784


# ------ train the model ------
# - compile/configure the models -
# NOTE: adam is a extension of adadelta. adam is more robust, but has a learning rate to be tuned
m = Autoencoder(intermediate_dim=encoding_dim, original_dim=original_dim)
opt = Adam()
m.compile(optimizer=opt, loss='binary_crossentropy')

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
m.fit(x_train, x_train,
      epochs=50,
      batch_size=256,
      callbacks=callbacks,
      shuffle=True, validation_data=(x_test, x_test))


m.save('./results/subclass_autoencoder')


# ------ display resutls ------
# - predict -
# encode and decode some digits
# note that we take them from the *test* set
decoded_imgs = m.predict(x_test)  # get the encoded image

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

    # # display reconstruction
    # ax = plt.subplot(2, n, i + 1 + n)
    # plt.imshow(decoded_imgs[i].reshape(28, 28))
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
plt.show()
