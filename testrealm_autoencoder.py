"""
This is test realm for autoencoder
Current: autoencoder for feature encoding and extracting using tf.keras functional API
"""


# ------ load modules ------
from inspect import Attribute
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Layer, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
# from tensorflow.python.compiler.mlcompute import mlcompute
from tqdm import tqdm

# tf.compat.v1.disable_eager_execution()
# mlcompute.set_mlc_device(device_name="gpu")


# ------ model ------
class Coder(object):
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        # self.latent_dim = latent_dim

    @property
    def Encoder(self):
        return self._Encoder

    @Encoder.setter
    def Encoder(self):
        # define encoder
        visible = Input(shape=(self.n_inputs,))
        # encoder level 1
        e = Dense(self.n_inputs*2)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # encoder level 2
        e = Dense(self.n_inputs)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # bottleneck
        n_bottleneck = self.n_inputs
        bottleneck = Dense(n_bottleneck)(e)

    @property
    def Decoder(self):
        return self._Decoder


# define encoder
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)
# define encoder
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)
# define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='linear')(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)
# compile autoencoder model
model.compile(optimizer='adam', loss='mse')
# plot the autoencoder
plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)


# ------ data ------
# -- loading data --
# x_train: 60000, 28, 28. no need to have y
(x_train, _), (x_test, _) = mnist.load_data()

# -- data transformation and normalization --
x_train, x_test = x_train.astype('float32') / 255, x_test.astype(
    'float32') / 255  # transform from int to float and min(0.0)-max(255.0) normalization into 0-1

# -- data vectorization: 28*28 = 784 --
# ndarray.shape: x, y, z. index: [0, 1, 2]. so y and z are ndarray.shape[1:]
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

# ------ training ------
# -- early stop and optimizer --
earlystop = EarlyStopping(monitor='val_loss', patience=5)
callbacks = [earlystop]
optm = Adam(learning_rate=0.001)

# -- model --


# -- training --


# -- inspection --


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
    plt.imshow(reconstruction_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# ------ save model ------


# ------ testing ------
