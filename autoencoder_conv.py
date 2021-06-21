"""
this is test realm for CNN+autoencoder

things to fiddle:
[ ] 1. CNN autoencoder_decoder with CNN
[ ] 2. CNN hyperparameter tuning
"""


# ------ load modules ------
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import BackupAndRestore
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tqdm import tqdm


# ------ model ------
class CNN2d_encoder(Model):  # this a model class not a layer class
    def __init__(self, input_shape):
        super(CNN2d_encoder, self).__initi__()
        # set up CNN layers
        self.conv2d_1 = Conv2D(16, (3, 3), activation='relu',
                               padding='same', input_shape=input_shape)
        self.maxpooling_1 = MaxPooling2D((2, 2), padding='same')
        self.conv2d_2 = Conv2D(8, (3, 3), activation='relu', padding='same')
        self.maxpooling_2 = MaxPooling2D((2, 2), padding='same')
        self.con2d_3 = Conv2D(8, (3, 3), activation='relu', padding='same')
        self.encoded = MaxPooling2D(
            (2, 2), padding='same')  # output shape: 4, 4, 8

    def call(self, input_shape):
        x = self.conv2d_1(input_shape=input_shape)
        x = self.maxpooling_1(x)
        x = self.conv2d_2(x)
        x = self.maxpooling_2(x)
        x = self.con2d_3(x)
        x = self.encoded(x)
        return x


class CNN2d_decoder(Model):  # this is a model class not a layer class
    def __init__(self, encoded_shape):
        """
        UpSampling2D layer: a reverse of pooling2d layer
        """
        super(CNN2d_decoder, self).__init__():
        self.conv2d_1 = Conv2D(8, (3, 3), activation='relu',
                               padding='same', input_shape=encoded_shape)
        self.upsampling_1 = UpSampling2D(size=(2, 2))
        self.conv2d_2 = Conv2D(8, (3, 3), acviation='relu', padding='same')
        self.upsampling_2 = UpSampling2D(size=(2, 2))
        self.conv2d_3 = Conv2D(16, (3, 3), acviation='relu', padding='same')
        self.upsampling_3 = UpSampling2D((2, 2))
        self.decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')

    def call(self, encoded_shape):
        x = self.conv2d_1(encoded_shape=encoded_shape)
        x = self.upsampling_1(x)
        x = self.conv2d_2(x)
        x = self.upsampling_2(x)
        x = self.conv2d_3(x)
        x = self.upsampling_3(x)
        x = self.decoded(x)
        return x


class autoencoder_decoder(Model):
    def __init__(self, input_shape):
        super(autoencoder_decoder, self).__init__()
        self.encoder = CNN2d_encoder(input_shape=input_shape)
        self.decoder = CNN2d_decoder(
            encoded_shape=self.encoder.encoded.output_shape)

    def call(self, input_shape):  # putting two models togeter
        x = self.encoder(input_shape)
        x = self.decoder(x)
        return x


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
# earlystop = EarlyStopping(monitor='loss', patience=5)
callbacks = [earlystop]
optm = Adam(learning_rate=0.001)

# -- model --
m = autoencoder_decoder(original_dim=x_train.shape[1], latent_dim=64)
# the output is sigmoid, therefore binary_crossentropy
m.compile(optimizer=optm, loss="binary_crossentropy")

# -- training --
m_history = m.fit(x=x_train, y=x_train, batch_size=256, epochs=150, callbacks=callbacks,
                  shuffle=True, validation_data=(x_test, x_test))

# -- inspection --
reconstruction_test = m.predict(x_test)

m.encoder.predict(x_test)  # use the trained encoder to encode the input data

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
m.save('../results/subclass_autoencoder', save_format='tf')


# ------ testing ------
