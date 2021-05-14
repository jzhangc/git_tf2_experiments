"""
this is test realm
NOTE: if to use subclassing API to build autoencoder, we cannot access intermediate layers for encoded data. 
This is by TF2's design.
"""


# ------ load modules ------
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# from tensorflow.python.compiler.mlcompute import mlcompute
from tqdm import tqdm

# tf.compat.v1.disable_eager_execution()
# mlcompute.set_mlc_device(device_name="cpu")


# ------ model ------
class Encoder(Layer):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.output_dim = 16
        self.hidden_layer1 = Dense(
            units=latent_dim, activation='relu', kernel_initializer='he_uniform')
        self.hidden_layer2 = Dense(units=32, activation='relu')
        self.output_layer = Dense(units=self.output_dim, activation='sigmoid')

    def call(self, input_dim):
        x = self.hidden_layer1(input_dim)
        x = self.hidden_layer2(x)
        x = self.output_layer(x)
        return x


class Decoder(Layer):
    def __init__(self, latent_dim, original_dim):
        super(Decoder, self).__init__()
        self.hidden_layer1 = Dense(
            units=latent_dim, activation='relu', kernel_initializer='he_uniform')
        self.hidden_layer2 = Dense(units=32, activation='relu')
        self.output_layer = Dense(units=original_dim, activation='sigmoid')

    def call(self, encoded_dim):
        x = self.hidden_layer1(encoded_dim)
        x = self.hidden_layer2(x)
        x = self.output_layer(x)
        return x


class autoencoder_decoder(Model):
    def __init__(self, original_dim, latent_dim):
        super(autoencoder_decoder, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=self.encoder.output_dim,
                               original_dim=original_dim)

    def call(self, input_dim):
        x = self.encoder(input_dim)
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
callbacks = [earlystop]
optm = Adam(learning_rate=0.001)

# -- model --
m = autoencoder_decoder(original_dim=x_train.shape[1], latent_dim=64)
# the output is sigmoid, therefore binary_crossentropy
m.compile(optimizer=optm, loss="binary_crossentropy")

# -- training --
m.fit(x=x_train, y=x_train, batch_size=256, epochs=50, callbacks=callbacks,
      shuffle=True, validation_data=(x_test, x_test))

# -- inspection --
reconstruction_test = m.predict(x_test)

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
