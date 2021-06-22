"""
this is test realm for CNN+autoencoder

things to fiddle:
[ ] 1. CNN autoencoder_decoder with CNN
[ ] 2. CNN hyperparameter tuning

overall notes:
1. how to calcualte output shape for the CNN layers, using input (28, 28, 1) as an example
a. padding='same': when padding is set to 'same', the output shape is the as as the input shape.
    '0's are padded to fill the shrinkage created by convolution
# filter number is the output channel (third dim) number
b. Conv2D(filters=16, kernel=(3, 3), padding='same', ...)
    output: 28, 28, 16
c. MaxPooling2D((2, 2))  # 28/2. no change to the number of channels
    output: 14, 14, 16
d. Conv2D(filters=8, kernel=(3, 3), padding='same', ...)
    output: 14, 14, 8
e. UpSampling2D((2, 2))  # 14*2. no change to the number of channels
    output: 28, 28, 8
"""


# ------ load modules ------
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Layer, Flatten, Dense, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm


# ------ model ------
class CNN2d_encoder(Layer):
    def __init__(self, initial_shape, bottleneck_dim):
        super(CNN2d_encoder, self).__init__()
        self.bottleneck_dim = bottleneck_dim
        # CNN encoding sub layers
        self.conv2d_1 = Conv2D(16, (3, 3), activation='relu',
                               padding='same', input_shape=initial_shape)  # output: 28, 28, 16
        self.maxpooling_1 = MaxPooling2D((2, 2))  # output: 14, 14, 16
        self.conv2d_2 = Conv2D(8, (3, 3), activation='relu',
                               padding='same')  # output: 14, 14, 8
        self.maxpooling_2 = MaxPooling2D((2, 2))  # output: 7, 7, 8
        self.fl = Flatten()  # 7*7*8=392
        self.encoded = Dense(bottleneck_dim, activation='relu')

    def call(self, initial_shape):
        x = self.conv2d_1(initial_shape)
        x = self.maxpooling_1(x)
        x = self.conv2d_2(x)
        x = self.maxpooling_2(x)
        x = self.fl(x)
        x = self.encoded(x)
        return x


class CNN2d_decoder(Layer):
    def __init__(self, encoded_dim):
        """
        UpSampling2D layer: a reverse of pooling2d layer
        """
        super(CNN2d_decoder, self).__init__()
        # CNN decoding sub layers
        self.encoded_input = Dense(encoded_dim, activation='relu')
        self.dense1 = Dense(7*7*8, activation='relu')  # output: 392
        self.reshape1 = Reshape(target_shape=(7, 7, 8))  # output: 7, 7, 8
        self.conv2d_1 = Conv2D(8, (3, 3), activation='relu',
                               padding='same')  # output: 7, 7, 8
        self.upsampling_1 = UpSampling2D(size=(2, 2))  # output: 14, 14, 28
        self.conv2d_2 = Conv2D(16, (3, 3), activation='relu',
                               padding='same')  # output: 14, 14, 16
        self.upsampling_2 = UpSampling2D((2, 2))  # output: 28, 28, 16
        self.decoded = Conv2D(1, (3, 3), activation='sigmoid',
                              padding='same')  # output: 28, 28, 1

    def call(self, encoded_dim):
        x = self.encoded_input(encoded_dim)
        x = self.dense1(x)
        x = self.reshape1(x)
        x = self.conv2d_1(x)
        x = self.upsampling_1(x)
        x = self.conv2d_2(x)
        x = self.upsampling_2(x)
        x = self.decoded(x)
        return x


class autoencoder_decoder(Model):
    def __init__(self, initial_shape, bottleneck_dim):
        super(autoencoder_decoder, self).__init__()
        self.initial_shape = initial_shape
        self.bottleneck_dim = bottleneck_dim
        self.encoder = CNN2d_encoder(
            initial_shape=self.initial_shape, bottleneck_dim=bottleneck_dim)
        self.decoder = CNN2d_decoder(encoded_dim=bottleneck_dim)

    def call(self, initial_shape):  # putting two models togeter
        x = self.encoder(initial_shape=initial_shape)
        z = self.decoder(x)
        return z

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, z):
        z = self.decoder(z)
        return z

    def model(self):
        x = Input(self.initial_shape)
        return Model(inputs=[x], outputs=self.call(x))


# ------ data ------
# -- loading data --
# x_train: 60000, 28, 28. no need to have y
(x_train, _), (x_test, _) = mnist.load_data()

# -- data transformation and normalization --
x_train, x_test = x_train.astype('float32') / 255, x_test.astype(
    'float32') / 255  # transform from int to float and min(0.0)-max(255.0) normalization into 0-1 (sigmoid)


x_train.shape
# -- data vectorization: 28*28 = 784 --
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# ------ training ------
# -- early stop and optimizer --
earlystop = EarlyStopping(monitor='val_loss', patience=5)
# earlystop = EarlyStopping(monitor='loss', patience=5)
callbacks = [earlystop]
optm = Adam(learning_rate=0.001)

# -- model --
m = autoencoder_decoder(initial_shape=x_train.shape[1:], bottleneck_dim=64)
# the output is sigmoid, therefore binary_crossentropy
m.compile(optimizer=optm, loss="binary_crossentropy")

m.model().summary()

# -- training --
m_history = m.fit(x=x_train, y=x_train, batch_size=256, epochs=100, callbacks=callbacks,
                  shuffle=True, validation_data=(x_test, x_test))

# -- inspection --
reconstruction_test = m.predict(x_test)

m.encoded(x_test)  # use the trained encoder to encode the input data

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
input_img = Input(shape=(28, 28, 1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
