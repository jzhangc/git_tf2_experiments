"""
things to fiddle:
[x] 1. CNN AutoEncoderDecoder with CNN
[ ] 2. CNN hyperparameter tuning

Overall notes:
1. how to calculate output shape for the CNN layers, using input (28, 28, 1) as an example
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

2. how to decide the filter size
a. there is a preference for odd number sizes over even number sizes, e.g. 3x3, 5x5 etc.
    Odd number sizes have a centroid. 3x3 is the most popular one. 
"""


# ------ load modules ------
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Layer, Flatten, Dense, Reshape, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from utils.plot_utils import epochsPlot, epochsPlotV2
from utils.dl_utils import WarmUpCosineDecayScheduler


# ------ TF device check ------
tf.config.list_physical_devices()


# ------ model ------
class CNN2d_encoder(Layer):
    def __init__(self, initial_shape, bottleneck_dim):
        super(CNN2d_encoder, self).__init__()
        self.bottleneck_dim = bottleneck_dim
        # CNN encoding sub layers
        self.conv2d_1 = Conv2D(16, (3, 3), activation='relu',
                               padding='same', input_shape=initial_shape)  # output: 28, 28, 16
        self.bn1 = BatchNormalization()
        self.leakyr1 = LeakyReLU()
        self.maxpooling_1 = MaxPooling2D((2, 2))  # output: 14, 14, 16
        self.conv2d_2 = Conv2D(8, (3, 3), activation='relu',
                               padding='same')  # output: 14, 14, 8
        self.bn2 = BatchNormalization()
        self.leakyr2 = LeakyReLU()
        self.maxpooling_2 = MaxPooling2D((2, 2))  # output: 7, 7, 8
        self.fl = Flatten()  # 7*7*8=392
        self.dense1 = Dense(bottleneck_dim, activation='relu')
        self.encoded = LeakyReLU()

    def call(self, input):
        x = self.conv2d_1(input)
        x = self.bn1(x)
        x = self.leakyr1(x)
        x = self.maxpooling_1(x)
        x = self.conv2d_2(x)
        x = self.bn2(x)
        x = self.leakyr2(x)
        x = self.maxpooling_2(x)
        x = self.fl(x)
        x = self.dense1(x)
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
        self.bn1 = BatchNormalization()
        self.leakyr1 = LeakyReLU()
        self.upsampling_1 = UpSampling2D(size=(2, 2))  # output: 14, 14, 28
        self.conv2d_2 = Conv2D(16, (3, 3), activation='relu',
                               padding='same')  # output: 14, 14, 16
        self.bn2 = BatchNormalization()
        self.leakyr2 = LeakyReLU()
        self.upsampling_2 = UpSampling2D((2, 2))  # output: 28, 28, 16
        self.decoded = Conv2D(1, (3, 3), activation='sigmoid',
                              padding='same')  # output: 28, 28, 1

    def call(self, input):
        x = self.encoded_input(input)
        x = self.dense1(x)
        x = self.reshape1(x)
        x = self.conv2d_1(x)
        x = self.bn1(x)
        x = self.leakyr1(x)
        x = self.upsampling_1(x)
        x = self.conv2d_2(x)
        x = self.bn2(x)
        x = self.leakyr2(x)
        x = self.upsampling_2(x)
        x = self.decoded(x)
        return x


class AutoEncoderDecoder(Model):
    def __init__(self, initial_shape, bottleneck_dim):
        super(AutoEncoderDecoder, self).__init__()
        self.initial_shape = initial_shape
        self.bottleneck_dim = bottleneck_dim
        self.encoder = CNN2d_encoder(
            initial_shape=self.initial_shape, bottleneck_dim=bottleneck_dim)
        self.decoder = CNN2d_decoder(encoded_dim=bottleneck_dim)

    def call(self, input):  # putting two models together
        x = self.encoder(input)
        z = self.decoder(x)
        return z

    def encode(self, x):
        """
        This method is used to encode data using the trained encoder
        """
        x = self.encoder(x)
        return x

    def decode(self, z):
        z = self.decoder(z)
        return z

    def model(self):
        """
        This method enables correct model.summary() results:
        model.model().summary()
        """
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
# earlystop = EarlyStopping(monitor='val_loss', patience=5)
# # earlystop = EarlyStopping(monitor='loss', patience=5)
# callbacks = [earlystop]
# optm = Adam(learning_rate=0.001, decay=0.001/80)

# Training batch size, set small value here for demonstration purpose.
batch_size = 512
epochs = 40
n_training_samples = x_train.shape[0]

# Base learning rate after warmup.
learning_rate_base = 0.001
total_steps = int(epochs * n_training_samples / batch_size)

# Number of warmup epochs.
warmup_epoch = 10
warmup_steps = int(warmup_epoch * n_training_samples / batch_size)
warmup_batches = warmup_epoch * n_training_samples / batch_size

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0)

# optimizer
# optm = Adam(learning_rate=0.001, decay=0.001/80)  # decay?? lr/epoch
optm = Adam()

# early stop
earlystop = EarlyStopping(monitor='val_loss', patience=5)

# callback
callbacks = [warm_up_lr, earlystop]


# -- model --
m = AutoEncoderDecoder(initial_shape=x_train.shape[1:], bottleneck_dim=64)
# the output is sigmoid, therefore binary_crossentropy
m.compile(optimizer=optm, loss="binary_crossentropy")

m.model().summary()

# -- training --
m_history = m.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                  shuffle=True, validation_data=(x_test, x_test))

m_history = m.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=5, callbacks=callbacks,
                  shuffle=True, validation_data=(x_test, x_test))

# -- inspection --
reconstruction_test = m.predict(x_test)

m.encode(x_test).shape
m.encode(x_test)[0]  # use the trained encoder to encode the input data

# - visualization -
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

epochsPlotV2(model_history=m_history)


# ------ save model ------
m.save('./results/subclass_autoencoder', save_format='tf')


# ------ testing ------
