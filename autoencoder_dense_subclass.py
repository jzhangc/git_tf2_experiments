"""
this is test realm
NOTE: if to use subclassing API to build autoencoder, we cannot access intermediate layers for encoded data. 
This is by TF2's design.

However, we can package the layers into "sub" models before stacking them into the final model. Then we can 
just use model.submodel.predict() to extract intermediate predictions.

"""


# ------ load modules ------
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Layer, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tqdm import tqdm


# ------ TF device check ------
tf.config.list_physical_devices()


# ------ model ------
class Encoder(Layer):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self.output_dim = 16  # bottleneck size

        # encoder sub layers
        self.hidden_layer1 = Dense(
            units=64, activation='relu', kernel_initializer='he_uniform', input_dim=input_dim)
        self.bn1 = BatchNormalization()
        self.leakyr1 = LeakyReLU()
        self.hidden_layer2 = Dense(units=32, activation='relu')
        self.bn2 = BatchNormalization()
        self.leakyr2 = LeakyReLU()
        self.output_layer = Dense(units=self.output_dim, activation='sigmoid')

    def call(self, input):
        x = self.hidden_layer1(input)
        x = self.bn1(x)
        x = self.leakyr1(x)
        x = self.hidden_layer2(x)
        x = self.bn2(x)
        x = self.leakyr2(x)
        x = self.output_layer(x)
        return x


class Decoder(Layer):
    def __init__(self, latent_dim, original_dim):
        super(Decoder, self).__init__()
        # decoder sub layers
        self.hidden_layer1 = Dense(
            units=latent_dim, activation='relu', kernel_initializer='he_uniform')
        self.bn1 = BatchNormalization()
        self.leakyr1 = LeakyReLU()
        self.hidden_layer2 = Dense(units=32, activation='relu')
        self.bn2 = BatchNormalization()
        self.leakyr2 = LeakyReLU()
        self.output_layer = Dense(units=original_dim, activation='sigmoid')

    def call(self, input):
        x = self.hidden_layer1(input)
        x = self.bn1(x)
        x = self.leakyr1(x)
        x = self.hidden_layer2(x)
        x = self.bn2(x)
        x = self.leakyr2(x)
        x = self.output_layer(x)
        return x


class autoencoder_decoder(Model):
    """
    Alternatively to writing layers into the __init__ method,
    one can also write them in a "build" method, e.g. def build(self, input_shape). 

    This is optional. Utility of this is for user to run Model.build()
    separately to build a model.  

    It also appears to have better performance with a build method
    """

    def __init__(self, original_dim):
        super(autoencoder_decoder, self).__init__()
        self.original_dim = original_dim
        self.encoder = Encoder(input_dim=self.original_dim)
        self.decoder = Decoder(latent_dim=self.encoder.output_dim,
                               original_dim=self.original_dim)

    def call(self, input):  # putting two layers togeter
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
        This method enables the correct model.summary() results:
        model.model().summary()
        """
        x = Input(self.original_dim)
        return Model(inputs=[x], outputs=self.call(x))


# ------ data ------
# -- loading data --
# x_train: 60000, 28, 28. no need to have y
(x_train, _), (x_test, _) = mnist.load_data()

# -- data transformation and normalization --
x_train, x_test = x_train.astype('float32') / 255, x_test.astype(
    'float32') / 255  # transform from int to float and min(0.0)-max(255.0) normalization into 0-1  (sigmoid)

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
m = autoencoder_decoder(original_dim=x_train.shape[1])
# the output is sigmoid, therefore binary_crossentropy
m.compile(optimizer=optm, loss="binary_crossentropy")

# -- training --
m_history = m.fit(x=x_train, y=x_train, batch_size=256, epochs=100, callbacks=callbacks,
                  shuffle=True, validation_data=(x_test, x_test))

# -- inspection --
reconstruction_test = m.predict(x_test)

m.encode(x_test)  # use the trained encoder to encode the input data


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
