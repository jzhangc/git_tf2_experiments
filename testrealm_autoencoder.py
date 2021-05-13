"""
this is test realm
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
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


# ------ model ------
class Encoder(Layer):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.output_dim = 16
        self.hidden_layer1 = Dense(
            units=latent_dim, activation='relu', kernel_initializer='he_uniform')
        self.hidden_layer2 = Dense(units=32, activation='relu')
        self.output = Dense(units=self.output_dim, activation='sigmoid')

    def call(self, input_dim):
        x = self.hidden_layer1(input_dim)
        x = self.hidden_layer2(x)
        x = self.output(x)
        return x


class Decoder(Layer):
    def __init__(self, latent_dim, original_dim):
        super(Decoder, self).__init__()
        self.hidden_layer1 = Dense(
            units=latent_dim, activation='relu', kernel_initializer='he_uniform')
        self.hidden_layer2 = Dense(units=32, activation='relu')
        self.output = Dense(unit=original_dim, activation='sigmoid')

    def call(self, encoded_dim):
        x = self.hidden_layer1(encoded_dim)
        x = self.hidden_layer2(x)
        x = self.output(x)
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
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# ------ training ------
# -- model --

# -- training --
