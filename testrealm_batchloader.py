"""
this is test realm for DL models with BatchMatrixLoader

Objectives:
[x] CNN autoencoder_decoder BatchMatrixLoader
[ ] Implement cross validation
[ ] Implement additional performance metrics (AUC, recall, precision, F1, top-k accuracy)
[ ] Implement plotting functions for the above metrics
"""


# ------ load modules ------
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Flatten, Input, Layer, LeakyReLU,
                                     MaxPooling2D, Reshape, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, roc_curve

from utils.dl_utils import BatchMatrixLoader
from utils.plot_utils import epochsPlot
from utils.other_utils import flatten

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
        # self.maxpooling_2 = MaxPooling2D((2, 2))  # output: 7, 7, 8
        self.maxpooling_2 = MaxPooling2D((5, 5))  # output: 9, 9, 8
        self.fl = Flatten()  # 7*7*8=392
        self.dense1 = Dense(bottleneck_dim, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))
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
        # self.dense1 = Dense(7*7*8, activation='relu')  # output: 392
        self.dense1 = Dense(9*9*8, kernel_regularizer=tf.keras.regularizers.l2(l2=0.01),
                            activation='relu')  # output: 648
        # self.reshape1 = Reshape(target_shape=(7, 7, 8))  # output: 7, 7, 8
        self.reshape1 = Reshape(target_shape=(9, 9, 8))  # output: 9, 9, 8

        self.conv2d_1 = Conv2D(8, (3, 3), activation='relu',
                               padding='same')  # output: 7, 7, 8
        self.bn1 = BatchNormalization()
        self.leakyr1 = LeakyReLU()
        # self.upsampling_1 = UpSampling2D(size=(2, 2))  # output: 14, 14, 28
        self.upsampling_1 = UpSampling2D(size=(5, 5))  # output: 14, 14, 28
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

    def call(self, input):  # putting two models togeter
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


class CnnClassifier(Model):
    def __init__(self, initial_shape, bottleneck_dim, outpout_n):
        super(CnnClassifier, self).__init__()
        self.initial_shape = initial_shape
        self.bottleneck_dim = bottleneck_dim
        # CNN encoding sub layers
        self.conv2d_1 = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.01),
                               padding='same', input_shape=initial_shape)  # output: 28, 28, 16
        self.bn1 = BatchNormalization()
        self.leakyr1 = LeakyReLU()
        self.maxpooling_1 = MaxPooling2D((2, 2))  # output: 14, 14, 16
        self.conv2d_2 = Conv2D(8, (3, 3), activation='relu',
                               padding='same')  # output: 14, 14, 8
        self.bn2 = BatchNormalization()
        self.leakyr2 = LeakyReLU()
        # self.maxpooling_2 = MaxPooling2D((2, 2))  # output: 7, 7, 8
        self.maxpooling_2 = MaxPooling2D((5, 5))  # output: 9, 9, 8
        self.fl = Flatten()  # 7*7*8=392
        self.dense1 = Dense(bottleneck_dim, activation='relu',
                            activity_regularizer=tf.keras.regularizers.l2(l2=0.01))
        self.encoded = LeakyReLU()
        self.dense2 = Dense(outpout_n, activation='softmax')

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
        x = self.dense2(x)
        return x

    def model(self):
        """
        This method enables correct model.summary() results:
        model.model().summary()
        """
        x = Input(self.initial_shape)
        return Model(inputs=[x], outputs=self.call(x))


# ------ data ------
# -- batch loader data --
tst_tf_dat = BatchMatrixLoader(filepath='./data/tf_data', target_file_ext='txt',
                               manual_labels=None, label_sep=None, pd_labels_var_name=None, model_type='semisupervised',
                               multilabel_classification=False, x_scaling='minmax', x_min_max_range=[0, 1], resmaple_method='random',
                               training_percentage=0.8, verbose=False, random_state=1)
tst_tf_train, tst_tf_test = tst_tf_dat.generate_batched_data()

for a in tst_tf_train:
    print(a[1].shape)
    # break

tst_tf_train.element_spec


# ------ training ------
# -- early stop and optimizer --
earlystop = EarlyStopping(monitor='val_loss', patience=5)
# earlystop = EarlyStopping(monitor='loss', patience=5)
callbacks = [earlystop]
optm = Adam(learning_rate=0.001)

# -- model --
# -batch loader data -
m = AutoEncoderDecoder(initial_shape=(90, 90, 1), bottleneck_dim=64)

m.model().summary()

# the output is sigmoid, therefore binary_crossentropy
m.compile(optimizer=optm, loss="binary_crossentropy")


# -- training --
# - batch loader data -
m_history = m.fit(tst_tf_train, epochs=80, callbacks=callbacks,
                  validation_data=tst_tf_test)

epochsPlot(model_history=m_history)

# -- inspection --
reconstruction_test = m.predict(tst_tf_test)

reconstruction_test.shape

plt.imshow(reconstruction_test[0])


# - visulization -
for a, _ in tst_tf_test:
    reconstruction = m.predict(a)
    # display decoded
    plt.imshow(a[0])
    # plt.gray()
    # # display original
    # plt.imshow(reconstruction[0])
    # plt.gray()
    break


# ------ save model ------
m.save('./results/subclass_autoencoder_batch', save_format='tf')


# ------ testing ------
tst_tf_dat = BatchMatrixLoader(filepath='./data/tf_data', target_file_ext='txt',
                               manual_labels=None, label_sep=None, pd_labels_var_name=None, model_type='classification',
                               multilabel_classification=False, x_scaling='minmax', x_min_max_range=[0, 1], resmaple_method='random',
                               training_percentage=0.8, verbose=False, random_state=1)
tst_tf_train, tst_tf_test = tst_tf_dat.generate_batched_data(batch_size=4)

tst_tf_train.element_spec


earlystop = EarlyStopping(monitor='val_loss', patience=5)
callbacks = [earlystop]
optm = Adam(learning_rate=0.001)
tst_m = CnnClassifier(initial_shape=(90, 90, 1),
                      bottleneck_dim=64, outpout_n=10)
tst_m.model().summary()

tst_m.compile(optimizer=optm, loss="categorical_crossentropy",
              metrics=['categorical_accuracy', tf.keras.metrics.AUC()])
tst_m_history = tst_m.fit(tst_tf_train, epochs=80,
                          callbacks=callbacks,
                          validation_data=tst_tf_test)

epochsPlot(model_history=tst_m_history,
           accuracy_var='categorical_accuracy',
           val_accuracy_var='val_categorical_accuracy')

pred = tst_m.predict(tst_tf_test)
pred[0]
pred.shape
tst_tf_test

tst_test = np.ndarray()

tst_t = np.ndarray((0, 10))
for _, b in tst_tf_test:
    # print(b.numpy())
    bn = b.numpy()
    # print(type(bn))
    tst_t = np.concatenate((tst_t, bn), axis=0)
tst_t.shape

tst_tf_dat.test_batch_n
