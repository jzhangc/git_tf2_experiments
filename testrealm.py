"""
current objectives:
1. class-based DL analysis
2. tensorflow 2.0 transition: tf.keras

follow: https://www.tensorflow.org/guide/keras/rnn
and: https://www.tensorflow.org/guide/keras/custom_layers_and_models

https://www.tensorflow.org/guide/keras/rnn
https://www.tensorflow.org/tutorials/structured_data/time_series
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

# ------ clear session ------
tf.keras.backend.clear_session()


# ------ DL classes ------
class myModel(Model):
    def __init__(self, num_classes=10):
        super(myModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # layers
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(
            num_classes, activation='sigmoid')  # output layer

    def call(self, inputs):
        # Define forward pass using the layers
        x = self.dense_1(inputs)
        # use output of dense_1 as input for output layer
        return self.dense_2(x)


model = MyModel(num_classes=10)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)


# ------ setup working dictory ------
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, '0.data')
res_dir = os.path.join(main_dir, '1.results')
