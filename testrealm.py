"""
current objectives:
1. class-based DL analysis
2. tensorflow 2.0 transition: tf.keras

follow: https://www.tensorflow.org/guide/keras/rnn
and: https://www.tensorflow.org/guide/keras/custom_layers_and_models
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# ------ clear session ------
tf.keras.backend.clear_session()


# ------ DL classes ------
class my_linear(layers.Layer):
    """
    a dense layer with linear activation.
    It has a state: the variables w and b.
    """

    def __init__(self, units, input_dim):
        super(my_linear, self).__init__()
        w_init = tf.random_normal_initializer()


# ------ setup working dictory ------
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, '0.data')
res_dir = os.path.join(main_dir, '1.results')
