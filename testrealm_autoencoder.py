"""
this is test realm
"""


# ------ load modules ------
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from datatime import datatime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


# ------ model ------


# ------ data ------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
