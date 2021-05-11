"""
this is test realm
"""


# ------ load modules ------
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datatime import datatime
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras.utils import to_categorical
from tqdm import tqdm

os.environ["KERAS_BACKEND"] = 'plaidml.keras.backend'


# ------ model ------


# ------ data ------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
