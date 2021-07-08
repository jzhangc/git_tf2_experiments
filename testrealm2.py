"""
Current objectives:
test smalls things for implementing 
"""

# ------ modules ------
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.data_utils import getSelectedDataset, getSingleCsvDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
# from skmultilearn.model_selection import iterative_train_test_split


# ------ check device ------
tf.config.list_physical_devices()


# ------ function -------
def map_func(filepath: tf.Tensor, label: tf.Tensor, processing=False):
    # - read file and assign label -
    fname = filepath.numpy().decode('utf-8')
    f = np.loadtxt(fname).astype('float32')
    lb = label

    # - processing if needed -
    if processing:
        f = f/f.max()
        # f_std = (f - f.min(axis=0)) / (f.max(axis=0) - f.min(axis=0))
        # f = f_std * (1 - 0) + 0
        # print(f.shape[:])
        f = np.reshape(f, (f.shape[0], f.shape[1], 1))
    f = tf.convert_to_tensor(f, dtype=tf.float32)

    return f, lb


# ------ test realm ------
main_dir = os.path.abspath('./')
file_path = os.path.join(main_dir, 'data/test_dat.csv')

# - test: load using the tf.data.Dataset API -
tst_dat, feature_list = getSingleCsvDataset(
    file_path, batch_size=32,
    label_var='group', column_to_exclude=['subject', 'PCL'])


n = 0  # batches
for i in tst_dat:
    n += 1
n

for a, b in tst_dat.take(6):
    print(a)
    print(b)
    print()
    break

# - below: create one hot encoding for multiclass labels -


# - resampling -
X_indices = np.arange(tst_data_size)

# multiclass
X_train_indices, X_val_indices, y_train_targets, y_val_targets = train_test_split(
    X_indices, encoded_labels, test_size=0.1, stratify=encoded_labels, random_state=53)

tst_test, test_n = getSelectedDataset(tst_dat, X_val_indices)

t = (0, 1)
t2 = t[:]+(2)

# ------ ref ------
# - tf.dataset reference: https://cs230.stanford.edu/blog/datapipeline/ -
# - real TF2 data loader example: https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py -
# - tf.data.Dataset API example: https://medium.com/deep-learning-with-keras/tf-data-build-efficient-tensorflow-input-pipelines-for-image-datasets-47010d2e4330 -
