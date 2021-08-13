"""
Current objectives:
test smalls things for implementing single file "out of memory" data loading

Objectives
[ ] Implement cross validation
[ ] Implement additional performance metrics (AUC, recall, precision, F1, top-k accuracy)
[ ] Implement plotting functions for the above metrics
"""

# ------ modules ------
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, StandardScaler

from utils.data_utils import getSelectedDataset, getSingleCsvDataset
from utils.other_utils import error

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
tst = pd.read_csv(file_path)

tst[['subject', 'PCL']].to_numpy().shape

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


# ------ ref ------
# - tf.dataset reference: https://cs230.stanford.edu/blog/datapipeline/ -
# - real TF2 data loader example: https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py -
# - tf.data.Dataset API example: https://medium.com/deep-learning-with-keras/tf-data-build-efficient-tensorflow-input-pipelines-for-image-datasets-47010d2e4330 -
