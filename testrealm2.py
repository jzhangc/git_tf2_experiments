"""
Current objectives:
test smalls things for implementing 
"""

# ------ modules ------
import csv
import os
from re import L
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.data_utils import labelMapping, labelOneHot, getSelectedDataset
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
tst_dat = tf.data.experimental.make_csv_dataset(file_path,
                                                batch_size=5, label_name='group')

for a in tst_dat:  # take 3 smaples
    print(a)
    break

csv_header = pd.read_csv(file_path, nrows=0).columns.tolist()

[i for i in csv_header if i not in ['PCL', 'group']]


def getSingleCsvDataset(csv_path, label_var, column_to_exclude=None,
                        batch_size=5, **kwargs):
    """
    # Purpose\n
        Write in a single CSV file into a tf.dataset object

    # Argument\n
        csv_path: str. File path. 
        label_var: str. Variable name.
        column_to_exclude: None, or list of str. A list of the variable names to exclude.
        batch_size: int. Batch size.

    # Return\n
        Two items (in following order): tf.dataset, feature list.

    # Details\n
        1. pd.read_csv is used to read in the header of the CSV file.
        2. label_var only supports one label, i.e. only binary and multi-class are supported.
    """

    # - write in only the header information -
    csv_header = pd.read_csv(csv_path, index_col=0,
                             nrows=0).columns.tolist()

    # - subset columns and establish feature list -
    if column_to_exclude is not None:
        column_to_include = [
            element for element in csv_header if element not in column_to_exclude]
        feature_list = column_to_include
    else:
        column_to_include = None
        feature_list = csv_header

    # - set up tf.dataset -
    ds = tf.data.experimental.make_csv_dataset(
        csv_path,
        batch_size=batch_size,
        label_name=label_var,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        select_columns=column_to_include,
        **kwargs)

    return ds, feature_list


tst_dat = getSingleCsvDataset(
    file_path, label_var='group', column_to_exclude=['subject', 'PCL'])

feature_names = []
for batch, label in tst_dat.take(1):
    for key, value in batch.items():
        print(f"{key:20s}: {value}")
        # feature_names.append(key)
    print()
    print(f"{'label':20s}: {label}")


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
