"""
Current objectives:
small things for data loaders
"""

# ------ modules ------
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow._api.v2 import data
from utils.data_utils import adjmat_annot_loader, multilabel_mapping, multilabel_one_hot
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


# ------ function -------
def get_selected_dataset(ds, X_indices_np):
    """
    stolen from https://www.kaggle.com/tt195361/splitting-tensorflow-dataset-for-validation
    """
    # Make a tensor of type tf.int64 to match the one by Dataset.enumerate().
    X_indices_ts = tf.constant(X_indices_np, dtype=tf.int64)

    def is_index_in(index, rest):
        # Returns True if the specified index value is included in X_indices_ts.
        #
        # '==' compares the specified index value with each values in X_indices_ts.
        # The result is a boolean tensor, looks like [ False, True, ..., False ].
        # reduce_any() returns Ture if True is included in the specified tensor.
        return tf.math.reduce_any(index == X_indices_ts)

    def drop_index(index, rest):
        return rest

    # Dataset.enumerate() is similter to Python's enumerate().
    # The method adds indices to each elements. Then, the elements are filtered
    # by using the specified indices. Finally unnecessary indices are dropped.
    selected_ds = ds.enumerate().filter(is_index_in).map(drop_index)
    return selected_ds


# ------ test realm ------
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, 'data/tf_data')

file_annot, labels = adjmat_annot_loader(dat_dir, targetExt='txt')
file_annot['path'][0]
file_annot.loc[0:1]

# - below: create one hot encoding for multiclass labels -
# lb_binarizer = LabelBinarizer()
# labels_binary = lb_binarizer.fit_transform(labels)
labels_list, lables_count, labels_map, labels_map_rev = multilabel_mapping(
    labels=labels, sep=None)  # use None to not slice strings

encoded_labels = multilabel_one_hot(
    labels_list=labels_list, labels_map=labels_map)

# # - below: create one hot encoding for multilabel labels -
# labels_list, lables_count, labels_map, labels_map_rev = multilabel_mapping(
#     labels=labels, sep='_')
# # one hot encoding
# encoded_labels = multilabel_one_hot(
#     labels_list=labels_list, labels_map=labels_map)


# - below: batch load data using tf.data.Dataset API -
# expand file_annot with one hot encoded labels
file_annot[list(labels_map.keys())] = encoded_labels
file_path = file_annot['path'].to_list()

# test: load using the tf.data.Dataset API
tf.config.list_physical_devices()
tst_dat = tf.data.Dataset.from_tensor_slices((file_path, encoded_labels))
tst_data_size = tst_dat.cardinality().numpy()  # sample size: 250

for a, b in tst_dat.take(3):  # take 3 smaples
    fname = a.numpy().decode('utf-8')
    flabel = b.numpy()
    # f = np.loadtxt(fname).astype('float32')
    f = np.loadtxt(fname).astype('float32')
    print(fname)
    print(f'label: {flabel}')
    print(f)
    break


# @tf.function
def map_func(filepath: tf.Tensor, label: tf.Tensor):
    fname = filepath.numpy().decode('utf-8')
    f = np.loadtxt(fname).astype('float32')
    lb = label.numpy()

    f = tf.convert_to_tensor(f, dtype=tf.float32)
    lb = tf.convert_to_tensor(lb, dtype=tf.uint8)
    # def processing(f, lb):
    #     of = f/f.max()
    #     ob = lb
    #     return of, ob

    # return tf.py_function(processing, [f, lb], [tf.float32, tf.int8])
    return f, lb


tst_dat_mapped = tst_dat.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)


tst_mtx = np.loadtxt(file_path[0]).astype(dtype='float32')
tst_mtx = tst_mtx/tst_mtx.max()
tst_mtx.shape
tst_mtx = tst_mtx.reshape((tst_mtx.shape[0], tst_mtx.shape[1], 1))
tst_mtx.shape
tst_mtx_label = encoded_labels[0]

X_indices = range(tst_data_size)
X_train_indices, X_val_indices, y_train_targets, y_val_targets = train_test_split(
    X_indices, encoded_labels, test_size=0.1, stratify=encoded_labels, random_state=53)

tst_train = get_selected_dataset(tst_dat, X_train_indices)


for i, a in enumerate(tst_train.take(3)):  # take 3 smaples
    fname = a.numpy().decode("utf-8")
    a_dat = np.loadtxt(fname).astype('float32')
    b = y_train_targets[0:3][i]
    print(fname)
    print(f'label: {b}')
    print(a_dat)
    # break


# multilabel classificatin cannot be used for stratified split
stk = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)

kf = KFold(n_splits=10, shuffle=True, random_state=12)
for train_idx, test_idx in kf.split(X_indices, encoded_labels):
    print(train_idx)
    print(test_idx)

# ------ ref ------
# - tf.dataset reference: https://cs230.stanford.edu/blog/datapipeline/ -
# - real TF2 data loader example: https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py -
# - tf.data.Dataset API example: https://medium.com/deep-learning-with-keras/tf-data-build-efficient-tensorflow-input-pipelines-for-image-datasets-47010d2e4330 -
