"""
Current objectives:
small things for data loaders
"""

# ------ modules ------
import os
import pandas as pd
import numpy as np
import tensorflow as tf
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
    selected_ds = ds \
        .enumerate() \
        .filter(is_index_in) \
        .map(drop_index)
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
tst_dat = tf.data.Dataset.from_tensor_slices(file_path)
tst_data_size = tst_dat.cardinality().numpy()  # sample size: 10
for i, a in enumerate(tst_dat.take(3)):  # take 3 smaples
    fname = a.numpy().decode("utf-8")
    a_dat = np.loadtxt(fname).astype('float32')
    b = encoded_labels[i]
    print(fname)
    print(f'label: {b}')
    print(a_dat)
    # break


tst_mtx = np.loadtxt(file_path[0]).astype(dtype='float32')
tst_mtx = tst_mtx/tst_mtx.max()
tst_mtx.shape
tst_mtx = tst_mtx.reshape((tst_mtx.shape[0], tst_mtx.shape[1], 1))
tst_mtx.shape
tst_mtx_label = encoded_labels[0]

X_indices = range(tst_data_size)
X_train_indices, X_val_indices, y_train_targets, y_val_targets = train_test_split(
    X_indices, encoded_labels, test_size=0.1, stratify=encoded_labels, random_state=53)

# multilabel classificatin cannot be used for stratified split
stk = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)

kf = KFold(n_splits=10, shuffle=True, random_state=12)
for train_idx, test_idx in kf.split(X_indices, encoded_labels):
    print(train_idx)
    print(test_idx)

# ------ ref ------
# - initial: https://debuggercafe.com/creating-efficient-image-data-loaders-in-pytorch-for-deep-learning/ -
# - tf.dataset reference: https://cs230.stanford.edu/blog/datapipeline/ -
# - real TF2 data loader example: https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py -
# - tf.data.Dataset API example: https://medium.com/deep-learning-with-keras/tf-data-build-efficient-tensorflow-input-pipelines-for-image-datasets-47010d2e4330 -
