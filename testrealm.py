"""
Current objectives:
small things for data loaders
"""

# ------ modules ------
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.data_utils import adjmat_annot_loader, label_mapping, label_one_hot
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
# from skmultilearn.model_selection import iterative_train_test_split


# ------ function -------
def get_selected_dataset(ds, X_indices_np):
    """
    modified from https://www.kaggle.com/tt195361/splitting-tensorflow-dataset-for-validation
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

    # calculate number of samples
    # cardinality would not work on "selected_ds"
    n = 0
    for e in selected_ds:
        n += 1

    return selected_ds, n


def map_func(filepath: tf.Tensor, label: tf.Tensor, processing=False):
    # - read file and assign label -
    fname = filepath.numpy().decode('utf-8')
    f = np.loadtxt(fname).astype('float32')
    lb = label

    # - processing if needed -
    if processing:
        f = f/f.max()
    f = tf.convert_to_tensor(f, dtype=tf.float32)

    return f, lb


# ------ test realm ------
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, 'data/tf_data')

file_annot, labels = adjmat_annot_loader(dat_dir, targetExt='txt')
file_annot['path'][0]
file_annot.loc[0:1]

# - below: create one hot encoding for multiclass labels -
# lb_binarizer = LabelBinarizer()
# labels_binary = lb_binarizer.fit_transform(labels)
labels_list, lables_count, labels_map, labels_map_rev = label_mapping(
    labels=labels, sep=None)  # use None to not slice strings

encoded_labels = label_one_hot(
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

    # f = np.loadtxt(fname).astype('float32')
    # f = tf.convert_to_tensor(f, dtype=tf.float32)
    f, lb = map_func(a, b, processing=True)

    print(type(a))
    print(fname)
    print(f'label: {lb}')
    print(f)
    break

tst_dat_working = tst_dat.map(lambda x, y: tf.py_function(map_func, [x, y, True], [tf.float32, tf.uint8]),
                              num_parallel_calls=tf.data.AUTOTUNE)

# - resampling -
X_indices = np.arange(tst_data_size)

# multiclass
X_train_indices, X_val_indices, y_train_targets, y_val_targets = train_test_split(
    X_indices, encoded_labels, test_size=0.1, stratify=encoded_labels, random_state=53)

tst_test, test_n = get_selected_dataset(tst_dat, X_val_indices)


# # multilabel
# X_indices_reshap = X_indices.reshape((len(X_indices), 1))
# X_train_indices, y_train_targets, X_val_indices, y_val_targets = iterative_train_test_split(
#     X_indices_reshap, encoded_labels, test_size=0.1)


# multilabel classificatin cannot be used for stratified split
stk = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)

kf = KFold(n_splits=10, shuffle=True, random_state=12)
for train_idx, test_idx in kf.split(X_indices, encoded_labels):
    print(train_idx)
    print(test_idx)


X, y, _, _ = load_dataset('scene', 'undivided')


# ------ ref ------
# - tf.dataset reference: https://cs230.stanford.edu/blog/datapipeline/ -
# - real TF2 data loader example: https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py -
# - tf.data.Dataset API example: https://medium.com/deep-learning-with-keras/tf-data-build-efficient-tensorflow-input-pipelines-for-image-datasets-47010d2e4330 -
