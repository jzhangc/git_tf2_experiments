"""
Current objectives:
small things for data loaders
"""

# ------ modules ------
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.data_utils import adjmatAnnotLoader, labelMapping, labelOneHot, getSelectedDataset
from utils.other_utils import error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
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


def tst_parse_file(filepath, target_ext=None, model_type='classification', manual_labels=None,
                   pd_labels_var_name=None):
    """
    - parse file path to get file path annotatin and, optionally, label information\n
    - set up manual label information\n
    """

    if model_type == 'classification':
        file_annot, labels = adjmatAnnotLoader(
            filepath, targetExt=target_ext)
    else:  # regeression
        if manual_labels is None:
            raise ValueError(
                'Set manual_labels when model_type=\"regression\".')
        file_annot, _ = adjmatAnnotLoader(
            filepath, targetExt=target_ext, autoLabel=False)

    if manual_labels is not None:  # update labels to the manually set array
        if isinstance(manual_labels, pd.DataFrame):
            if pd_labels_var_name is None:
                raise TypeError(
                    'Set pd_labels_var_name when manual_labels is a pd.Dataframe.')
            else:
                try:
                    labels = manual_labels[pd_labels_var_name].to_numpy(
                    )
                except Exception as e:
                    error('Manual label parsing failed.',
                          'check if pd_labels_var_name is present in the maual label data frame.')
        elif isinstance(manual_labels, np.ndarray):
            labels = manual_labels
        else:
            raise TypeError(
                'When not None, manual_labels needs to be pd.Dataframe or np.ndarray.')

        labels = manual_labels

    return file_annot, labels


def tst_get_file_annot(filepath, model_type='classification', multilabel_class=False, label_sep=None, **kwargs):
    file_annot, labels = tst_parse_file(
        filepath=filepath, model_type=model_type, **kwargs)

    if model_type == 'classification':
        if multilabel_class:
            if label_sep is None:
                raise ValueError(
                    'set label_sep for multilabel classification.')

            labels_list, lables_count, labels_map, labels_map_rev = labelMapping(
                labels=labels, sep=label_sep)
        else:
            labels_list, lables_count, labels_map, labels_map_rev = labelMapping(
                labels=labels, sep=None)
        encoded_labels = labelOneHot(labels_list, labels_map)
    else:
        encoded_labels = labels
        lables_count, labels_map_rev = None, None

    filepath_list = file_annot['path'].to_list()
    return filepath_list, encoded_labels, lables_count, labels_map_rev


def tst_map_func(filepath: tf.Tensor, label: tf.Tensor, processing=False):
    # - read file and assign label -
    fname = filepath.numpy().decode('utf-8')
    f = np.loadtxt(fname).astype('float32')
    lb = label

    # # - processing if needed -
    # if processing:
    #     f = _x_data_process(f)

    f = tf.convert_to_tensor(f, dtype=tf.float32)
    return f, lb


def tst_data_resample(total_data, n_total_sample, encoded_labels, resample_method='random'):
    """
    NOTE: regression cannot use stratified splitting\n
    NOTE: "stratified" (keep class ratios) is NOT the same as "balanced" (make class ratio=1)\n
    NOTE: "balanced" mode will be implemented at a later time\n
    NOTE: depending on how "balanced" is implemented, the if/else block could be implified\n
    """
    X_indices = np.arange(n_total_sample)
    if resample_method == 'random':
        X_train_indices, X_test_indices, _, _ = train_test_split(
            X_indices, encoded_labels, test_size=0.8, stratify=None, random_state=1)
    elif resample_method == 'stratified':
        X_train_indices, X_test_indices, _, _ = train_test_split(
            X_indices, encoded_labels, test_size=0.8, stratify=encoded_labels, random_state=1)
    else:
        raise NotImplementedError(
            '\"balanced\" resmapling method has not been implemented.')

    train_ds, train_n = getSelectedDataset(total_data, X_train_indices)
    test_ds, test_n = getSelectedDataset(total_data, X_test_indices)

    return train_ds, train_n, test_ds, test_n


def tst_generate_data(batch_size=4, cv_only=False, shuffle=True, **kwargs):
    """
    # Purpose\n
        To generate working data.\n

    # Arguments\n
        batch_size: int. Batch size for the tf.dataset batches.\n
        cv_only: bool. When True, there is no train/test split.\n
        shuffle: bool. Effective when cv_only=True, if to shuffle the order of samples for the output data.\n

    # Details\n
        - When cv_only=True, the loader returns only one tf.dataset object, without train/test split.
            In such case, further cross validation resampling can be done using followup resampling functions.
            However, it is not to say train/test split data cannot be applied with further CV operations.\n        
    """
    batch_size = batch_size
    cv_only = cv_only
    shuffle = shuffle

    # - load paths -
    filepath_list, encoded_labels, _, _ = tst_get_file_annot(**kwargs)
    total_ds = tf.data.Dataset.from_tensor_slices(
        (filepath_list, encoded_labels))
    n_total_sample = total_ds.cardinality().numpy()

    # return total_ds, n_total_sample

    # - resample data -
    if cv_only:
        train_set = total_ds.map(lambda x, y: tf.py_function(tst_map_func, [x, y, True], [tf.float32, tf.uint8]),
                                 num_parallel_calls=tf.data.AUTOTUNE)
        train_n = n_total_sample
        if shuffle:  # check this
            train_set = train_set.shuffle(seed=1)
        test_set = None
        test_n = None
    else:
        train_ds, train_n, test_ds, test_n = tst_data_resample(
            total_data=total_ds, n_total_sample=n_total_sample, encoded_labels=encoded_labels)
        train_set = train_ds.map(lambda x, y: tf.py_function(tst_map_func, [x, y, True], [tf.float32, tf.uint8]),
                                 num_parallel_calls=tf.data.AUTOTUNE)
        test_set = test_ds.map(lambda x, y: tf.py_function(tst_map_func, [x, y, True], [tf.float32, tf.uint8]),
                               num_parallel_calls=tf.data.AUTOTUNE)

    # - set up batch and prefeching -
    train_set = train_set.batch(
        batch_size).cache().prefetch(tf.data.AUTOTUNE)
    test_set = train_set.batch(
        batch_size).cache().prefetch(tf.data.AUTOTUNE) if test_set is not None else None

    return train_set, test_set


# ------ test realm ------
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, 'data/tf_data')

file_annot, labels = adjmatAnnotLoader(dat_dir, targetExt='txt')
file_annot['path'][0]

file_annot['path'].to_list()
file_annot.loc[0:1]


file_annot, labels = tst_parse_file(
    filepath=dat_dir, model_type='classification', target_ext='csv')

filepath_list, encoded_labels, lables_count, labels_map_rev = tst_get_file_annot(
    filepath=dat_dir, target_ext='txt')

tst_ds, tst_n = tst_generate_data(
    filepath=dat_dir, target_ext='txt', cv_only=False, shuffle=False)

# - below: create one hot encoding for multiclass labels -
# lb_binarizer = LabelBinarizer()
# labels_binary = lb_binarizer.fit_transform(labels)
labels_list, lables_count, labels_map, labels_map_rev = labelMapping(
    labels=labels, sep=None)  # use None to not slice strings

encoded_labels = labelOneHot(
    labels_list=labels_list, labels_map=labels_map)

# # - below: create one hot encoding for multilabel labels -
# labels_list, lables_count, labels_map, labels_map_rev = multilabelMapping(
#     labels=labels, sep='_')
# # one hot encoding
# encoded_labels = multilabelOneHot(
#     labels_list=labels_list, labels_map=labels_map)


# - below: batch load data using tf.data.Dataset API -
# expand file_annot with one hot encoded labels
file_annot[list(labels_map.keys())] = encoded_labels
file_path = file_annot['path'].to_list()

# test: load using the tf.data.Dataset API
tst_dat = tf.data.Dataset.from_tensor_slices((file_path, encoded_labels))
tst_data_size = tst_dat.cardinality().numpy()  # sample size: 250

tst_dat.shuffle()


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

for a, b in tst_dat_working:  # take 3 smaples
    print(type(a))
    print(a.shape)
    print(f'label: {b}')
    print(f)
    break


# - resampling -
X_indices = np.arange(tst_data_size)

# multiclass
X_train_indices, X_val_indices, y_train_targets, y_val_targets = train_test_split(
    X_indices, encoded_labels, test_size=0.1, stratify=encoded_labels, random_state=53)

tst_test, test_n = getSelectedDataset(tst_dat, X_val_indices)

t = (0, 1)
t2 = t[:]+(2)
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

# ------ ref ------
# - tf.dataset reference: https://cs230.stanford.edu/blog/datapipeline/ -
# - real TF2 data loader example: https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py -
# - tf.data.Dataset API example: https://medium.com/deep-learning-with-keras/tf-data-build-efficient-tensorflow-input-pipelines-for-image-datasets-47010d2e4330 -