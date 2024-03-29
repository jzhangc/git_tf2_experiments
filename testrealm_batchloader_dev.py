"""
Current objectives:
small things for data loaders
"""


# ------ modules ------
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     StratifiedShuffleSplit, train_test_split)
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, StandardScaler
from tqdm import tqdm
from collections import Counter

from utils.data_utils import (adjmatAnnotLoader, adjmatAnnotLoaderV2, getSelectedDataset,
                              labelMapping, labelOneHot, scanFiles, sameFileCheck,
                              findFilePath)
from utils.other_utils import csvPath, flatten
from utils.error_handling import error

# from skmultilearn.model_selection import iterative_train_test_split


# ------ check device ------
tf.config.list_physical_devices()


# ------ function -------
def map_func(filepath: tf.Tensor, label: tf.Tensor, processing=False):
    # - read file and assign label -
    fname = filepath.numpy().decode('utf-8')
    f = np.loadtxt(fname).astype('float32')
    lb = label
    lb.set_shape(lb.shape)

    # - processing if needed -
    if processing:
        f = f/f.max()
        # f_std = (f - f.min(axis=0)) / (f.max(axis=0) - f.min(axis=0))
        # f = f_std * (1 - 0) + 0
        # print(f.shape[:])
        f = np.reshape(f, (f.shape[0], f.shape[1], 1))
    f = tf.convert_to_tensor(f, dtype=tf.float32)
    f.set_shape(f.shape)

    return f, lb


def tst_parse_file(filepath, target_ext=None, model_type='classification', manual_labels=None,
                   pd_labels_var_name=None):
    """
    - parse file path to get file path annotation and, optionally, label information\n
    - set up manual label information\n
    """

    if model_type == 'classification':
        file_annot, labels = adjmatAnnotLoader(
            filepath, targetExt=target_ext)
    else:  # regression
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


def tst_sameFileCheck(dir, **kwargs):
    """check if dir or sub dirs contain duplicated filenames."""
    filepaths = list(scanFiles(dir, **kwargs))
    filenames = []

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        filenames.append(filename)

    dup = [k for k, v in Counter(filenames).items() if v > 1]
    if len(dup) > 0:
        print(f'Sub-directories contain duplicated file names: {dup}.')
    return dir, dup, filepaths, filenames


def tst_findFilePath(tgt_filename, dir):
    """find specific file in a dir and return full path"""
    for rootDir, dirNames, filenames in os.walk(dir):
        if tgt_filename in filenames:
            filePath = os.path.join(rootDir, tgt_filename)
            yield filePath


def tst_adjmatAnnotLoader(dir, targetExt=None, autoLabel=True, annotFile=None, fileNameVar=None, labelVar=None):
    """
    # Purpose\n
        Scan and extract file paths (export as pandas data frame).\n
        Optionally, the function can also construct file labels using
            folder names and exports as a numpy array.\n

    # Arguments\n
        path: str. The root directory path to scan.\n
        targetExt: str. Optionally set target file extension to extract.\n
        autoLabel: bool. If to automatically construct file labels using folder names.\n
        annotFile: str or None. Required when autoLabel=False, a csv file for file names and labels.\n
        fileNameVar: str or None. Required when autoLabel=False, variable name in annotFile for file names.\n
        labelVar: str or None. Required when autoLabel=False, variable nam ein annotFile for lables.\n

    # Return\n
        Pandas data frame containing all file paths. Optionally, a numpy array with all
            file labels. Order: file_path, labels.\n

    # Details\n
        - When targetExt=None, the function scans root and sub directories.\n
        - The targetExt string should exclude the "." symbol, e.g. 'txt' instead of '.txt'.\n
        - The function returns None for "labels" when autoLabel=False.\n
        - When autoLabel=False, the sub folder is not currently supported.
            Sub folder support is not impossible. It is just too complicated to implement in a timely fashion. 
            This means all data files should be in one folder, i.e. dir.\n
        - When autoLabel=False, the CSV file should at least two columens, one for file name and one for the corresponding labels.\n
    """
    # -- check arguments for autoLabel=False --
    if autoLabel == False:
        if (any(annotF is None for annotF in [annotFile, fileNameVar, labelVar])):
            raise ValueError(
                'Set annotFile, fileNameVar and labelVar when autoLabel=False.')
        else:
            annotFile_path = os.path.normpath(
                os.path.abspath(os.path.expanduser(annotFile)))

            if os.path.isfile(annotFile_path):
                # return full_path
                _, file_ext = os.path.splitext(annotFile_path)
                if file_ext != '.csv':
                    raise ValueError('annotFile needs to be .csv type.')
            else:
                raise ValueError('Invalid annotFile or annotFile not found.')

            annot_pd = pd.read_csv(annotFile_path, engine='python')
            if not all(annot_var in annot_pd.columns for annot_var in [fileNameVar, labelVar]):
                raise ValueError(
                    'fileNameVar and labelVar should both be present in the annotFile')

    # -- scan files --
    adjmat_paths = list(scanFiles(dir, validExts=targetExt))
    file_annot = pd.DataFrame()

    # -- labels --
    if autoLabel:
        labels = []
        for i, adjmat_path in tqdm(enumerate(adjmat_paths), total=len(adjmat_paths)):
            # os.path.sep returns "/" which is used for str.split
            label = adjmat_path.split(os.path.sep)[-2]
            file_annot.loc[i, 'path'] = adjmat_path
            labels.append(label)

        labels = np.array(labels)
        file_annot['label'] = labels
    else:  # manual label
        # Check duplicated files
        dup = sameFileCheck(dir=dir, validExts=targetExt)
        if len(dup) > 0:
            raise ValueError(
                f'File duplicates found when autoLabel=False: {dup}')

        labels = annot_pd[labelVar].to_numpy()
        manual_filenames = annot_pd[fileNameVar].to_list()
        manual_filename_paths = []
        for manual_filename in tqdm(manual_filenames):
            manual_filename_paths.append(
                list(findFilePath(manual_filename, dir)))
        manual_filename_paths = flatten(manual_filename_paths)

        file_annot['filename'] = annot_pd[fileNameVar]
        file_annot['path'] = manual_filename_paths
        file_annot['label'] = annot_pd[labelVar]

    return file_annot, labels


# ------ test realm ------
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, 'data/tf_data')

file_annot, labels = adjmatAnnotLoader(
    dat_dir, targetExt='txt', autoLabel=True)
file_annot['path'][0]


file_annot, labels = tst_adjmatAnnotLoader(
    dat_dir, targetExt='txt', autoLabel=True)

file_annot, labels = adjmatAnnotLoaderV2(
    dat_dir, targetExt='txt', autoLabel=True)

file_annot['path'].to_list()
file_annot.loc[0:1]

_, _, _, filenames = tst_sameFileCheck('./data', validExts='txt')
dup = tst_sameFileCheck('./data', validExts='txt')
tst_pd = pd.DataFrame()
tst_pd['filename'] = filenames
tst_pd['label'] = labels
tst_pd.to_csv(os.path.join(dat_dir, 'tst_annot.csv'), index=False)


file_annot, labels = adjmatAnnotLoaderV2(
    dat_dir, targetExt='txt', autoLabel=False, annotFile='./data/tst_annot.csv', fileNameVar='filename', labelVar='label')

file_annot['path'].to_list()
file_annot.loc[0:1]


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


def tst_set_shape_foo(f: tf.Tensor, lb: tf.Tensor):
    f.set_shape(f.shape)
    lb.set_shape(lb.shape)
    return f, lb


tst_dat_working = tst_dat.map(lambda x, y: tf.py_function(map_func, [x, y, True], [tf.float32, tf.uint8]),
                              num_parallel_calls=tf.data.AUTOTUNE)
tst_dat_working = tst_dat_working.map(tst_set_shape_foo)


for a, b in tst_dat_working:  # take 3 smaples
    print(type(a))
    print(a.shape)
    a.set_shape([None, None, a.shape[-1]])
    print(a.shape)
    print(b.shape)
    # print(f'label: {b}')
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


mylist = ['a.txt', 'a.txt', 25, 20]
[k for k, v in Counter(mylist).items() if v > 1]


# ------ ref ------
# - tf.dataset reference: https://cs230.stanford.edu/blog/datapipeline/ -
# - real TF2 data loader example: https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py -
# - tf.data.Dataset API example: https://medium.com/deep-learning-with-keras/tf-data-build-efficient-tensorflow-input-pipelines-for-image-datasets-47010d2e4330 -


# file_annot, labels = tst_parse_file(
#     filepath=dat_dir, model_type='classification', target_ext='csv')

# filepath_list, encoded_labels, lables_count, labels_map_rev = tst_get_file_annot(
#     filepath=dat_dir, target_ext='txt')

# tst_ds, tst_n = tst_generate_data(
#     filepath=dat_dir, target_ext='txt', cv_only=False, shuffle=False)
