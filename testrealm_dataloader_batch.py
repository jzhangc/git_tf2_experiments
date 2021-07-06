#!/usr/bin/env python3
"""
Current objectives:
[ ] Test argparse
    [ ] Add groupped arguments
[ ] Test output directory creation
[ ] Test file reading
[ ] Test file processing
    [ ] normalization and scalling
    [ ] converting to numpy arrays

NOTE
All the argparser inputs are loaded from method arguments, making the class more portable, i.e. not tied to
the application.
"""
# ------ import modules ------
import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.other_utils import error, warn, flatten, addBoolArg, outputDir, csvPath, colr
from utils.data_utils import adjmatAnnotLoader, labelMapping, labelOneHot, getSelectedDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ------ system classes ------
class AppArgParser(argparse.ArgumentParser):
    """
    This is a sub class to argparse.ArgumentParser.
    Purpose
        The help page will display when (1) no argumment was provided, or (2) there is an error
    """

    def error(self, message, *lines):
        string = "\n{}ERROR: " + message + "{}\n" + \
            "\n".join(lines) + ("{}\n" if lines else "{}")
        print(string.format(colr.RED_B, colr.RED, colr.ENDC))
        self.print_help()
        sys.exit(2)


# ------ GLOBAL variables -------
__version__ = '0.1.0'
AUTHOR = 'Jing Zhang, PhD'
DESCRIPITON = """
{}--------------------------------- Description -------------------------------------------
Data loader for batch adjacency matrix CSV file data table for deep learning.
The loaded CSV files are stored in a 3D numpy array, with size: .
This loader also handels the following:
    1. Data resampling, e.g. traning/test split, cross validation
    2. Data normalization
-----------------------------------------------------------------------------------------{}
""".format(colr.YELLOW, colr.ENDC)

# ------ augment definition ------
# - set up parser and argument groups -
parser = AppArgParser(description=DESCRIPITON,
                      epilog='Written by: {}. Current version: {}\n\r'.format(
                          AUTHOR, __version__),
                      formatter_class=argparse.RawTextHelpFormatter)
parser._optionals.title = "{}Help options{}".format(colr.CYAN_B, colr.ENDC)

arg_g1 = parser.add_argument_group(
    '{}Input and output{}'.format(colr.CYAN_B, colr.ENDC))
arg_g2 = parser.add_argument_group(
    '{}Resampling and normalization{}'.format(colr.CYAN_B, colr.ENDC))
arg_g3 = parser.add_argument_group(
    '{}Modelling{}'.format(colr.CYAN_B, colr.ENDC))
arg_g4 = parser.add_argument_group('{}Other{}'.format(colr.CYAN_B, colr.ENDC))

add_g1_arg = arg_g1.add_argument
add_g2_arg = arg_g2.add_argument
add_g3_arg = arg_g3.add_argument
add_g4_arg = arg_g4.add_argument

# - add arugments to the argument groups -
# g1: inpout and ouput
add_g1_arg('file', nargs=1, type=csvPath,
           help='One and only one input CSV file. (Default: %(default)s)')

add_g1_arg('-s', '--sample_id_var', type=str, default=None,
           help='str. Vairable name for sample ID. NOTE: only needed with single file processing. (Default: %(default)s)')
add_g1_arg('-a', '--annotation_vars', type=str, nargs="+", default=[],
           help='list of str. names of the annotation columns in the input data, excluding the outcome variable. (Default: %(default)s)')
# add_g1_arg('-cl', '--n_classes', type=int, default=None,
#            help='int. Number of class for classification models. (Default: %(default)s)')
add_g1_arg('-y', '--outcome_var', type=str, default=None,
           help='str. Vairable name for outcome. NOTE: only needed with single file processing. (Default: %(default)s)')
addBoolArg(parser=arg_g1, name='y_scale', input_type='flag', default=False,
           help='str. If to min-max scale outcome for regression study. (Default: %(default)s)')
add_g1_arg('-o', '--output_dir', type=outputDir,
           default='.',
           help='str. Output directory. NOTE: not an absolute path, only relative to working directory -w/--working_dir.')

# g4: others
addBoolArg(parser=arg_g4, name='verbose', input_type='flag', default=False,
           help='Verbose or not. (Default: %(default)s)')

args = parser.parse_args()

# check arguments. did not use parser.error as error() has fancy colours
print(args)
if args.man_split and len(args.holdout_samples) < 1:
    error('Set -t/--holdout_samples when --man_split was set.')


# ------ loacl classes ------
class DataLoader(object):
    """
    # Purpose\n
        Data loading class.
    # Methods\n
        __init__: load data and other information from argparser, as well as class label encoding for classification study
    # Details\n
        This class is designed to load the data and set up data for training LSTM models.
        This class uses the custom error() function. So be sure to load it.
    # Class property\n
        modelling_data: dict. data for model training. data is split if necessary.
            No data splitting for the "CV only" mode.
            returns a dict object with 'training' and 'test' items
    """

    def __init__(self, filepath,
                 new_shape=None,
                 manual_labels=None, label_sep=None, pd_labels_var_name=None,
                 target_ext=None,
                 model_type='classification', multilabel=False,
                 x_scaling="none", x_min_max_range=(0, 1),
                 resmaple_method="random",
                 batch_size=None,
                 training_percentage=0.8,
                 shuffle=True,
                 cross_validation=False, k=10,
                 verbose=True, random_state=1):
        """
        # Arguments\n
            resample_method: str. options: "random", "stratified" and "balanced".
            x_scaling: str. Options are "none", "max", or "minmax". Default is None (i.e. not scaling data).
            x_min_max_range: two tuple. set when x_scaling="minmax", (min, max) range.

        # Details\n
            1. resample_method is automatically set to "random" when model_type='regression'.
        """
        # model information
        self.model_type = model_type
        self.multilabel = multilabel
        self.filepath = filepath
        self.target_ext = target_ext
        self.manual_labels = manual_labels
        self.pd_labels_var_name = pd_labels_var_name
        self.label_sep = label_sep
        self.new_shape = new_shape

        # processing
        self.x_scaling = x_scaling
        self.x_min_max_range = x_min_max_range
        self.batch_size = batch_size

        # resampling
        self.resample_method = resmaple_method
        self.train_percentage = training_percentage
        self.test_percentage = 1 - training_percentage
        self.shuffle = shuffle
        self.cross_validation = cross_validation
        self.cv_k = k

        # random state and other settings
        self.rand = random_state
        self.verbose = verbose

    def _parse_file(self):
        """
        1. parse file path to get file path annotatin and, optionally, label information
        2. set up manual label information
        """

        if self.model_type == 'classification':
            file_annot, labels = adjmatAnnotLoader(
                self.filepath, targetExt=self.target_ext)
        else:  # regeression
            file_annot, _ = adjmatAnnotLoader(
                self.filepath, targetExt=self.target_ext, autoLabel=False)
            if self.manual_labels is None:
                raise TypeError(
                    'Set manual_labels when model_type=\"regression\".')

        if self.manual_labels is not None:  # update labels to the manually set array
            if isinstance(self.manual_labels, pd.DataFrame):
                if self.pd_labels_var_name is None:
                    raise TypeError(
                        'Set pd_labels_var_name when manual_labels is a pd.Dataframe.')
                else:
                    labels = self.manual_labels[self.pd_labels_var_name].to_numpy(
                    )
            elif isinstance(self.manual_labels, np.ndarray):
                labels = self.manual_labels
            else:
                raise TypeError(
                    'When not None, manual_labels needs to be pd.Dataframe or np.ndarray.')

            labels = self.manual_labels

        return file_annot, labels

    def _get_file_annot(self):
        file_annot, labels = self._parse_file()

        if self.model_type == 'classification':
            if self.multilabel:
                labels_list, lables_count, labels_map, labels_map_rev = labelMapping(
                    labels=labels, sep=self.label_sep)
            else:
                labels_list, lables_count, labels_map, labels_map_rev = labelMapping(
                    labels=labels, sep=None)
            encoded_labels = labelOneHot(labels_list, labels_map)
        else:
            encoded_labels = labels
            lables_count, labels_map_rev = None, None

        filepath_list = file_annot['path'].to_list()

        return filepath_list, encoded_labels, lables_count, labels_map_rev

    def _x_data_process(self, x_array):
        """NOTE: reshaping to (_, _, 1) is mandatory"""
        # - variables -
        if isinstance(x_array, np.ndarray):  # this check can be done outside of the classs
            X = x_array
        else:
            raise TypeError('data processing function should be a np.ndarray.')

        if self.x_scaling == 'max':
            X = X/X.max()
        elif self.x_scaling == 'minmax':
            Min = self.x_min_max_range[0]
            Max = self.x_min_max_range[1]
            X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            X = X_std * (Max - Min) + Min

        if self.new_shape is not None:  # reshape
            X = np.reshape(X, self.new_shape)
        else:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X

    def _map_func(self, filepath: tf.Tensor, label: tf.Tensor, processing=False):
        # - read file and assign label -
        fname = filepath.numpy().decode('utf-8')
        f = np.loadtxt(fname).astype('float32')
        lb = label

        # - processing if needed -
        if processing:
            f = self._x_data_process(f)

        f = tf.convert_to_tensor(f, dtype=tf.float32)
        return f, lb

    def _data_resample(self, total_data, n_total_sample):
        """
        NOTE: regression cannot use stratified splitting
        NOTE: "stratified" (keep class ratios) is NOT the same as "balanced" (make class ratio=1)
        NOTE: "balanced" mode will be implemented at a later time
        NOTE: depending on how "balanced" is implemented, the if/else block could be implified
        """
        _, encoded_labels, _, _ = self._get_file_annot()
        X_indices = np.arange(n_total_sample)
        if self.resample_method == 'random':
            X_train_indices, X_test_indices, _, _ = train_test_split(
                X_indices, encoded_labels, test_size=self.test_percentage, stratify=None, random_state=self.rand)
        elif self.resample_method == 'stratified':
            X_train_indices, X_test_indices, _, _ = train_test_split(
                X_indices, encoded_labels, test_size=self.test_percentage, stratify=encoded_labels, random_state=self.rand)
        else:
            raise NotImplementedError(
                '\"balanced\" resmapling method has not been implemented.')

        train_ds, train_n = getSelectedDataset(total_data, X_train_indices)
        test_ds, test_n = getSelectedDataset(total_data, X_test_indices)

        return train_ds, train_n, test_ds, test_n

    def load_data(self, batch_size, shuffle=False):

        return None


# below: ad-hoc testing
mydata = DataLoader(file='./data/test_dat.csv', outcome_var='PCL', annotation_vars=['subject', 'group'], sample_id_var='subject',
                    holdout_samples=None, minmax=True, x_standardize=True,
                    model_type='regression', cv_only=False, man_split=False, training_percentage=0.8, random_state=1, verbose=True)

# ------ process/__main__ statement ------
# if __name__ == '__main__':
#     mydata = DataLoader(file=args.file[0],
#                         outcome_var=args.outcome_var, annotation_vars=args.annotation_vars, sample_id_var=args.sample_id_var,
#                         holdout_samples=args.holdout_samples,
#                         model_type=args.model_type, cv_only=args.cv_only, man_split=args.man_split, training_percentage=args.training_percentage,
#                         random_state=args.random_state, verbose=args.verbose)
