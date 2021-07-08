#!/usr/bin/env python3
"""
Current objectives:
[x] Test argparse
    [x] Add groupped arguments
[ ] Test output directory creation
[x] Test file reading
[x] Test file processing
    [x] normalization and scalling
    [x] converting to numpy arrays
[x] use convert to tf.dataset

NOTE
All the argparser inputs are loaded from method arguments, making the class more portable, i.e. not tied to
the application.
"""
# ------ import modules ------
import argparse
import os
import sys
# import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.other_utils import error, warn, addBoolArg, csvPath, outputDir, colr
from utils.data_utils import labelMapping, labelOneHot
from sklearn.model_selection import train_test_split


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


# ------ custom functions ------
# below: a lambda funciton to flatten the nested list into a single list


# ------ GLOBAL variables -------
__version__ = '0.1.0'
AUTHOR = 'Jing Zhang, PhD'
DESCRIPITON = """
{}--------------------------------- Description -------------------------------------------
Data loader for single CSV file data table for deep learning. 
The loaded CSV file is stored in numpy arrays. 
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
           help='list of str. names of the annotation columns in the input data, excluding the label variable. (Default: %(default)s)')
# add_g1_arg('-cl', '--n_classes', type=int, default=None,
#            help='int. Number of class for classification models. (Default: %(default)s)')
add_g1_arg('-y', '--label_var', type=str, default=None,
           help='str. Vairable name for label. NOTE: only needed with single file processing. (Default: %(default)s)')
add_g1_arg('-c', '--label_string_sep', type=str, default=None,
           help='str. Separator to separate label string, to create multilabel labels. (Default: %(default)s)')
addBoolArg(parser=arg_g1, name='y_scale', input_type='flag', default=False,
           help='str. If to min-max scale label for regression study. (Default: %(default)s)')
add_g1_arg('-o', '--output_dir', type=outputDir,
           default='.',
           help='str. Output directory. NOTE: not an absolute path, only relative to working directory -w/--working_dir.')

# g2: resampling and normalization
addBoolArg(parser=arg_g2, name='cv_only', input_type='flag',
           help='If to do cv_only mode for training, i.e. no holdout test split. (Default: %(default)s)',
           default=False)
add_g2_arg('-v', '--cv_type', type=str,
           choices=['kfold', 'LOO', 'monte'], default='kfold',
           help='str. Cross validation type. Default is \'kfold\'')
addBoolArg(parser=arg_g2, name='man_split', input_type='flag',
           help='Manually split data into training and test sets. When set, the split is on -s/--sample_id_var. (Default: %(default)s)',
           default=False)
add_g2_arg('-t', '--holdout_samples', nargs='+', type=str, default=[],
           help='str. Sample IDs selected as holdout test group when --man_split was set. (Default: %(default)s)')
add_g2_arg('-p', '--training_percentage', type=float, default=0.8,
           help='num, range: 0~1. Split percentage for training set when --no-man_split is set. (Default: %(default)s)')
add_g2_arg('-m', '--resample_method', type=str,
           choices=['random', 'stratified', 'balanced'], default='random',
           help='str. training-test split method. (Default: %(default)s)')
add_g2_arg('-r', '--random_state', type=int,
           default=1, help='int. Random state. (Default: %(default)s)')
addBoolArg(parser=arg_g2, name='x_standardize', input_type='flag',
           default='False',
           help='If to apply z-score stardardization for x. (Default: %(default)s)')
addBoolArg(parser=arg_g2, name='minmax', input_type='flag',
           default='False',
           help='If to apply min-max normalization for x and, if regression, y (to range 0~1). (Default: %(default)s)')

# g3: modelling
add_g3_arg('-m', '--model_type', type=str, choices=['regression', 'classification'],
           default='classifciation',
           help='str. Model type. Options: \'regression\' and \'classification\'. (Default: %(default)s)')

# g4: others
addBoolArg(parser=arg_g4, name='verbose', input_type='flag', default=False,
           help='Verbose or not. (Default: %(default)s)')

args = parser.parse_args()

# check arguments. did not use parser.error as error() has fancy colours
print(args)
if not args.sample_id_var:
    error('-s/--sample_id_var missing.',
          'Be sure to set the following: -s/--sample_id_var, -y/--label_var, -a/--annotation_vars')

if not args.label_var:
    error('-y/--label_var flag missing.',
          'Be sure to set the following: -s/--sample_id_var, -y/--label_var, -a/--annotation_vars')

if len(args.annotation_vars) < 1:
    error('-a/--annotation_vars missing.',
          'Be sure to set the following: -s/--sample_id_var, -y/--label_var, -a/--annotation_vars')

if args.man_split and len(args.holdout_samples) < 1:
    error('Set -t/--holdout_samples when --man_split was set.')

if args.cv_type == 'monte':
    if args.monte_test_rate < 0.0 or args.monte_test_rate > 1.0:
        error('-mt/--monte_test_rate should be between 0.0 and 1.0.')


# ------ loacl classes ------
class singleCsvMemLoader(object):
    """
    # Purpose\n
        In memory data loader for single file CSV.
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

    def __init__(self, file,
                 label_var, annotation_vars, sample_id_var,
                 model_type,
                 cv_only,
                 minmax,
                 x_standardize,
                 man_split, holdout_samples, training_percentage,
                 resample_method='random',
                 label_string_sep=None, random_state=1, verbose=True):
        """
        # Arguments\n
            file: str. complete input file path. "args.file[0]" from argparser]
            label_var: str. variable nanme for label. Only one is accepted for this version. "args.label_var" from argparser
            annotation_vars: list of strings. Column names for the annotation variables in the input dataframe, EXCLUDING label variable.
                "args.annotation_vars" from argparser
            sample_id_var: str. variable used to identify samples. "args.sample_id_var" from argparser
            model_type: str. model type, classification or regression
            n_classes: int. number of classes when model_type='classification'
            cv_only: bool. If to split data into training and holdout test sets. "args.cv_only" from argparser
            man_split: bool. If to use manual split or not. "args.man_split" from argparser
            holdout_samples: list of strings. sample IDs for holdout sample, when man_split=True. "args.holdout_samples" from argparser
            training_percentage: float, betwen 0 and 1. percentage for training data, when man_split=False. "args.training_percentage" from argparser
            random_state: int. random state
            verbose: bool. verbose. "args.verbose" from argparser
        # Public class attributes\n
            Below are attributes read from arguments
                self.model_type
                self.n_classes
                self.file
                self.label_var
                self.annotation_vars
                self.cv_only
                self.holdout_samples
                self.training_percentage
                self.rand: int. random state
            self.y_var: single str list. variable nanme for label
            self.filename: str. input file name without extension
            self.raw: pandas dataframe. input data
            self.raw_working: pands dataframe. working input data
            self.complete_annot_vars: list of strings. column names for the annotation variables in the input dataframe, INDCLUDING label varaible
            self.n_features: int. number of features
            self.le: sklearn LabelEncoder for classification study
            self.label_mapping: dict. Class label mapping codes, when model_type='classification'
        # Private class attributes (excluding class properties)\n
            self._basename: str. complete file name (with extension), no path
            self._n_annot_col: int. number of annotation columns
        """
        # - random state and other settings -
        self.rand = random_state
        self.verbose = verbose

        # - model and data info -
        self.model_type = model_type
        # convert to a list for trainingtestSpliterFinal() to use
        self.label_var = label_var
        self.label_sep = label_string_sep
        self.annotation_vars = annotation_vars
        self.y_var = [self.label_var]  # might not need this anymore
        self.complete_annot_vars = self.annotation_vars + self.y_var
        self._n_annot_col = len(self.complete_annot_vars)

        # - args.file is a list. so use [0] to grab the string -
        self.file = file
        self._basename, self._file_ext = os.path.splitext(file)

        # - parse file -
        self.raw = pd.read_csv(self.file, engine='python')
        self.raw_working = self.raw.copy()  # value might be changed
        self.n_features = int(
            (self.raw_working.shape[1] - self._n_annot_col))  # pd.shape[1]: ncol
        self.total_n = self.raw_working.shape[0]
        if model_type == 'classification':
            self.n_class = self.raw[label_var].nunique()
        else:
            self.n_class = None

        self.x = self.raw_working[self.raw_working.columns[
            ~self.raw_working.columns.isin(self.complete_annot_vars)]].to_numpy()
        self.labels = self.raw_working[self.label_var].to_numpy()

        # - resampling settings -
        self.cv_only = cv_only
        self.resample_method = resample_method
        self.sample_id_var = sample_id_var
        self.holdout_samples = holdout_samples
        self.training_percentage = training_percentage
        self.test_percentage = 1 - training_percentage
        self.x_standardize = x_standardize
        self.minmax = minmax

        # call setter here
        if verbose:
            print('Setting up modelling data...', end=' ')
        self.modelling_data = man_split
        if verbose:
            print('done!')

    def _label_onehot_encode(self, labels):
        """one hot encoding for labels. labels: shoud be a np.ndarray"""
        labels_list, labels_count, labels_map, labels_map_rev = labelMapping(
            labels, sep=self.label_sep)

        onehot_encoded = labelOneHot(labels_list, labels_map)

        return onehot_encoded, labels_count, labels_map_rev

    @property
    def modelling_data(self):
        # print("called getter") # for debugging
        return self.train_ds, self.test_ds, self.train_n, self.test_n

    @modelling_data.setter
    def modelling_data(self, man_split):
        """
        Private attributes for the property\n
            _m_data: dict. output dictionary
            _training: pandas dataframe. data for model training.
            _test: pandas dataframe. holdout test data. Only available without the "--cv_only" flag
        Return\n
            1. self.training_y_scaler
            2. tf.datasets for training and (if applicable) test sets
        """
        # print("called setter") # for debugging
        if self.model_type == 'classification':  # one hot encoding
            self.labels_working, self.labels_count, self.labels_rev = self._label_onehot_encode(
                self.labels)
        else:
            self.labels_working, self.labels_count, self.labels_rev = self.labels, None, None

        # - data resampling -
        if self.cv_only:  # only training is stored
            # training set prep
            self._training_x = self.x
            self._training_y = self.labels_working

            # test set prep
            self._test_x, self._test_y = None, None
            self.training_y_scaler = None
            self.test_n = None
        else:  # training and holdout test data split
            # self._training, self._test, _, _, self.training_y_scaler = trainingtestSpliterFinal(data=self.raw_working, random_state=self.rand,
            #                                                                                     model_type=self.model_type,
            #                                                                                     man_split=man_split, man_split_colname=self.sample_id_var,
            #                                                                                     man_split_testset_value=self.holdout_samples,
            #                                                                                     x_standardization=self.x_standardize,
            #                                                                                     x_min_max_scaling=self.minmax,
            #                                                                                     x_scale_column_to_exclude=self.complete_annot_vars,
            #                                                                                     y_min_max_scaling=self.minmax, y_column=self.y_var)
            # self._training_x, self._test_x = self._training[self._training.columns[
            #     ~self._training.columns.isin(self.complete_annot_vars)]], self._test[self._test.columns[~self._test.columns.isin(self.complete_annot_vars)]]
            # self._training_y, self._test_y = self._training[
            #     self.label_var], self._test[self.label_var]

            X_indices = np.arange(self.total_n)
            if self.resample_method == 'random':
                X_train_indices, X_test_indices, self._training_y, self._test_y = train_test_split(
                    X_indices, self.labels_working, test_size=self.test_percentage, stratify=None, random_state=self.rand)
            elif self.resample_method == 'stratified':
                X_train_indices, X_test_indices, self._training_y, self._test_y = train_test_split(
                    X_indices, self.labels_working, test_size=self.test_percentage, stratify=self.labels_working, random_state=self.rand)
            else:
                raise NotImplementedError(
                    '\"balanced\" resmapling method has not been implemented.')
            self._training_x, self._test_x = self.x[X_train_indices], self.x[X_test_indices]

        # - output -
        self.train_ds = tf.data.Dataset.from_tensor_slices(
            (self._training_x, self._training_y))
        self.train_n = self.train_ds.cardinality().numpy()
        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (self._test_x, self._test_y)) if (self._test_x is not None) and (self._test_y is not None) else None
        self.test_n = self.test_ds.cardinality().numpy() if self.test_ds is not None else None


# below: ad-hoc testing
mydata = singleCsvMemLoader(file='./data/test_dat.csv', label_var='group', annotation_vars=['subject', 'PCL'], sample_id_var='subject',
                            holdout_samples=None, minmax=True, x_standardize=True,
                            model_type='classification', cv_only=False, man_split=False, training_percentage=0.8, random_state=1, verbose=True)


# ------ process/__main__ statement ------
# if __name__ == '__main__':
#     mydata = DataLoader(file=args.file[0],
#                         label_var=args.label_var, annotation_vars=args.annotation_vars, sample_id_var=args.sample_id_var,
#                         holdout_samples=args.holdout_samples,
#                         model_type=args.model_type, cv_only=args.cv_only, man_split=args.man_split, training_percentage=args.training_percentage,
#                         random_state=args.random_state, verbose=args.verbose)
