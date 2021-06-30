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
from utils.other_utils import error, warn, flatten, add_bool_arg, csv_path, output_dir, colr
from utils.data_utils import training_test_spliter_final
from sklearn.model_selection import StratifiedShuffleSplit
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
add_g1_arg('file', nargs=1, type=csv_path,
           help='One and only one input CSV file. (Default: %(default)s)')

add_g1_arg('-s', '--sample_id_var', type=str, default=None,
           help='str. Vairable name for sample ID. NOTE: only needed with single file processing. (Default: %(default)s)')
add_g1_arg('-a', '--annotation_vars', type=str, nargs="+", default=[],
           help='list of str. names of the annotation columns in the input data, excluding the outcome variable. (Default: %(default)s)')
# add_g1_arg('-cl', '--n_classes', type=int, default=None,
#            help='int. Number of class for classification models. (Default: %(default)s)')
add_g1_arg('-y', '--outcome_var', type=str, default=None,
           help='str. Vairable name for outcome. NOTE: only needed with single file processing. (Default: %(default)s)')
add_bool_arg(parser=arg_g1, name='y_scale', input_type='flag', default=False,
             help='str. If to min-max scale outcome for regression study. (Default: %(default)s)')
add_g1_arg('-o', '--output_dir', type=output_dir,
           default='.',
           help='str. Output directory. NOTE: not an absolute path, only relative to working directory -w/--working_dir.')

# g2: resampling and normalization
add_g2_arg('-v', '--cv_type', type=str,
           choices=['kfold', 'LOO', 'monte'], default='kfold',
           help='str. Cross validation type. Default is \'kfold\'')
add_bool_arg(parser=arg_g2, name='cv_only', input_type='flag',
             help='If to do cv_only mode for training, i.e. no holdout test split. (Default: %(default)s)',
             default=False)
add_bool_arg(parser=arg_g2, name='man_split', input_type='flag',
             help='Manually split data into training and test sets. When set, the split is on -s/--sample_id_var. (Default: %(default)s)',
             default=False)
add_g2_arg('-t', '--holdout_samples', nargs='+', type=str, default=[],
           help='str. Sample IDs selected as holdout test group when --man_split was set. (Default: %(default)s)')
add_g2_arg('-p', '--training_percentage', type=float, default=0.8,
           help='num, range: 0~1. Split percentage for training set when --no-man_split is set. (Default: %(default)s)')
add_g2_arg('-r', '--random_state', type=int,
           default=1, help='int. Random state. (Default: %(default)s)')
add_bool_arg(parser=arg_g2, name='x_standardize', input_type='flag',
             default='False',
             help='If to apply z-score stardardization for x. (Default: %(default)s)')
add_bool_arg(parser=arg_g2, name='minmax', input_type='flag',
             default='False',
             help='If to apply min-max normalization for x and, if regression, y (to range 0~1). (Default: %(default)s)')

# g3: modelling
add_g3_arg('-m', '--model_type', type=str, choices=['regression', 'classification'],
           default='classifciation',
           help='str. Model type. Options: \'regression\' and \'classification\'. (Default: %(default)s)')

# g4: others
add_bool_arg(parser=arg_g4, name='verbose', input_type='flag', default=False,
             help='Verbose or not. (Default: %(default)s)')

args = parser.parse_args()

# check arguments. did not use parser.error as error() has fancy colours
print(args)
if not args.sample_id_var:
    error('-s/--sample_id_var missing.',
          'Be sure to set the following: -s/--sample_id_var, -y/--outcome_var, -a/--annotation_vars')

if not args.outcome_var:
    error('-y/--outcome_var flag missing.',
          'Be sure to set the following: -s/--sample_id_var, -y/--outcome_var, -a/--annotation_vars')

if len(args.annotation_vars) < 1:
    error('-a/--annotation_vars missing.',
          'Be sure to set the following: -s/--sample_id_var, -y/--outcome_var, -a/--annotation_vars')

if args.man_split and len(args.holdout_samples) < 1:
    error('Set -t/--holdout_samples when --man_split was set.')

if args.cv_type == 'monte':
    if args.monte_test_rate < 0.0 or args.monte_test_rate > 1.0:
        error('-mt/--monte_test_rate should be between 0.0 and 1.0.')


# ------ loacl classes ------
class DataLoader(object):
    """
    # Purpose
        Data loading class.
    # Methods
        __init__: load data and other information from argparser, as well as class label encoding for classification study
    # Details
        This class is designed to load the data and set up data for training LSTM models.
        This class uses the custom error() function. So be sure to load it.
    # Class property
        modelling_data: dict. data for model training. data is split if necessary.
            No data splitting for the "CV only" mode.
            returns a dict object with 'training' and 'test' items
    """

    def __init__(self, file,
                 outcome_var, annotation_vars, sample_id_var,
                 model_type,
                 cv_only,
                 minmax,
                 x_standardize,
                 man_split, holdout_samples, training_percentage, random_state, verbose):
        """
        # Arguments
            file: str. complete input file path. "args.file[0]" from argparser]
            outcome_var: str. variable nanme for outcome. Only one is accepted for this version. "args.outcome_var" from argparser
            annotation_vars: list of strings. Column names for the annotation variables in the input dataframe, EXCLUDING outcome variable.
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
        # Public class attributes
            Below are attributes read from arguments
                self.model_type
                self.n_classes
                self.file
                self.outcome_var
                self.annotation_vars
                self.cv_only
                self.holdout_samples
                self.training_percentage
                self.rand: int. random state
            self.y_var: single str list. variable nanme for outcome
            self.filename: str. input file name without extension
            self.raw: pandas dataframe. input data
            self.raw_working: pands dataframe. working input data
            self.complete_annot_vars: list of strings. column names for the annotation variables in the input dataframe, INDCLUDING outcome varaible
            self.n_features: int. number of features
            self.le: sklearn LabelEncoder for classification study
            self.label_mapping: dict. Class label mapping codes, when model_type='classification'
        # Private class attributes (excluding class properties)
            self._basename: str. complete file name (with extension), no path
            self._n_annot_col: int. number of annotation columns
        """
        # random state and other settings
        self.rand = random_state
        self.verbose = verbose

        # load files
        self.model_type = model_type
        # convert to a list for training_test_spliter_final() to use
        self.outcome_var = outcome_var
        self.annotation_vars = annotation_vars
        self.y_var = [self.outcome_var]

        # args.file is a list. so use [0] to grab the string
        self.file = file
        self._basename, self._file_ext = os.path.splitext(file)
        # self.filename, self._name_ext = os.path.splitext(self._basename)[
        #     0], os.path.splitext(self._basename)[1]

        self.raw = pd.read_csv(self.file, engine='python')
        self.raw_working = self.raw.copy()  # value might be changed
        self.complete_annot_vars = self.annotation_vars + self.y_var
        self._n_annot_col = len(self.complete_annot_vars)
        self.n_features = int(
            (self.raw_working.shape[1] - self._n_annot_col))  # pd.shape[1]: ncol

        if model_type == 'classification':
            self.n_class = self.raw[outcome_var].nunique()
        else:
            self.n_class = None

        self.cv_only = cv_only
        self.sample_id_var = sample_id_var
        self.holdout_samples = holdout_samples
        self.training_percentage = training_percentage
        self.x_standardize = x_standardize
        self.minmax = minmax

        if verbose:
            print('done!')

        if self.model_type == 'classification':
            self.le = LabelEncoder()
            self.le.fit(self.raw_working[self.outcome_var])
            self.raw_working[self.outcome_var] = self.le.transform(
                self.raw_working[self.outcome_var])
            self.label_mapping = dict(
                zip(self.le.classes_, self.le.transform(self.le.classes_)))
            if verbose:
                print('Class label encoding: ')
                for i in self.label_mapping.items():
                    print('{}: {}'.format(i[0], i[1]))

        # call setter here
        if verbose:
            print('Setting up modelling data...', end=' ')
        self.modelling_data = man_split
        if verbose:
            print('done!')

    @property
    def modelling_data(self):
        # print("called getter") # for debugging
        return self._modelling_data

    @modelling_data.setter
    def modelling_data(self, man_split):
        """
        Private attributes for the property
            _m_data: dict. output dictionary
            _training: pandas dataframe. data for model training.
            _test: pandas dataframe. holdout test data. Only available without the "--cv_only" flag
        """
        # print("called setter") # for debugging
        # data resampling
        if self.cv_only:  # only training is stored
            self._training, self._test = self.raw_working, None
        else:
            # training and holdout test data split
            self._training, self._test, _, _, self._training_y_scaler = training_test_spliter_final(data=self.raw_working, random_state=self.rand,
                                                                                                    model_type=self.model_type,
                                                                                                    man_split=man_split, man_split_colname=self.sample_id_var,
                                                                                                    man_split_testset_value=self.holdout_samples,
                                                                                                    x_standardization=self.x_standardize,
                                                                                                    x_min_max_scaling=self.minmax,
                                                                                                    x_scale_column_to_exclude=self.complete_annot_vars,
                                                                                                    y_min_max_scaling=self.minmax, y_column=self.y_var)
            # if man_split:
            #     # manual data split: the checks happen in the training_test_spliter_final() function
            #     self._training, self._test, _, _, _ = training_test_spliter_final(data=self.raw_working, random_state=self.rand,
            #                                                                       man_split=man_split, man_split_colname=self.sample_id_var,
            #                                                                       man_split_testset_value=self.holdout_samples,
            #                                                                       x_standardization=False, y_min_max_scaling=False)
            # else:
            #     if self.model_type == 'classification':  # stratified
            #         train_idx, test_idx = list(), list()
            #         stf = StratifiedShuffleSplit(
            #             n_splits=1, train_size=self.training_percentage, random_state=self.rand)
            #         for train_index, test_index in stf.split(self.raw_working, self.raw_working[self.y_var]):
            #             train_idx.append(train_index)
            #             test_idx.append(test_index)
            #         self._training, self._test = self.raw_working.iloc[train_idx[0],
            #                                                            :].copy(), self.raw_working.iloc[test_idx[0], :].copy()
            #     else:  # regression
            #         self._training, self._test, _, _ = training_test_spliter_final(
            #             data=self.raw_working, random_state=self.rand, man_split=man_split, training_percent=self.training_percentage,
            #             x_standardization=False, y_min_max_scaling=False)  # data transformation will be doen during modeling

        self._training_x, self._test_x = self._training[self._training.columns[
            ~self._training.columns.isin(self.complete_annot_vars)]], self._test[self._test.columns[~self._test.columns.isin(self.complete_annot_vars)]]
        self._training_y, self._test_y = self._training[self.outcome_var], self._test[self.outcome_var]

        self._training_x, self._test_x = self._training_x.to_numpy(), self._test_x.to_numpy()
        self._training_y, self._test_y = self._training_y.to_numpy(), self._test_y.to_numpy()

        # output
        self._modelling_data = {
            'training_x': self._training_x, 'training_y': self._training_y,
            'training_y_scaler': self._training_y_scaler,
            'test_x': self._test_x, 'test_y': self._test_y}


# below: ad-hoc testing
mydata = DataLoader(file='./data/test_dat.csv', outcome_var='PCL', annotation_vars=['subject', 'group'], sample_id_var='subject',
                    holdout_samples=None, minmax=True, x_standardize=True,
                    model_type='regression', cv_only=False, man_split=False, training_percentage=0.8, random_state=1, verbose=True)

mydata.model_type
mydata.modelling_data['training_y_scaler']
mydata.modelling_data['test_y']

# ------ process/__main__ statement ------
# if __name__ == '__main__':
#     mydata = DataLoader(file=args.file[0],
#                         outcome_var=args.outcome_var, annotation_vars=args.annotation_vars, sample_id_var=args.sample_id_var,
#                         holdout_samples=args.holdout_samples,
#                         model_type=args.model_type, cv_only=args.cv_only, man_split=args.man_split, training_percentage=args.training_percentage,
#                         random_state=args.random_state, verbose=args.verbose)
