#!/usr/bin/env python3
"""
Current objectives:
[X] Test argparse
[X] Test output directory creation
[X] Test file reading
[X] Test file processing

NOTE
All the argparser inputs are loaded from method arguments, making the class more portable, i.e. not tied to
the application.
"""
# ------ import modules ------
import argparse
import math
import os
import sys
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import (KFold, LeaveOneOut, ShuffleSplit,
                                     StratifiedKFold, StratifiedShuffleSplit)
from sklearn.preprocessing import LabelEncoder


# ------ system classes ------
class colr:
    WHITE = '\033[0;97m'
    WHITE_B = '\033[1;97m'
    YELLOW = '\033[0;33m'
    YELLOW_B = '\033[1;33m'
    RED = '\033[0;31m'
    RED_B = '\033[1;31m'
    BLUE = '\033[0;94m'
    BLUE_B = '\033[1;94m'
    CYAN = '\033[0;36m'
    CYAN_B = '\033[1;36m'
    ENDC = '\033[0m'  # end colour


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
def flatten(x): return [item for sublist in x for item in sublist]


def error(message, *lines):
    """
    stole from: https://github.com/alexjc/neural-enhance
    """
    string = "\n{}ERROR: " + message + "{}\n" + \
        "\n".join(lines) + ("{}\n" if lines else "{}")
    print(string.format(colr.RED_B, colr.RED, colr.ENDC))
    sys.exit(2)


def warn(message, *lines):
    """
    stole from: https://github.com/alexjc/neural-enhance
    """
    string = "\n{}WARNING: " + message + "{}\n" + "\n".join(lines) + "{}\n"
    print(string.format(colr.YELLOW_B, colr.YELLOW, colr.ENDC))


def add_bool_arg(parser, name, help, input_type, default=False):
    """
    Purpose\n
                    autmatically add a pair of mutually exclusive boolean arguments to the
                    argparser
    Arguments\n
                    parser: a parser object
                    name: str. the argument name
                    help: str. the help message
                    input_type: str. the value type for the argument
                    default: the default value of the argument if not set
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name,
                       action='store_true', help=input_type + '. ' + help)
    group.add_argument('--no-' + name, dest=name,
                       action='store_false', help=input_type + '. ''(Not to) ' + help)
    parser.set_defaults(**{name: default})


# ------ GLOBAL variables -------
__version__ = '0.1.0'
AUTHOR = 'Jing Zhang, PhD'
DESCRIPITON = """
---------------------------------- Description ---------------------------------
Data loader for single CSV file data table for deep learning
--------------------------------------------------------------------------------
"""

# ------ augment definition ------
# set the arguments
parser = AppArgParser(description=DESCRIPITON,
                      epilog='Written by: {}. Current version: {}\n\r'.format(
                          AUTHOR, __version__),
                      formatter_class=argparse.RawDescriptionHelpFormatter)

add_arg = parser.add_argument
add_arg('file', nargs=1, default=[],
        help='Input CSV file. Currently only one file is accepable.')
add_arg('-w', "--working_dir", type=str, default=None,
        help='str. Working directory if not the current one. Default is None.')

add_arg('-s', '--sample_id_var', type=str, default=None,
        help='str. Vairable name for sample ID. NOTE: only needed with single file processing. Default is None.')
add_arg('-a', '--annotation_vars', type=str, nargs="+", default=[],
        help='list of str. names of the annotation columns in the input data, excluding the outcome variable. Default is [].')
add_arg('-n', '--n_timepoints', type=int, default=None,
        help='int. Number of timepoints. NOTE: only needed with single file processing. Default is None.')
add_arg('-cl', '--n_classes', type=int, default=None,
        help='int. Number of class for classification models. Default is None.')
add_arg('-y', '--outcome_var', type=str, default=None,
        help='str. Vairable name for outcome. NOTE: only needed with single file processing. Default is None.')
add_bool_arg(parser=parser, name='y_scale', input_type='flag', default=False,
             help='str. If to min-max scale outcome for regression study. Defatuls is False.')

add_arg('-v', '--cv_type', type=str,
        choices=['kfold', 'LOO', 'monte'], default='kfold',
        help='str. Cross validation type. Default is \'kfold\'')
add_arg('-kf', '--cv_fold', type=int, default=10,
        help='int. Number of cross validation fold when --cv_type=\'kfold\'. Default is 10.')
add_arg('-mn', '--n_monte', type=int, default=10,
        help='int. Number of Monte Carlo cross validation iterations when --cv_type=\'monte\'. Default is 10')
add_arg('-mt', '--monte_test_rate', type=float, default=0.2,
        help='float. Ratio for cv test data split when --cv_type=\'monte\'. Default is 0.2.')
add_bool_arg(parser=parser, name='cv_only', input_type='flag',
             help='If to do cv_only mode for training, i.e. no holdout test split. Default is False.',
             default=False)
add_bool_arg(parser=parser, name='man_split', input_type='flag',
             help='Manually split data into training and test sets. When set, the split is on -s/--sample_id_var. Default is False.',
             default=False)
add_arg('-t', '--holdout_samples', nargs='+', type=str, default=[],
        help='str. Sample IDs selected as holdout test group when --man_split was set. Default is None.')
add_arg('-p', '--training_percentage', type=float, default=0.8,
        help='num, range: 0~1. Split percentage for training set when --no-man_split is set. Default is 0.8.')
add_arg('-r', '--random_state', type=int, default=1, help='int. Random state.')

add_arg('-m', '--model_type', type=str, choices=['regression', 'classification'],
        default='classifciation',
        help='str. Model type. Options: \'regression\' and \'classification\'. Default is  \'regression\'.')
add_arg('-l', '--lstm_type', type=str, choices=['simple', 'bidirectional'],
        default='simple',
        help='str. LSTM model type. \'simple\' also contains stacked strcuture. Default is \'simple\'.')
add_arg('-ns', '--n_stack', type=int, default=1,
        help='int. Number of LSTM stacks. 1 means no stack. Default is 1 (no stack).')
add_arg('-e', '--epochs', type=int, default=500,
        help='int. Number of epochs for LSTM modelling. Default is 500. ')
add_arg('-b', '--batch_size', type=int, default=32,
        help='int. The batch size for LSTM modeling. Default is 32. ')
add_arg('-d', '--dense_activation', type=str, choices=['relu', 'linear', 'sigmoid', 'softmax'],
        default='linear', help='str. Acivitation function for the dense layer of the LSTM model. Default is \'linear\'')
add_arg('-c', '--loss', type=str,
        choices=['mean_squared_error', 'binary_crossentropy',
                 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'hinge'],
        default='mean_squared_error',
        help='str. Loss function for LSTM models. Default is \'mean_squared_error\'.')
add_arg('-u', '--hidden_units', type=int, default=50,
        help='int. Number of hidden unit for the LSTM network. Default is 50.')
add_arg('-x', '--dropout_rate', type=float, default=0.0,
        help='float, 0.0~1.0. Dropout rate for LSTM models . 0.0 means no dropout. Default is 0.0.')
add_arg('-g', '--optimizer', type=str,
        choices=['adam', 'sgd'], default='adam', help='str. Model optimizer. Default is \'adam\'.')
add_arg('-lr', '--learning_rate', type=float, default=0.001,
        help='foalt. Learning rate for the optimizer. Note: use 0.01 for sgd. Default is 0.001.')
add_bool_arg(parser=parser, name='stateful', input_type='flag', default=False,
             help="Use stateful LSTM for modelling. Default is False.")

add_arg('-o', '--output_dir', type=str,
        default='.',
        help='str. Output directory. NOTE: not an absolute path, only relative to working directory -w/--working_dir.')

add_bool_arg(parser=parser, name='verbose', input_type='flag', default=False,
             help='Verbose or not. Default is False.')

add_bool_arg(parser=parser, name='plot', input_type='flag',
             help='Explort a scatter plot', default=False)
add_arg('-j', '--plot-type', type=str,
        choices=['scatter', 'bar'], default='scatter', help='str. Plot type')

args = parser.parse_args()
# check the arguments. did not use parser.error as error() has fancy colours
if not args.sample_id_var:
    error('-s/--sample_id_var missing.',
          'Be sure to set the following: -s/--sample_id_var, -n/--n_timepoints, -y/--outcome_var, -a/--annotation_vars')
if not args.n_timepoints:
    error('-n/--n_timepoints flag missing.',
          'Be sure to set the following: -s/--sample_id_var, -n/--n_timepoints, -y/--outcome_var, -a/--annotation_vars')
if not args.outcome_var:
    error('-y/--outcome_var flag missing.',
          'Be sure to set the following: -s/--sample_id_var, -n/--n_timepoints, -y/--outcome_var, -a/--annotation_vars')
if len(args.annotation_vars) < 1:
    error('-a/--annotation_vars missing.',
          'Be sure to set the following: -s/--sample_id_var, -n/--n_timepoints, -y/--outcome_var, -a/--annotation_vars')

if args.man_split and len(args.holdout_samples) < 1:
    error('Set -t/--holdout_samples when --man_split was set.')

if args.dropout_rate < 0.0 or args.dropout_rate > 1.0:
    error('-x/--dropout_rate should be between 0.0 and 1.0.')

if args.n_stack < 1:
    error('-ns/--n_stack should be equal to or greater than 1.')

if args.cv_type == 'monte':
    if args.monte_test_rate < 0.0 or args.monte_test_rate > 1.0:
        error('-mt/--monte_test_rate should be between 0.0 and 1.0.')

if args.model_type == 'classification':
    if args.n_classes is None:
        error('Set -nc/n_classes when -m/--model_type=\'classification\'.')
    elif args.n_classes < 1:
        error('Set -nc/n_classes needs to be greater than 1 when -m/--model_type=\'classification\'.')
    elif args.n_classes > 2 and args.loss == 'binary_crossentropy':
        error('-l/--loss cannot be \'binary_crossentropy\' when -m/--model_type=\'classification\' and -nc/n_classes greater than 2.')


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

    def __init__(self, cwd, file,
                 outcome_var, annotation_vars, n_timepoints, sample_id_var,
                 model_type, n_classes,
                 cv_only,
                 man_split, holdout_samples, training_percentage, random_state, verbose):
        """
        # Arguments
            cwd: str. working directory
            file: str. complete input file path. "args.file[0]" from argparser]
            outcome_var: str. variable nanme for outcome. Only one is accepted for this version. "args.outcome_var" from argparser
            annotation_vars: list of strings. Column names for the annotation variables in the input dataframe, EXCLUDING outcome variable.
                "args.annotation_vars" from argparser
            n_timepoints: int. number of timepoints. "args.n_timepoints" from argparser
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
                self.cwd
                self.model_type
                self.n_classes
                self.file
                self.outcome_var
                self.annotation_vars
                self.n_timepoints
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
        # setup working director
        self.cwd = cwd

        # random state
        self.rand = random_state
        self.verbose = verbose

        # load files
        self.model_type = model_type
        self.n_classes = n_classes
        # convert to a list for training_test_spliter_final() to use
        self.outcome_var = outcome_var
        self.annotation_vars = annotation_vars
        self.y_var = [self.outcome_var]

        # args.file is a list. so use [0] to grab the string
        self.file = os.path.join(self.cwd, file)
        self._basename = os.path.basename(file)
        self.filename,  self._name_ext = os.path.splitext(self._basename)[
            0], os.path.splitext(self._basename)[1]

        if self.verbose:
            print('Loading file: ', self._basename, '...', end=' ')

        if self._name_ext != ".csv":
            error('The input file should be in csv format.',
                  'Please check.')
        elif not os.path.exists(self.file):
            error('The input file or directory does not exist.',
                  'Please check.')
        else:
            self.raw = pd.read_csv(self.file, engine='python')
            self.raw_working = self.raw.copy()  # value might be changed
            self.complete_annot_vars = self.annotation_vars + self.y_var
            self._n_annot_col = len(self.complete_annot_vars)
            self.n_timepoints = n_timepoints
            self.n_features = int(
                (self.raw_working.shape[1] - self._n_annot_col) // self.n_timepoints)  # pd.shape[1]: ncol

            self.cv_only = cv_only
            self.sample_id_var = sample_id_var
            self.holdout_samples = holdout_samples
            self.training_percentage = training_percentage

        if self.verbose:
            print('done!')

        if self.model_type == 'classification':
            self.le = LabelEncoder()
            self.le.fit(self.raw_working[self.outcome_var])
            self.raw_working[self.outcome_var] = self.le.transform(
                self.raw_working[self.outcome_var])
            self.label_mapping = dict(
                zip(self.le.classes_, self.le.transform(self.le.classes_)))
            if self.verbose:
                print('Class label encoding: ')
                for i in self.label_mapping.items():
                    print('{}: {}'.format(i[0], i[1]))

        # call setter here
        if self.verbose:
            print('Setting up modelling data...', end=' ')
        self.modelling_data = man_split
        if self.verbose:
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
        if self.cv_only:  # only training is stored
            self._training, self._test = self.raw_working, None
        else:
            # training and holdout test data split
            if man_split:
                # manual data split: the checks happen in the training_test_spliter_final() function
                self._training, self._test, _, _ = training_test_spliter_final(data=self.raw_working, random_state=self.rand,
                                                                               man_split=man_split, man_split_colname=self.sample_id_var,
                                                                               man_split_testset_value=self.holdout_samples,
                                                                               x_standardization=False, y_min_max_scaling=False)
            else:
                if self.model_type == 'classification':  # stratified
                    train_idx, test_idx = list(), list()
                    stf = StratifiedShuffleSplit(
                        n_splits=1, train_size=self.training_percentage, random_state=self.rand)
                    for train_index, test_index in stf.split(self.raw_working, self.raw_working[self.y_var]):
                        train_idx.append(train_index)
                        test_idx.append(test_index)
                    self._training, self._test = self.raw_working.iloc[train_idx[0],
                                                                       :].copy(), self.raw_working.iloc[test_idx[0], :].copy()
                else:  # regression
                    self._training, self._test, _, _ = training_test_spliter_final(
                        data=self.raw_working, random_state=self.rand, man_split=man_split, training_percent=self.training_percentage,
                        x_standardization=False, y_min_max_scaling=False)  # data transformation will be doen during modeling
        self._modelling_data = {
            'training': self._training, 'test': self._test}


# ------ model evaluation when cv_only=True ------
# below: single round lstm modelling
# mylstm = lstmModel(n_timepoints=mydata.n_timepoints,
#                    model_type=mydata.model_type, n_features=mydata.n_features,
#                    n_stack=args.n_stack, hidden_units=args.hidden_units, epochs=args.epochs,
#                    batch_size=args.batch_size, stateful=args.stateful, dropout=args.dropout_rate,
#                    dense_activation=args.dense_activation, loss=args.loss,
#                    optimizer=args.optimizer, learning_rate=args.learning_rate, verbose=True)


# # ------ model evaluation when cv_only=False ------
# # below: model ensemble testing

# # below: single production model testing
# myfinal_lstm = lstmProduction(training=mydata.modelling_data['training'], n_timepoints=mydata.n_timepoints, n_features=mydata.n_features,
#                               model_type=mydata.model_type, y_scale=args.y_scale, lstm_type=args.lstm_type, outcome_var=mydata.outcome_var,
#                               annotation_vars=mydata.annotation_vars, random_state=mydata.rand, verbose=mydata.verbose)
# # modelling
# myfinal_lstm.productionRun(res_dir=res_dir, optim_epochs=math.ceil(mycv.cv_bestparam_epocs_mean), n_timepoints=mydata.n_timepoints,
#                            n_stack=args.n_stack, hidden_units=args.hidden_units, batch_size=args.batch_size, stateful=args.stateful,
#                            dropout=args.dropout, dense_activation=args.dense_activation, loss=args.loss, optimizer=args.optimizer,
#                            learning_rate=args.learning_rate)

# # prepare test data
# test = mydata.modelling_data['test']


# ------ process/__main__ statement ------
# ------ setup output folders ------
# if __name__ == '__main__':
#     mydata = DataLoader()
