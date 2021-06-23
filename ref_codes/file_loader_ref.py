#!/usr/bin/env python3
"""
Current objectives:
[X] Test argparse
[X] Test output directory creation
[X] Test file reading
[X] Test file processing
[X] Test training
[X] Folder setup
[X] Save models and data
[X] Test and debug for the classification module
[X] ROC-AUC
[X] Code cleanup, generalization and optimization
[ ] Package everthing up to make the app portable, docker??
[X] Multi-class classification support
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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import (LSTM, BatchNormalization, Bidirectional,
                                     Dense, Dropout)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical

from custom_functions.cv_functions import (idx_func, longitudinal_cv_xy_array,
                                           lstm_cv_train, lstm_ensemble_eval,
                                           lstm_ensemble_predict)
from custom_functions.data_processing import training_test_spliter_final
from custom_functions.plot_functions import auc_plot

# from tensorflow.keras.callbacks import History  # for input argument type check
# from matplotlib import pyplot as plt
# # StratifiedKFold should be used for classification problems
# # StratifiedKFold makes sure the fold has an equal representation of the classes
# from sklearn.model_selection import KFold
# from custom_functions.data_processing import training_test_spliter_final
# from custom_functions.plot_functions import y_yhat_plot
# from custom_functions.util_functions import logging_func


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
LSTM regression/classification modelling using multiple-timepoint MEG connectome.
Currently, the program only accepts same feature size per timepoint.
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


class lstmModel(object):
    """
    # Purpose
        Simple or stacked LSTM modelling class
    # Details
        This class uses the custom error() function. So be sure to load it.
        It is recommended to use this class inside other classes as a dependency.
        Specifically, the lstm_eval() method only access numpy arrays and does not include new data scaling.
    # Methods
        __init__: load data and other information from DataLoader class and argparser
        simple_lstm_m: setup simple or stacked LSTM model and compile
        bidir_lstm_m: setup bidirectional LSTM model and compile
        lstm_fit: LSTM model fitting
        lstm_eval: additional LSTM model evaluation
    """

    def __init__(self, trainX, trainY,
                 model_type, n_classes,
                 n_timepoints, n_features,
                 n_stack, hidden_units, epochs, batch_size, stateful, dropout, dense_activation,
                 loss, optimizer, learning_rate, verbose, testX=None, testY=None):
        """
        # Behaviour
            The initilizer loads model configs
        # Arguments
            trainX: numpy ndarray for training X. shape requirment: n_samples x n_timepoints x n_features
            trainY: numpy ndarray for training Y. shape requirement: n_samples
            model_type: str. model type, "classification" or "regression". "args.model_type" from argparser, or DataLoader.model_type
            n_classes: int. number of classes when model_type='classification'
            n_timepoints: int. number of timeopints (steps). "n_timepoint" from argparser, or DataLoader.n_timepoints
            n_features: int. number of features per timepoint. could be from the DataLoader class attribute DataLoader.n_features
            n_stack: int. number of (simple) LSTM stacks. "args.n_stack" from argparser
            hidden_units: int. number of hidden units. "args.hidden_units" from argparser
            epochs: int. number of epochs. "args.epochs" from argparser
            batch_size: int. batch size. "args.batch" from argparser
            stateful: bool. if to use stateful LSTM. "args.stateful" from argparser
            dropout: float. dropout rate for LSTM. "args.dropout_rate" from argparser
            dense_activation: str. activation function for the MLP (decision making/output DNN). "args.dense_activation" from argparser
            loss: str. loss function. "args.loss" from argparser
            optimizer: str. optimizer. "args.optimizer" from argparser
            learning_rate: float. leanring rate for optimizer . "args.learning_rate" from argparser
            verbose: str. Optimizer type. "args.verbose" from argparser, or DataLoader.verbose. But it is recommneded to set it separately
            testX: (optional) numpy ndarray for test X. shape requirment: n_samples x n_timepoints x n_features
            testY: (optional) numpy ndarray for test Y. shape requirement: n_samples
        # Details
            NOTE: trainX, trainY, testX and testY are NOT transformed in this class. Therefore, the data need to be
            transformed prior to running this class.
        # Public class attributes
            Below: attributes read from arguments
                self.trainX
                self.trainY
                self.testX
                self.testY
                self.model_type
                self.n_classes
                self.n_timepoints
                self.n_features
                self.n_stack
                self.hidden_units
                self.epochs
                self.batch_size
                self.stateful
                self.dropout
                self.dense_activation
                self.loss
                self.optimizer
                self.lr: learning rate
        # Private class attributes (excluding class propterties)
            Below: private attributes read from arguments
                self._verbose
            self._opt: working optimizer with custom learning rate
            self._trainY_working: working Y
        """
        # import data
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

        # set up model
        self.model_type = model_type
        self.n_classes = n_classes
        self.n_timepoints = n_timepoints
        self.n_features = n_features

        self.n_stack = n_stack
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.stateful = stateful
        self.dropout = dropout
        self.dense_activation = dense_activation
        self.loss = loss
        self._verbose = verbose
        self.lr = learning_rate

        # setup optimizer
        self.optimizer = optimizer
        if self.optimizer == 'adam':
            self._opt = Adam(lr=self.lr)
        else:
            self._opt = SGD(lr=self.lr)

        # process data
        if self.model_type == 'classification':
            if self.n_classes > 2:
                # One Hot Encode for classification y lables only for training data
                self._trainY_working = to_categorical(self.trainY)
                if testY is None:
                    self._testY_working = self.testY
                else:
                    self._testY_working = to_categorical(
                        self.testY, num_classes=self.n_classes)
            else:
                self._trainY_working, self._testY_working = self.trainY, self.testY
        else:
            self._trainY_working, self._testY_working = self.trainY, self.testY

        # setup dense output number
        self.dense_n_outpout = self._trainY_working.shape[1]

    def simple_lstm_m(self):
        """
        # Behaviour
            This method uses dropout and batch normalization
        # Arguments
            n_outpout: int. number of output featurs.
                Since One Hot Encoder is used for class labelling in classification studies,
                make sure to use y.shape[1] as the output.
        # Public class attributes
            simple_m: simple or stacked LSTM model
            m: the final LSTM model
        """
        # model setup
        self.simple_m = Sequential()
        if self.n_stack > 1:  # if to use stacked LSTM or not
            for _ in range(self.n_stack):
                self.simple_m.add(LSTM(units=self.hidden_units, return_sequences=True,
                                       batch_size=self.batch_size,
                                       input_shape=(
                                           self.n_timepoints, self.n_features),
                                       stateful=self.stateful, dropout=self.dropout))
                self.simple_m.add(BatchNormalization())
            self.simple_m.add(LSTM(units=self.hidden_units))
            self.simple_m.add(BatchNormalization())
        else:
            self.simple_m.add(LSTM(units=self.hidden_units,
                                   batch_size=self.batch_size,
                                   input_shape=(self.n_timepoints,
                                                self.n_features),
                                   stateful=self.stateful, dropout=self.dropout))
            self.simple_m.add(BatchNormalization())
        self.simple_m.add(
            Dense(units=self.dense_n_outpout, activation=self.dense_activation))

        # model compiling
        self.simple_m.compile(
            loss=self.loss, optimizer=self._opt, metrics=['mse', 'accuracy'])
        self.m = self.simple_m

    def bidir_lstm_m(self):
        """
        # Behaviour
            This method uses dropout and batch normalization
        # Argument
            n_outpout: int. number of output featurs.
                Since One Hot Encoder is used for class labelling in classification studies,
                make sure to use y.shape[1] as the output.
        # Public class attributes
            bidir_m: bidirectional LSTM model
            m: the final LSTM model
            m_history: model history with metrices etc
        """
        # model setup
        self.bidir_m = Sequential()
        self.bidir_m.add(Bidirectional(LSTM(units=self.hidden_units, return_sequences=True,
                                            batch_size=self.batch_size,
                                            input_shape=(
                                                self.n_timepoints, self.n_features), stateful=self.stateful, dropout=self.dropout)))
        self.bidir_m.add(BatchNormalization())
        self.bidir_m.add(
            Dense(units=self.dense_n_outpout, activation=self.dense_activation))
        self.bidir_m.compile(loss=self.loss, optimizer=self._opt, metrics=[
                             'mse', 'accuracy'])
        self.m = self.bidir_m

    def lstm_fit(self, tfboard_dir=None):
        """
        # Arguments
            tfboard_dir: str. path to output tensorboard results. It is opitonal
        # Details
            Make sure to scale and transform trainX (and trainY if necessary for regression) before running
            For classification, One Hot Encode is used.
            NOTE: for classification models, we DO NOT feed One Hot Encoded trainY and testY: the class will do it for us
        # Public class attributes
            self.trainX: numpy ndarray for training X. shape requirment: n_samples x n_timepoints x n_features
            self.trainY: numpy ndarray for training Y. shape requirement: n_samples
            self.testX: numpy ndarray for test X. shape requirment: n_samples x n_timepoints x n_features
            self.testY: numpy ndarray for test Y. shape requirement: n_samples
        # Private class attributes (excluding class property)
            self._earlystop_callback: early stop callback
            self._tfboard_callback: tensorboard callback
            self._callbacks: list. a list of callbacks for model fitting
        """
        # training
        # below: use the epochs training from CV if no test data was provided for modelling
        if any(elem is None for elem in [self.testX, self.testY]):
            # fitting
            if self._verbose:
                print(
                    "lstm_fit: Fitting with training data Early Stopping and TF board.")

            self._earlystop_callback = EarlyStopping(
                monitor='loss', patience=5)
            self.m_history = self.m.fit(x=self.trainX, y=self._trainY_working, epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        verbose=self._verbose)
        else:
            if self._verbose:
                print(
                    "lstm_fit: Fitting with test/validation data Early Stopping and TF board.")
            # callbakcs
            self._earlystop_callback = EarlyStopping(
                monitor='val_loss', patience=5)
            if tfboard_dir:
                self._tfboard_callback = TensorBoard(log_dir=tfboard_dir)
                self._callbacks = [
                    self._earlystop_callback, self._tfboard_callback]
            else:
                self._callbacks = [self._earlystop_callback]

            # fitting
            self.m_history = self.m.fit(x=self.trainX, y=self._trainY_working, epochs=self.epochs,
                                        batch_size=self.batch_size, callbacks=self._callbacks,
                                        validation_data=(
                                            self.testX, self._testY_working),
                                        verbose=self._verbose)

            # export early stop epoch
            self.earlystopping_epochs = len(self.m_history.history['loss'])
            # Below: 5 is patience in EarlyStopping()
            self.bestparam_epochs = self.earlystopping_epochs - 5

    def lstm_eval(self, newX, newY, y_scaler=None):
        """
        # Purpose
            Evalutate model performance with new data
        # Arguments
            newX: np.array. Input new data. Shape: n_samples, n_timepoints, n_features
            newY: np.array. Input new data label. Shape: n_samples, n_output.
        # Details
            Make sure to scale and transform newX (and newY if necessary for regression) before running
            For classification, One Hot Encode is used. The class will convert the One Hot Encode into normal
            encode (integer) via np.argmax
            NOTE: for classification models, we DO NOT feed One Hot Encoded newY: the class will do it for us
        # Public class attributes
            self.loss: float. set by self.loss, e.g. 'mse' for regression models.
            self.rmse: float.
            self.accuracy: float. None when model_type='regression'
        # Private class attributes
            self._yhat: predicted values.
            self._newY_enc: for classification only, One Hot Encoded newY
            self._yhat_inversed: for regression study, inversed outcome to original scale if raw outcome is min-max scaled
            self._newY_inversed: for regression study, inversed input Y to original scale if the input is min-max scaled
        # Arguments
            newX: numpy ndarray for new data X. shape requirment: n_samples x n_timepoints x n_features
            newY: numpy ndarray for new data Y. shape requirement: n_samples
        """
        # evaluate

        if self.model_type == 'regression':
            self._yhat = self.m.predict(newX)
            if y_scaler:
                self._yhat_inversed = y_scaler.inverse_transform(self._yhat)
                self._newY_inversed = y_scaler.inverse_transform(newY)
                self._mse = mean_squared_error(
                    y_true=self._newY_inversed, y_pred=self._yhat_inversed)
                self.loss = self._mse
                self.yhat_out = self._yhat_inversed
            else:
                # eval output (list): [loss, mse, accu]
                self.loss, self._mse, _ = self.m.evaluate(
                    newX, newY, verbose=False)
                self.yhat_out = self._yhat
            self.accuracy = None
            self.yhat_out_prob = None
        else:  # classification
            self._yhat = self.m.predict_classes(
                newX, batch_size=self.batch_size)
            self.yhat_out_prob = self.m.predict_proba(
                newX, batch_size=self.batch_size)
            if self.n_classes > 2:
                self.yhat_out = np.argmax(self._yhat, axis=1)
                self._newY_enc = to_categorical(
                    newY, num_classes=self.n_classes)  # One Hot Encode
            else:
                self.yhat_out = self._yhat
                self._newY_enc = newY
            self.loss, self._mse, self.accuracy = self.m.evaluate(
                newX, self._newY_enc, verbose=False)
        self.rmse = math.sqrt(self._mse)

        # # ROC for classification
        # if self.model_type == 'classification':
        #     # TBC: ROC-AUC
        #     # Make sure to also export ROC AUC data so custom ROC figure can be made
        #     None


class cvTraining(object):
    """
    # Purpose
        Use cross-validation to train models.
    # Behaviours
        This class uses the LSTM model classes
    # Methods
        __init__: load the CV configuration from arg parser
        cvRun: run the CV modelling process according to the LSTM type
    # Class property
        cvSplitIdx: dict. Sample indices (row number) of training and test sets split for cross validation
    """

    def __init__(self, training, n_timepoints, n_features,
                 model_type, y_scale, lstm_type,
                 cv_type, cv_fold, n_monte, monte_test_rate,
                 outcome_var, annotation_vars,
                 random_state, verbose):
        """
        # argument
            training: pandas dataframe. input data: row is sample
            n_timepoints: int. number of timepoints. "args.n_timepoints" from argparser, or DataLoader.n_timepoints attribute
            n_features: int. number of features per timepoint. could be from DataLoader.n_features attribute
            model_type: str. model type, "classification" or "regression". "args.model_type" from argparser,
                or DataLoader.model_type attribute
            y_scale: bool. if to min-max scale outcome when model_type='regression'.
            lstm_type: str. lstm type. "args.lstm_type" from argparser
            cv_type: str. cross validation type. "args.cv_type" from argparser
            cv_fold: int. number of fold when cv_type="LOO" or "kfold". "args.cv_fold" from argparser
            n_monte: int. number of Monte Carlo iteratins when cv_type="monte". "args.n_monte" from argparser
            monte_test_rate: float, between 0 and 1. resampling percentage for test set when cv_type="monte"
            outcome_var: str. variable nanme for outcome. Only one is accepted for this version. "args.outcome_var" from argparser,
                or DataLoader.outcome_var
            annotation_vars: list of strings. Column names for the annotation variables in the input dataframe,
                EXCLUDING outcome variable. "args.annotation_vars" from argparser, or DataLoader.annotation_vars attribute
            random_state: int. random state. "args.random_state" from argparser, or DataLoader.rand attribute
            verbose: bool. verbose. "args.verbose", or DataLoader.verbose
        # Public class attributes
            Below are private attribute(s) read from arguments
                self.cv_type
                self.lstm_type
                self.y_scale
            self.n_iter: int. number of cv iterations according to cv_type
        # Private class attributes (excluding class property)
            Below are private attribute(s) read from arguments
                self._outcome_var
                self._annoation_vars
                self._n_timepoints
                self._n_features
                self._rand
                self._model_type
                self._verbose
            self._y_var: single str list. name of the outcome variable
            self._complete_annot_vars: list of strings. column names for the annotation variables in the input dataframe,
                INDCLUDING outcome varaible.
        """
        self.training = training.copy()
        self.cv_type = cv_type
        self.lstm_type = lstm_type
        self.y_scale = y_scale

        if self.cv_type == 'kfold':
            self.n_iter = cv_fold
        elif self.cv_type == 'LOO':
            self.n_iter = training.shape[0]  # number of rows/samples
        else:
            self.n_iter = n_monte
            self.monte_test_rate = monte_test_rate

        self._n_timepoints = n_timepoints
        self._n_features = n_features
        self._outcome_var = outcome_var
        self._annotation_vars = annotation_vars  # list of strings
        self._y_var = [self._outcome_var]
        self._complete_annot_vars = self._annotation_vars + self._y_var

        self._rand = random_state
        self._model_type = model_type
        self._verbose = verbose

        # property
        self.cvSplitIdx = self.cv_type

    @property
    def cvSplitIdx(self):
        return self._cvSplitIdx

    @cvSplitIdx.setter
    def cvSplitIdx(self, cv_type):
        """
        # Private class attributes for the property
            self._kfold: sklearn.KFold/sklearn.StratifiedKFold object if cv_type='kfold', according to the model type
            self._loo: sklearn.LeaveOneOut object if cv_type='LOO'
            self._monte: sklearn.ShuffleSplit/sklearn.StratifiedShuffleSplit object if cv_type='monte', according to the model type
            self._train_index: int array. sample (row) index for one cv training data fold
            self._test_index: int array. sample (row) index for one cv test data fold
            self._cv_training_idx: list of int array. sample (row) index for cv training data folds
            self._cv_test_idx: list of int array. sample (row) index for cv test data folds
        """
        # spliting
        if self._verbose:
            print('Cross validationo type: {}'.format(self.cv_type))
            print('Setting up data for cross validation...', end=' ')

        self._cv_training_idx, self._cv_test_idx = list(), list()

        if cv_type == 'LOO':  # leave one out, same for both regression and classification models
            self._loo = LeaveOneOut()
            for _train_index, _test_index in self._loo.split(self.training):
                self._cv_training_idx.append(_train_index)
                self._cv_test_idx.append(_test_index)
        else:
            if self._model_type == 'regression':
                if cv_type == 'kfold':
                    self._kfold = KFold(n_splits=self.n_iter,
                                        shuffle=True, random_state=self._rand)
                    for _train_index, _test_index in self._kfold.split(self.training):
                        self._cv_training_idx.append(_train_index)
                        self._cv_test_idx.append(_test_index)
                else:
                    self._monte = ShuffleSplit(
                        n_splits=self.n_iter, test_size=self.monte_test_rate, random_state=self._rand)
                    for _train_index, _test_index in self._monte.split(self.training):
                        self._cv_training_idx.append(_train_index)
                        self._cv_test_idx.append(_test_index)
            else:  # classification
                if cv_type == 'kfold':  # stratified
                    self._kold = StratifiedKFold(n_splits=self.n_iter,
                                                 shuffle=True, random_state=self._rand)
                    for _train_index, _test_index in self._kold.split(self.training, self.training[self._y_var]):
                        self._cv_training_idx.append(_train_index)
                        self._cv_test_idx.append(_test_index)
                else:  # stratified
                    self._monte = StratifiedShuffleSplit(
                        n_splits=self.n_iter, test_size=self.monte_test_rate, random_state=self._rand)
                    for _train_index, _test_index in self._monte.split(self.training, self.training[self._y_var]):
                        self._cv_training_idx.append(_train_index)
                        self._cv_test_idx.append(_test_index)

        # output
        self._cvSplitIdx = {
            'cv_training_idx': self._cv_training_idx, 'cv_test_idx': self._cv_test_idx}

        if self._verbose:
            print('done!')

    def cvRun(self, res_dir, tfboard_dir, *args, **kwargs):
        """
        # Purpose
            Run the CV training modelling. This class is less portable, as it is tied to the lstmModel class.
        # Arguments
            res_dir: str. output directory
            tfboard_dir: str. output directory for tensforboard, usually a sub directory to res_dir
        # Public class attributes
            self.cv_m_ensemble
            self.cv_m_history_ensemble
            self.cv_test_accuracy_ensemble
            self.cv_test_rmse_ensemble
        # Private class attributes (excluding class property)
            below: private attributes read from arguments
                self._res_dir: str. working_dir + output_dir
                sefl._tfboard_dir
            self._cv_training: a fold of cv training data
            self._cv_test: a fold of cv test data
            self._cv_train_scaler_X: sklearn StandardScaler object for X data standardization
            self._cv_train_scaler_Y: sklearn MinMacScaler object for Y min-max scaling for regression models and when y_scale=True
        """
        # check and set up output path
        self._res_dir = res_dir
        self._tfboard_dir = tfboard_dir

        # set up data
        self.cv_yhat_ensemble = list()
        self.cv_m_ensemble, self.cv_m_history_ensemble = list(), list()
        self.cv_test_loss_ensemble, self.cv_test_accuracy_ensemble, self.cv_test_rmse_ensemble = list(), list(), list()
        self.cv_earlystopped_epochs_ensemble, self.cv_bestparam_epochs_ensemble = list(), list()
        for i in range(self.n_iter):
            iter_id = str(i+1)
            if self._verbose:
                print('cv iteration: ', iter_id, '...', end=' ')
            # below: .copy for pd dataframe makes an explicit copy, avoiding Pandas SettingWithCopyWarning
            self._cv_training, self._cv_test = self.training.iloc[self.cvSplitIdx['cv_training_idx'][i],
                                                                  :].copy(), self.training.iloc[self.cvSplitIdx['cv_test_idx'][i], :].copy()

            # x standardization
            self._cv_train_scaler_X = StandardScaler()
            self._cv_training[self._cv_training.columns[~self._cv_training.columns.isin(self._complete_annot_vars)]] = self._cv_train_scaler_X.fit_transform(
                self._cv_training[self._cv_training.columns[~self._cv_training.columns.isin(self._complete_annot_vars)]])
            self._cv_test[self._cv_test.columns[~self._cv_test.columns.isin(self._complete_annot_vars)]] = self._cv_train_scaler_X.transform(
                self._cv_test[self._cv_test.columns[~self._cv_test.columns.isin(self._complete_annot_vars)]])  # DO NOT use fit_transform here

            # process outcome variable
            if self._model_type == 'regression' and self.y_scale:
                self._cv_train_scaler_Y = MinMaxScaler(feature_range=(0, 1))
                self._cv_training[self._cv_training.columns[self._cv_training.columns.isin(self._y_var)]] = self._cv_train_scaler_Y.fit_transform(
                    self._cv_training[self._cv_training.columns[self._cv_training.columns.isin(self._y_var)]])
                self._cv_test[self._cv_test.columns[self._cv_test.columns.isin(self._y_var)]] = self._cv_train_scaler_Y.transform(
                    self._cv_test[self._cv_test.columns[self._cv_test.columns.isin(self._y_var)]])  # DO NOT use fit_transform here

            # convert data to np arrays
            self._cv_train_x, self._cv_train_y = longitudinal_cv_xy_array(input=self._cv_training, Y_colnames=self._y_var,
                                                                          remove_colnames=self._annotation_vars, n_features=self._n_features)
            self._cv_test_x, self._cv_test_y = longitudinal_cv_xy_array(input=self._cv_test, Y_colnames=self._y_var,
                                                                        remove_colnames=self._annotation_vars, n_features=self._n_features)

            # training
            # below: make sure to have all the argumetns ready
            cv_lstm = lstmModel(trainX=self._cv_train_x, trainY=self._cv_train_y,
                                testX=self._cv_test_x, testY=self._cv_test_y,
                                model_type=self._model_type,
                                n_features=self._n_features,
                                *args, **kwargs)

            if self.lstm_type == "simple":
                cv_lstm.simple_lstm_m()
            else:  # stacked
                cv_lstm.bidir_lstm_m()

            cv_lstm.lstm_fit(tfboard_dir=os.path.join(
                self._tfboard_dir, 'cv_iter_'+iter_id))

            if self._model_type == 'regression' and self.y_scale:  # self._cv_test_x is already standardized
                cv_lstm.lstm_eval(
                    newX=self._cv_test_x, newY=self._cv_test_y, y_scaler=self._cv_train_scaler_Y)
            else:
                cv_lstm.lstm_eval(newX=self._cv_test_x, newY=self._cv_test_y)

            # saving and exporting
            cv_lstm.m.save(os.path.join(
                self._res_dir, 'lstm_cv_model_'+'iter_'+str(i+1)+'.h5'))
            self.cv_m_ensemble.append(cv_lstm.m)
            self.cv_m_history_ensemble.append(cv_lstm.m_history)
            self.cv_yhat_ensemble.append(cv_lstm.yhat_out)
            self.cv_test_accuracy_ensemble.append(cv_lstm.accuracy)
            self.cv_test_rmse_ensemble.append(cv_lstm.rmse)
            self.cv_test_loss_ensemble.append(cv_lstm.loss)
            self.cv_earlystopped_epochs_ensemble.append(
                cv_lstm.earlystopping_epochs)
            self.cv_bestparam_epochs_ensemble.append(cv_lstm.bestparam_epochs)

            # verbose
            if self._verbose:
                print("done!")
        self.cv_test_accuracy_mean = np.mean(self.cv_test_accuracy_ensemble)
        self.cv_test_accuracy_sd = np.std(self.cv_test_accuracy_ensemble)
        self.cv_test_rmse_mean = np.mean(self.cv_test_rmse_ensemble)
        self.cv_test_rmse_sd = np.std(self.cv_test_rmse_ensemble)
        self.cv_test_loss_mean = np.mean(self.cv_test_loss_ensemble)
        self.cv_test_loss_sd = np.std(self.cv_test_loss_ensemble)

        self.cv_earlystopped_epochs_mean = np.mean(
            self.cv_earlystopped_epochs_ensemble)
        self.cv_bestparam_epochs_mean = np.mean(
            self.cv_bestparam_epochs_ensemble)


class lstmProduction(object):
    """
    # Purpose
        To train final LSTM model for production using the optimal epochs usually obtained from cross validation
    # Behaviours
        This class uses entire data set to train the final LSTM model for porudction,
             and is therefore dependent on the modelling class, i.e. lstmModel.
        This class trains and saves the production model in file: final_lstm_model.h5
    # Methods
        __init__: load the CV configuration from arg parser
        productionRun: run the final modelling process according to the LSTM type
    """

    def __init__(self, training, n_timepoints, n_features,
                 model_type, n_classes,
                 y_scale, lstm_type,
                 outcome_var, annotation_vars,
                 random_state, test=None, verbose=True):
        """
        # Arguments
            training: pandas dataframe. input training data: row is sample
            n_timepoints: int. number of timepoints. "args.n_timepoints" from argparser, or DataLoader.n_timepoints attribute
            n_features: int. number of features per timepoint. could be from DataLoader.n_features attribute
            model_type: str. model type, "classification" or "regression". "args.model_type" from argparser, or DataLoader.model_type attribute
            n_classes: int. number of classes when model_type='classification'
            y_scale: bool. if to min-max scale outcome when model_type='regression'
            lstm_type: str. lstm type. "args.lstm_type" from argparser
            n_monte: int. number of Monte Carlo iteratins when cv_type="monte". "args.n_monte" from argparser
            monte_test_rate: float, between 0 and 1. resampling percentage for test set when cv_type="monte"
            outcome_var: str. variable nanme for outcome. Only one is accepted for this version. "args.outcome_var" from argparser, or DataLoader.outcome_var
            annotation_vars: list of strings. Column names for the annotation variables in the input dataframe, EXCLUDING outcome variable.
                "args.annotation_vars" from argparser, or DataLoader.annotation_vars attribute
            random_state: int. random state. "args.random_state" from argparser, or DataLoader.rand attribute
            test:(Optional) pandas dataframe. input test data: row is sample
            verbose: bool. verbose. "args.verbose", or DataLoader.verbose
        # Public class attributes
            Below are private attribute(s) read from arguments
            self.n_iter: int. number of cv iterations according to cv_type
        self.train_scaler_X: sklearn StandardScaler object for X data standardization
        self.train_scaler_Y: sklearn MinMacScaler object for Y min-max scaling for regression models and when y_scale=True
        # Private class attributes (excluding class property)
            Below are private attribute(s) read from arguments
                self._lstm_type
                self._training
                self._y_scale
                self._outcome_var
                self._annoation_vars
                self._n_timepoints
                self._n_features
                self._rand
                self._model_type
                self._n_classes
                self._verbose
            self._train_x, self._trainy, (if avaiable, self._test_x, self._test_y): np.arrays for the data
            self._y_var: single str list. name of the outcome variable
            self._complete_annot_vars: list of strings. column names for the annotation variables in the input dataframe, INDCLUDING outcome varaible.
        """
        self._training = training.copy()  # use copy() otherwise the original data will be replaced
        self._test = test.copy()  # use copy() otherwise the original data will be replaced
        self._n_timepoints = n_timepoints
        self._n_features = n_features
        self._outcome_var = outcome_var
        self._y_scale = y_scale
        # Below: list of strings, EXCLUDING outcome variable
        self._annotation_vars = annotation_vars
        self._rand = random_state
        self._model_type = model_type
        self._n_classes = n_classes
        self._lstm_type = lstm_type
        self._verbose = verbose

        self._y_var = [self._outcome_var]
        self._complete_annot_vars = self._annotation_vars + self._y_var

        # produce data scalers and transform training or test data
        # x standardization
        self.train_scaler_X = StandardScaler()
        self._training[self._training.columns[~self._training.columns.isin(self._complete_annot_vars)]] = self.train_scaler_X.fit_transform(
            self._training[self._training.columns[~self._training.columns.isin(self._complete_annot_vars)]])
        # process outcome variable
        if self._model_type == 'regression' and self._y_scale:
            self.train_scaler_Y = MinMaxScaler(feature_range=(0, 1))
            self._training[self._training.columns[self._training.columns.isin(self._y_var)]] = self.train_scaler_Y.fit_transform(
                self._training[self._training.columns[self._training.columns.isin(self._y_var)]])

        # convert data to np arrays
        self._train_x, self._train_y = longitudinal_cv_xy_array(input=self._training, Y_colnames=self._y_var,
                                                                remove_colnames=self._annotation_vars, n_features=self._n_features)

        if self._test is not None:
            # x standardization: use the training data information
            self._test[self._test.columns[~self._test.columns.isin(self._complete_annot_vars)]] = self.train_scaler_X.transform(
                self._test[self._test.columns[~self._test.columns.isin(self._complete_annot_vars)]])
            # process outcome variable: use the training data information
            if self._model_type == 'regression' and self._y_scale:
                self._test[self._test.columns[self._test.columns.isin(self._y_var)]] = self.train_scaler_Y.transform(
                    self._test[self._test.columns[self._test.columns.isin(self._y_var)]])

    # convert data to np arrays
            self._test_x, self._test_y = longitudinal_cv_xy_array(input=self._test, Y_colnames=self._y_var,
                                                                  remove_colnames=self._annotation_vars, n_features=self._n_features)

    def productionRun(self, res_dir, *args, tfboard_dir, **kwargs):
        """
        # Purpose
            Produce the final LTSM model
        # Arguments
            res_dir: str. output directory
            tfboard_dir: str. output directory for tensforboard, usually a sub directory to res_dir
        # Details
            When no test data is provided, make sure to provide the opitimal epochs.
            For example, epochs=math.ceil(mycv.cv_bestparam_epochs_mean) from crosss validation class cvTraining
        # Details
            The class puts X (or Y if available) scalers public as the naive data will use those for scaling
        # Private class attributes (excluding class property)
            Below are private attribute(s) read from arguments
                self._res_dir
                self._tfboard_dir
        """
        # load arguments
        self._res_dir = res_dir
        self._tfboard_dir = tfboard_dir

        # modelling
        self.final_lstm = lstmModel(
            trainX=self._train_x, trainY=self._train_y,
            model_type=self._model_type, n_classes=self._n_classes,
            n_features=self._n_features,
            testX=self._test_x,  testY=self._test_y,
            *args, **kwargs)

        if self._lstm_type == "simple":
            self.final_lstm.simple_lstm_m()
        else:  # stacked
            self.final_lstm.bidir_lstm_m()

        if self._test is not None:
            self.final_lstm.lstm_fit(tfboard_dir=os.path.join(
                self._tfboard_dir, 'final_lstm_model'))
        else:
            self.final_lstm.lstm_fit(tfboard_dir=None)

        self.final_lstm.m.save(os.path.join(
            self._res_dir, 'final_lstm_model.h5'))

    def productionEval(self):
        """
        # Purpose
            Test the perfomance on the single production model
        # Details
            the test data should be the exact same format as the raw/training data, including the outcome and annotation columns.
            the method will use these same annotation and scaler attribues from self.__init__ to transform data
        """
        # eval
        if self._model_type == 'regression' and self._y_scale:  # self._cv_test_x is already standardized
            self.final_lstm.lstm_eval(
                newX=self._test_x, newY=self._test_y, y_scaler=self.train_scaler_Y)
        else:
            self.final_lstm.lstm_eval(newX=self._test_x, newY=self._test_y)


# ------ local variables ------
# set up working and output directories
if args.working_dir:
    cwd = args.working_dir
else:
    cwd = os.getcwd()

# check and set up output path
if args.verbose:
    print("Set up results directory...", end=' ')
output_dir = args.output_dir
res_dir = os.path.join(cwd, output_dir)

if not os.path.exists(res_dir):  # set up out path
    os.mkdir(res_dir)
else:
    res_dir = res_dir + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(res_dir)

tfboard_dir = os.path.join(res_dir, 'tensorboard_res')  # set up tf board path
# below: no need to check as res_dir is new for sure
os.mkdir(tfboard_dir)
if args.verbose:
    print('Done!')

# ------ test ------
# print(args)
mydata = DataLoader(n_classes=args.n_classes,
                    cwd=cwd, file=args.file[0],
                    outcome_var=args.outcome_var, annotation_vars=args.annotation_vars,
                    sample_id_var=args.sample_id_var, n_timepoints=args.n_timepoints,
                    model_type=args.model_type, cv_only=args.cv_only,
                    man_split=args.man_split, holdout_samples=args.holdout_samples, training_percentage=args.training_percentage,
                    random_state=args.random_state, verbose=args.verbose)

# print('\n')
# print('input file path: {}'.format(mydata.file))
print('\n')
print('input file name: {}'.format(mydata.filename))
print('number of timepoints in the input file: {}'.format(mydata.n_timepoints))
print('number of features in the inpout file: {}'.format(mydata.n_features))

# ------ cv ------
mycv = cvTraining(training=mydata.modelling_data['training'],
                  n_timepoints=mydata.n_timepoints, n_features=mydata.n_features,
                  model_type=mydata.model_type, y_scale=args.y_scale,
                  lstm_type=args.lstm_type, cv_type=args.cv_type, cv_fold=args.cv_fold, n_monte=args.n_monte,
                  monte_test_rate=args.monte_test_rate,
                  outcome_var=mydata.outcome_var, annotation_vars=mydata.annotation_vars, random_state=mydata.rand,
                  verbose=mydata.verbose)

print('\n')
print('CV indices for training:\n{}'.format(
    mycv.cvSplitIdx['cv_training_idx']))
print('CV indices for test:\n{}'.format(mycv.cvSplitIdx['cv_test_idx']))
print('\n\r')
print('Working directory: {}'.format(cwd))

mycv.cvRun(res_dir=res_dir, tfboard_dir=tfboard_dir,
           n_timepoints=mydata.n_timepoints,
           n_stack=args.n_stack, hidden_units=args.hidden_units, epochs=args.epochs,
           batch_size=args.batch_size, stateful=args.stateful, dropout=args.dropout_rate,
           dense_activation=args.dense_activation, loss=args.loss,
           optimizer=args.optimizer, learning_rate=args.learning_rate, verbose=True)


for i in range(len(mycv.cv_test_rmse_ensemble)):
    print('iter: ', i+1, 'yhat: {}'.format(mycv.cv_yhat_ensemble[i]))
print('\n')
for i in range(len(mycv.cv_test_rmse_ensemble)):
    print('iter: ', i+1, 'RMSE: {}'.format(mycv.cv_test_rmse_ensemble[i]))
print('\n')
for i in range(len(mycv.cv_test_rmse_ensemble)):
    print('iter: ', i+1, 'loss: {}'.format(mycv.cv_test_loss_ensemble[i]))

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
