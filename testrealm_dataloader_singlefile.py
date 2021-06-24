#!/usr/bin/env python3
"""
Current objectives:
[ ] Test argparse
    [ ] Add groupped arguments
[ ] Test output directory creation
[ ] Test file reading
[ ] Test file processing

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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


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
# def flatten(x): return [item for sublist in x for item in sublist]


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


def file_path(string, type='f'):
    input_path = os.path.dirname(__file__)
    full_path = os.path.normpath(os.path.join(input_path, string))

    if os.path.isfile(full_path):
        return full_path
    else:
        error("invalid input file or input file not found.")


def output_dir(string):
    input_path = os.path.dirname(__file__)
    full_path = os.path.normpath(os.path.join(input_path, string))

    if os.path.isdir(full_path):
        return full_path
    else:
        error("output directory not found.")


# def training_test_spliter_final(data,
#                                 training_percent=0.8, random_state=None,
#                                 man_split=False, man_split_colname=None,
#                                 man_split_testset_value=None,
#                                 x_standardization=True,
#                                 x_scale_column_to_exclude=None,
#                                 y_min_max_scaling=False, y_column_to_scale=None,
#                                 y_scale_range=(0, 1)):
#     """
#     # Purpose:
#         This is a final verion of the training_test_spliter.
#         This version splits the data into training and test prior to Min-Max scaling.
#         The z score standardization is used for X standardization
#     # Return:
#         Pandas DataFrame (for now) for training and test data.
#         Y scalers for training and test data sets are also returned.
#         Order: training (np.array), test (np.array), training_scaler_X, training_scaler_Y
#     # Arguments:
#         data: Pnadas DataFrame. Input data.
#         man_split: boolean. If to manually split the data into training/test sets.
#         man_split_colname: string. Set only when fixed_split=True, the variable name for the column to use for manual splitting.
#         man_split_testset_value: list. Set only when fixed_split=True, the splitting variable values for test set.
#         training_percent: float. percentage of the full data to be the training
#         random_state: int. seed for resampling RNG
#         x_standardization: boolean. if to center scale (z score standardization) the input X data
#         x_scale_column_to_exclude: list. the name of the columns
#                                 to remove from the X columns for scaling.
#                                 makes sure to also inlcude the y column(s)
#         y_column_to_scale: list. column(s) to use as outcome for scaling
#         y_min_max_scaling: boolean. For regression study, if to do a Min_Max scaling to outcome
#         y_scale_range: two-tuple. the Min_Max range.
#     # Details:
#         The data normalization is applied AFTER the training/test splitting
#         The x_standardization is z score standardization ("center and scale"): (x - mean(x))/SD
#         The y_min_max_scaling is min-max nomalization
#         When x_standardization=True, the test data is standardized using training data mean and SD.
#         When y_min_max_scaling=True, the test data is scaled using training data max-min parameters.
#     # Examples
#     1. with normalization
#         training, test, training_scaler_X, training_scaler_Y = training_test_spliter_final(
#             data=raw, random_state=1,
#             man_split=True, man_split_colname='subject', man_split_testset_value=selected_features[0],
#             x_standardization=True,
#             x_scale_column_to_exclude=['subject', 'PCL', 'group'],
#             y_min_max_scaling=True,
#             y_column_to_scale=['PCL'], y_scale_range=(0, 1))
#     2. without noralization
#         training, test, _, _ = training_test_spliter_final(
#             data=raw, random_state=1,
#             man_split=True, man_split_colname='subject', man_split_testset_value=selected_features[1],
#             x_standardization=False,
#             y_min_max_scaling=False)
#     """
#     # argument check
#     if not isinstance(data, pd.DataFrame):
#         raise TypeError("Input needs to be a pandas DataFrame.")

#     if x_standardization:
#         if not isinstance(x_scale_column_to_exclude, list):
#             raise ValueError(
#                 'x_scale_column_to_exclude needs to be a list.')
#     if y_min_max_scaling:
#         if not isinstance(y_column_to_scale, list):
#             raise ValueError(
#                 'y_column_to_scale needs to be a list.')

#     if man_split:
#         if (not man_split_colname) or (not man_split_testset_value):
#             raise ValueError(
#                 'set man_split_colname and man_split_testset_value when man_split=True.')
#         else:
#             if not isinstance(man_split_colname, str):
#                 raise ValueError('man_split_colname needs to be a string.')
#             if not isinstance(man_split_testset_value, list):
#                 raise ValueError(
#                     'man_split_colvaue needs to be a list.')
#             if not all(test_value in list(data[man_split_colname]) for test_value in man_split_testset_value):
#                 raise ValueError(
#                     'One or all man_split_test_value missing from the splitting variable.')

#     # split
#     if man_split:
#         # .copy() to make it explicit that it is a copy, to avoid Pandas SettingWithCopyWarning
#         training = data.loc[~data[man_split_colname].isin(
#             man_split_testset_value), :].copy()
#         test = data.loc[data[man_split_colname].isin(
#             man_split_testset_value), :].copy()
#     else:
#         training = data.sample(frac=training_percent,
#                                random_state=random_state)
#         test = data.iloc[~data.index.isin(training.index), :].copy()

#     # normalization if needed
#     # set the variables
#     training_scaler_X, training_scaler_Y, test_scaler_Y = None, None, None
#     if x_standardization:
#         if all(selected_col in data.columns for selected_col in x_scale_column_to_exclude):
#             training_scaler_X = StandardScaler()
#             training[training.columns[~training.columns.isin(x_scale_column_to_exclude)]] = training_scaler_X.fit_transform(
#                 training[training.columns[~training.columns.isin(x_scale_column_to_exclude)]])
#             test[test.columns[~test.columns.isin(x_scale_column_to_exclude)]] = training_scaler_X.transform(
#                 test[test.columns[~test.columns.isin(x_scale_column_to_exclude)]])
#         else:
#             print(
#                 'Not all columns are found in the input X. Proceed without X standardization. \n')

#     if y_min_max_scaling:
#         if all(selected_col in data.columns for selected_col in y_column_to_scale):
#             training_scaler_Y = MinMaxScaler(feature_range=y_scale_range)
#             training[training.columns[training.columns.isin(y_column_to_scale)]] = training_scaler_Y.fit_transform(
#                 training[training.columns[training.columns.isin(y_column_to_scale)]])
#             test[test.columns[test.columns.isin(y_column_to_scale)]] = training_scaler_Y.transform(
#                 test[test.columns[test.columns.isin(y_column_to_scale)]])
#         else:
#             print(
#                 'Y column to scale not found. Proceed without Y scaling. \n')

#     return training, test, training_scaler_X, training_scaler_Y


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
add_g1_arg('file', nargs=1, type=file_path,
           help='One and only one input CSV file. (Default: %(default)s)')

add_g1_arg('-s', '--sample_id_var', type=str, default=None,
           help='str. Vairable name for sample ID. NOTE: only needed with single file processing. (Default: %(default)s)')
add_g1_arg('-a', '--annotation_vars', type=str, nargs="+", default=[],
           help='list of str. names of the annotation columns in the input data, excluding the outcome variable. (Default: %(default)s)')
add_g1_arg('-cl', '--n_classes', type=int, default=None,
           help='int. Number of class for classification models. (Default: %(default)s)')
add_g1_arg('-y', '--outcome_var', type=str, default=None,
           help='str. Vairable name for outcome. NOTE: only needed with single file processing. (Default: %(default)s)')
add_bool_arg(parser=arg_g1, name='y_scale', input_type='flag', default=False,
             help='str. If to min-max scale outcome for regression study. (Default: %(default)s)')
add_g1_arg('-o', '--output_dir', type=output_dir,
           default='.',
           help='str. Output directory. NOTE: not an absolute path, only relative to working directory -w/--working_dir.')

add_g2_arg('-v', '--cv_type', type=str,
           choices=['kfold', 'LOO', 'monte'], default='kfold',
           help='str. Cross validation type. Default is \'kfold\'')
add_g2_arg('-kf', '--cv_fold', type=int, default=10,
           help='int. Number of cross validation fold when --cv_type=\'kfold\'. (Default: %(default)s)')
add_g2_arg('-mn', '--n_monte', type=int, default=10,
           help='int. Number of Monte Carlo cross validation iterations when --cv_type=\'monte\'. (Default: %(default)s)')
add_g2_arg('-mt', '--monte_test_rate', type=float, default=0.2,
           help='float. Ratio for cv test data split when --cv_type=\'monte\'. (Default: %(default)s)')
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
           default=1, help='int. Random state.')

add_g3_arg('-m', '--model_type', type=str, choices=['regression', 'classification'],
           default='classifciation',
           help='str. Model type. Options: \'regression\' and \'classification\'. (Default: %(default)s)')

add_bool_arg(parser=arg_g4, name='verbose', input_type='flag', default=False,
             help='Verbose or not. (Default: %(default)s)')

args = parser.parse_args()

# # check arguments. did not use parser.error as error() has fancy colours
print(args)
# if not args.sample_id_var:
#     error('-s/--sample_id_var missing.',
#           'Be sure to set the following: -s/--sample_id_var, -y/--outcome_var, -a/--annotation_vars')

# if not args.outcome_var:
#     error('-y/--outcome_var flag missing.',
#           'Be sure to set the following: -s/--sample_id_var, -y/--outcome_var, -a/--annotation_vars')
# if len(args.annotation_vars) < 1:
#     error('-a/--annotation_vars missing.',
#           'Be sure to set the following: -s/--sample_id_var, -y/--outcome_var, -a/--annotation_vars')

# if args.man_split and len(args.holdout_samples) < 1:
#     error('Set -t/--holdout_samples when --man_split was set.')

# if args.cv_type == 'monte':
#     if args.monte_test_rate < 0.0 or args.monte_test_rate > 1.0:
#         error('-mt/--monte_test_rate should be between 0.0 and 1.0.')

# if args.model_type == 'classification':
#     if args.n_classes is None:
#         error('Set -nc/n_classes when -m/--model_type=\'classification\'.')
#     elif args.n_classes < 1:
#         error('Set -nc/n_classes needs to be greater than 1 when -m/--model_type=\'classification\'.')


# ------ loacl classes ------
# class DataLoader(object):
#     """
#     # Purpose
#         Data loading class.
#     # Methods
#         __init__: load data and other information from argparser, as well as class label encoding for classification study
#     # Details
#         This class is designed to load the data and set up data for training LSTM models.
#         This class uses the custom error() function. So be sure to load it.
#     # Class property
#         modelling_data: dict. data for model training. data is split if necessary.
#             No data splitting for the "CV only" mode.
#             returns a dict object with 'training' and 'test' items
#     """

#     def __init__(self, cwd, file,
#                  outcome_var, annotation_vars, sample_id_var,
#                  model_type, n_classes,
#                  cv_only,
#                  man_split, holdout_samples, training_percentage, random_state, verbose):
#         """
#         # Arguments
#             cwd: str. working directory
#             file: str. complete input file path. "args.file[0]" from argparser]
#             outcome_var: str. variable nanme for outcome. Only one is accepted for this version. "args.outcome_var" from argparser
#             annotation_vars: list of strings. Column names for the annotation variables in the input dataframe, EXCLUDING outcome variable.
#                 "args.annotation_vars" from argparser
#             sample_id_var: str. variable used to identify samples. "args.sample_id_var" from argparser
#             model_type: str. model type, classification or regression
#             n_classes: int. number of classes when model_type='classification'
#             cv_only: bool. If to split data into training and holdout test sets. "args.cv_only" from argparser
#             man_split: bool. If to use manual split or not. "args.man_split" from argparser
#             holdout_samples: list of strings. sample IDs for holdout sample, when man_split=True. "args.holdout_samples" from argparser
#             training_percentage: float, betwen 0 and 1. percentage for training data, when man_split=False. "args.training_percentage" from argparser
#             random_state: int. random state
#             verbose: bool. verbose. "args.verbose" from argparser
#         # Public class attributes
#             Below are attributes read from arguments
#                 self.cwd
#                 self.model_type
#                 self.n_classes
#                 self.file
#                 self.outcome_var
#                 self.annotation_vars
#                 self.cv_only
#                 self.holdout_samples
#                 self.training_percentage
#                 self.rand: int. random state
#             self.y_var: single str list. variable nanme for outcome
#             self.filename: str. input file name without extension
#             self.raw: pandas dataframe. input data
#             self.raw_working: pands dataframe. working input data
#             self.complete_annot_vars: list of strings. column names for the annotation variables in the input dataframe, INDCLUDING outcome varaible
#             self.n_features: int. number of features
#             self.le: sklearn LabelEncoder for classification study
#             self.label_mapping: dict. Class label mapping codes, when model_type='classification'
#         # Private class attributes (excluding class properties)
#             self._basename: str. complete file name (with extension), no path
#             self._n_annot_col: int. number of annotation columns
#         """
#         # setup working director
#         self.cwd = cwd

#         # random state
#         self.rand = random_state
#         self.verbose = verbose

#         # load files
#         self.model_type = model_type
#         self.n_classes = n_classes
#         # convert to a list for training_test_spliter_final() to use
#         self.outcome_var = outcome_var
#         self.annotation_vars = annotation_vars
#         self.y_var = [self.outcome_var]

#         # args.file is a list. so use [0] to grab the string
#         self.file = os.path.join(self.cwd, file)
#         self._basename = os.path.basename(file)
#         self.filename,  self._name_ext = os.path.splitext(self._basename)[
#             0], os.path.splitext(self._basename)[1]

#         if self.verbose:
#             print('Loading file: ', self._basename, '...', end=' ')

#         if self._name_ext != ".csv":
#             error('The input file should be in csv format.',
#                   'Please check.')
#         elif not os.path.exists(self.file):
#             error('The input file or directory does not exist.',
#                   'Please check.')
#         else:
#             self.raw = pd.read_csv(self.file, engine='python')
#             self.raw_working = self.raw.copy()  # value might be changed
#             self.complete_annot_vars = self.annotation_vars + self.y_var
#             self._n_annot_col = len(self.complete_annot_vars)
#             self.n_features = int(
#                 (self.raw_working.shape[1] - self._n_annot_col))  # pd.shape[1]: ncol

#             self.cv_only = cv_only
#             self.sample_id_var = sample_id_var
#             self.holdout_samples = holdout_samples
#             self.training_percentage = training_percentage

#         if self.verbose:
#             print('done!')

#         if self.model_type == 'classification':
#             self.le = LabelEncoder()
#             self.le.fit(self.raw_working[self.outcome_var])
#             self.raw_working[self.outcome_var] = self.le.transform(
#                 self.raw_working[self.outcome_var])
#             self.label_mapping = dict(
#                 zip(self.le.classes_, self.le.transform(self.le.classes_)))
#             if self.verbose:
#                 print('Class label encoding: ')
#                 for i in self.label_mapping.items():
#                     print('{}: {}'.format(i[0], i[1]))

#         # call setter here
#         if self.verbose:
#             print('Setting up modelling data...', end=' ')
#         self.modelling_data = man_split
#         if self.verbose:
#             print('done!')

#     @property
#     def modelling_data(self):
#         # print("called getter") # for debugging
#         return self._modelling_data

#     @modelling_data.setter
#     def modelling_data(self, man_split):
#         """
#         Private attributes for the property
#             _m_data: dict. output dictionary
#             _training: pandas dataframe. data for model training.
#             _test: pandas dataframe. holdout test data. Only available without the "--cv_only" flag
#         """
#         # print("called setter") # for debugging
#         if self.cv_only:  # only training is stored
#             self._training, self._test = self.raw_working, None
#         else:
#             # training and holdout test data split
#             if man_split:
#                 # manual data split: the checks happen in the training_test_spliter_final() function
#                 self._training, self._test, _, _ = training_test_spliter_final(data=self.raw_working, random_state=self.rand,
#                                                                                man_split=man_split, man_split_colname=self.sample_id_var,
#                                                                                man_split_testset_value=self.holdout_samples,
#                                                                                x_standardization=False, y_min_max_scaling=False)
#             else:
#                 if self.model_type == 'classification':  # stratified
#                     train_idx, test_idx = list(), list()
#                     stf = StratifiedShuffleSplit(
#                         n_splits=1, train_size=self.training_percentage, random_state=self.rand)
#                     for train_index, test_index in stf.split(self.raw_working, self.raw_working[self.y_var]):
#                         train_idx.append(train_index)
#                         test_idx.append(test_index)
#                     self._training, self._test = self.raw_working.iloc[train_idx[0],
#                                                                        :].copy(), self.raw_working.iloc[test_idx[0], :].copy()
#                 else:  # regression
#                     self._training, self._test, _, _ = training_test_spliter_final(
#                         data=self.raw_working, random_state=self.rand, man_split=man_split, training_percent=self.training_percentage,
#                         x_standardization=False, y_min_max_scaling=False)  # data transformation will be doen during modeling
#         self._modelling_data = {
#             'training': self._training, 'test': self._test}


# ------ model evaluation when cv_only=True ------
# below: single round lstm modelling


# # ------ model evaluation when cv_only=False ------
# # below: model ensemble testing

# # below: single production model testing
# # modelling

# # prepare test data
# test = mydata.modelling_data['test']


# ------ process/__main__ statement ------
# if __name__ == '__main__':
#     mydata = DataLoader()
