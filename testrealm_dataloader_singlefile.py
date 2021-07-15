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

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utils.dl_utils import SingleCsvMemLoader
from utils.data_utils import labelMapping, labelOneHot
from utils.other_utils import addBoolArg, colr, csvPath, error, warn


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

add_g1_arg = arg_g1.add_argument  # input/output
add_g2_arg = arg_g2.add_argument  # processing and resampling
add_g3_arg = arg_g3.add_argument  # modelling
add_g4_arg = arg_g4.add_argument  # others

# - add arugments to the argument groups -
# g1: inpout and ouput
add_g1_arg('file', nargs=1, type=csvPath,
           help='One and only one input CSV file. (Default: %(default)s)')

add_g1_arg('-sv', '--sample_id_var', type=str, default=None,
           help='str. Vairable name for sample ID. NOTE: only needed with single file processing. (Default: %(default)s)')
add_g1_arg('-av', '--annotation_vars', type=str, nargs="+", default=[],
           help='list of str. names of the annotation columns in the input data, excluding the label variable. (Default: %(default)s)')
# add_g1_arg('-cl', '--n_classes', type=int, default=None,
#            help='int. Number of class for classification models. (Default: %(default)s)')
add_g1_arg('-y', '--label_var', type=str, default=None,
           help='str. Vairable name for label. NOTE: only needed with single file processing. (Default: %(default)s)')
add_g1_arg('-ls', '--label_string_sep', type=str, default=None,
           help='str. Separator to separate label string, to create multilabel labels. (Default: %(default)s)')
addBoolArg(parser=arg_g1, name='y_scale', input_type='flag', default=False,
           help='str. If to min-max scale label for regression study. (Default: %(default)s)')
add_g1_arg('-o', '--output_dir', type=str,
           default='.',
           help='str. Output directory. NOTE: not an absolute path, only relative to working directory -w/--working_dir.')

# g2: resampling and normalization
addBoolArg(parser=arg_g2, name='cv_only', input_type='flag',
           help='If to do cv_only mode for training, i.e. no holdout test split. (Default: %(default)s)',
           default=False)
addBoolArg(parser=arg_g2, name='shuffle_for_cv_only', input_type='flag',
           help='Only effective when -cv_only is active, whether to randomly shuffle before proceeding. (Default: %(default)s)',
           default=True)
add_g2_arg('-cv', '--cv_type', type=str,
           choices=['kfold', 'LOO', 'monte'], default='kfold',
           help='str. Cross validation type. Default is \'kfold\'')
add_g2_arg('-tp', '--training_percentage', type=float, default=0.8,
           help='num, range: 0~1. Split percentage for training set. (Default: %(default)s)')
add_g2_arg('-gm', '--resample_method', type=str,
           choices=['random', 'stratified', 'balanced'], default='random',
           help='str. training-test split method. (Default: %(default)s)')
addBoolArg(parser=arg_g2, name='x_standardize', input_type='flag',
           default='False',
           help='If to apply z-score stardardization for x. (Default: %(default)s)')
addBoolArg(parser=arg_g2, name='minmax', input_type='flag',
           default='False',
           help='If to apply min-max normalization for x and, if regression, y (to range 0~1). (Default: %(default)s)')

# g3: modelling
add_g3_arg('-mt', '--model_type', type=str, choices=['regression', 'classification'],
           default='classifciation',
           help='str. Model type. Options: \'regression\' and \'classification\'. (Default: %(default)s)')

# g4: others
add_g4_arg('-se', '--random_state', type=int,
           default=1, help='int. Random state. (Default: %(default)s)')
addBoolArg(parser=arg_g4, name='verbose', input_type='flag', default=False,
           help='Verbose or not. (Default: %(default)s)')

args = parser.parse_args()
print(args)

# ------ check arguments. did not use parser.error as error() has fancy colours ------
# below: we use custom script to check this (not fileDir function) for custom error messages.
if os.path.isdir(args.output_dir):
    output_dir = os.path.normpath(os.path.abspath(
        os.path.expanduser(args.output_dir)))
else:
    error("Output directory not found.")

if not args.sample_id_var:
    error('-s/--sample_id_var missing.',
          'Be sure to set the following: -s/--sample_id_var, -y/--label_var, -a/--annotation_vars')

if not args.label_var:
    error('-y/--label_var flag missing.',
          'Be sure to set the following: -s/--sample_id_var, -y/--label_var, -a/--annotation_vars')

if len(args.annotation_vars) < 1:
    error('-a/--annotation_vars missing.',
          'Be sure to set the following: -s/--sample_id_var, -y/--label_var, -a/--annotation_vars')

if args.cv_type == 'monte':
    if args.monte_test_rate < 0.0 or args.monte_test_rate > 1.0:
        error('-mt/--monte_test_rate should be between 0.0 and 1.0.')


# below: ad-hoc testing
mydata = SingleCsvMemLoader(file='./data/test_dat.csv', label_var='group', annotation_vars=['subject', 'PCL'], sample_id_var='subject',
                            holdout_samples=None, minmax=True, x_standardize=True,
                            model_type='classification', training_percentage=0.8,
                            cv_only=False, shuffle_for_cv_only=False,
                            random_state=1, verbose=True)

tst_train, tst_test = mydata.generate_batched_data(
    batch_size=4, cv_only=True, shuffle_for_cv_only=True)

# ------ process/__main__ statement ------
# if __name__ == '__main__':
#     mydata = DataLoader(file=args.file[0],
#                         label_var=args.label_var, annotation_vars=args.annotation_vars, sample_id_var=args.sample_id_var,
#                         holdout_samples=args.holdout_samples,
#                         model_type=args.model_type, cv_only=args.cv_only, training_percentage=args.training_percentage,
#                         random_state=args.random_state, verbose=args.verbose)
