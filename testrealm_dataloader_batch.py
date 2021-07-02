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
from numpy.core.numeric import cross
# import numpy as np
import pandas as pd
from utils.other_utils import error, warn, flatten, add_bool_arg, csv_path, output_dir, colr
from utils.data_utils import training_test_spliter_final
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

# g4: others
add_bool_arg(parser=arg_g4, name='verbose', input_type='flag', default=False,
             help='Verbose or not. (Default: %(default)s)')

args = parser.parse_args()

# check arguments. did not use parser.error as error() has fancy colours
print(args)
if args.man_split and len(args.holdout_samples) < 1:
    error('Set -t/--holdout_samples when --man_split was set.')


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

    def __init__(self, random_state, shape, new_shape,
                 model_type='classification',
                 multilabel=False,
                 cross_validation=False, k=10,
                 shuffle=True, verbose=True):
        """
        TBC
        """
        # model information
        self.model_type = model_type
        self.multilabel = multilabel

        self.new_shape = new_shape

        # resampling
        self.shuffle = shuffle
        self.cross_validation = cross_validation
        self.cv_k = k

        # random state and other settings
        self.rand = random_state
        self.verbose = verbose
        self.original_shape = shape

    def _data_process(self):
        print('TBC')
        return None

    def _data_reshape(self):
        print('TBC')
        return None

    def _data_resample(self):
        """NTOE: multilabel and regression can not use stratified splitting"""
        print('TBC')
        return None

    def _cross_val_split(self):
        print('TBC')
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
