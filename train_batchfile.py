#!/usr/bin/env python3
"""
Objectives:
[x] Test argparse
    [x] Add groupped arguments
[ ] Test output directory creation
[x] Test file reading
[x] Test file processing
    [x] normalization and scaling
    [x] converting to numpy arrays
[ ] Implement "balanced" data resampling
[x] Implement data resampling for cross-validation (maybe move this out of the dataloader)


NOTE
All the argparser inputs are loaded from method arguments, making the class more portable, i.e. not tied to
the application.
"""
# ------ import modules ------
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from utils.dl_utils import BatchMatrixLoader
from utils.other_utils import (AppArgParser, addBoolArg, colr, error, fileDir,
                               warn)

# ------ GLOBAL variables -------
__version__ = '0.1.0'
AUTHOR = 'Jing Zhang, PhD'
DESCRIPITON = """
{}--------------------------------- Description -------------------------------------------
Data loader for batch adjacency matrix CSV file data table for deep learning.
The loaded CSV files are stored in a 3D numpy array, with size: _,_,c (c: channels)
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
parser._optionals.title = '{}Help options{}'.format(colr.CYAN_B, colr.ENDC)

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

# - add arguments to the argument groups -
# g1: input and output
add_g1_arg('path', nargs=1, type=fileDir,
           help='Directory contains all input files. (Default: %(default)s)')
add_g1_arg('-te', '--target_file_ext', type=str, default=None,
           help='str. When manual labels are provided and imported as a pandas dataframe, the label variable name for this pandas dataframe, e.g. \'txt\'. (Default: %(default)s)')
add_g1_arg('-lf', '--manual_label_file', type=str,
           help='Optional one manual labels CSV file. (Default: %(default)s)')
add_g1_arg('-ln', '--manual_label_filename_var', type=str,
           help='Filename variable name in the optional manual labels CSV file. (Default: %(default)s)')
add_g1_arg('-lv', '--manual_label_var', type=str, default=None,
           help='str. Label variable name in the optional manual labels CSV file. (Default: %(default)s)')
add_g1_arg('-ls', '--label_string_sep', type=str, default=None,
           help='str. Separator to separate label string, to create multilabel labels. (Default: %(default)s)')
add_g1_arg('-o', '--output_dir', type=str,
           default='.',
           help='str. Output directory. NOTE: relative to working directory.')

# g2: processing and resampling
add_g2_arg('-ns', '--new_shape', type=str, default=None,
           help='str. Optional new shape tuple. (Default: %(default)s)')
add_g2_arg('-xs', '--x_scaling', type=str, choices=['none', 'max', 'minmax'], default='minmax',
           help='If and how to scale x values. (Default: %(default)s)')
add_g2_arg('-xr', '--x_min_max_range', type=float,
           nargs='+', default=[0.0, 1.0], help='Only effective when x_scaling=\'minmax\', the range for the x min max scaling. (Default: %(default)s)')
add_g2_arg('-tp', '--training_percentage', type=float, default=0.8,
           help='num, range: 0~1. Split percentage for training set when --no-man_split is set. (Default: %(default)s)')
# # cv_type will be implemented later
# add_g2_arg('-cv', '--cv_type', type=str,
#            choices=['kfold', 'LOO', 'monte'], default='kfold',
#            help='str. Cross validation type. Default is \'kfold\'')
addBoolArg(parser=arg_g2, name='cv_only', input_type='flag',
           help='If to do cv_only mode for training, i.e. no holdout test split. (Default: %(default)s)',
           default=False)
addBoolArg(parser=arg_g2, name='shuffle_for_cv_only', input_type='flag',
           help='Only effective when -cv_only is active, whether to randomly shuffle before proceeding. (Default: %(default)s)',
           default=True)

# g3: modelling
add_g3_arg('-mt', '--model_type', type=str, default='classification',
           choices=['classification', 'regression'],
           help='Model (label) type. (Default: %(default)s)')
addBoolArg(parser=arg_g3, name='multilabel_classification', input_type='flag', default=False,
           help='If the classification is a \"multilabel\" type. Only effective when model_type=\'classification\'. (Default: %(default)s)')

# g4: others
add_g4_arg('-se', '--random_state', type=int,
           default=1, help='int. Random state. (Default: %(default)s)')
addBoolArg(parser=arg_g4, name='verbose', input_type='flag', default=False,
           help='Verbose or not. (Default: %(default)s)')

# - load arguments -
args = parser.parse_args()
print(args)


# ------- check arguments -------
# - below: we use custom script to check files and folders (not csvPath or fileDir functions) for custom error messages. -
if os.path.isdir(args.output_dir):
    # below: the output directory information is stored in output_dir
    # instead of args.output_dir, which only has the input string.
    output_dir = os.path.normpath(os.path.abspath(
        os.path.expanduser(args.output_dir)))
else:
    error('Output directory not found.')

if args.manual_label_file is not None:
    if os.path.isfile(args.manual_label_file):
        # return full_path
        _, file_ext = os.path.splitext(args.manual_label_file)
        if file_ext != '.csv':
            error('Manual label file needs to be .csv type.')
        else:
            manual_label_file = os.path.normpath(os.path.abspath(
                os.path.expanduser(args.manual_label_file)))
            manual_label_fileNameVar = args.manual_label_filename_var
            manual_label_labelVar = args.manual_label_var
    else:
        error('Invalid manual label file or file not found.')
else:
    manual_label_file, manual_label_fileNameVar, manual_label_labelVar = None, None, None


# ------ ad-hoc test ------
tst_dat = BatchMatrixLoader(filepath='./data/tf_data', target_file_ext='txt',
                            model_type='classification', multilabel_classification=False,
                            manual_labels=manual_label_file,
                            manual_labels_fileNameVar=manual_label_fileNameVar, manual_labels_labelVar=manual_label_labelVar,
                            label_sep=None,
                            x_scaling='none', x_min_max_range=[0, 1], resmaple_method='stratified',
                            training_percentage=0.8, verbose=False, random_state=1)

tst_train, tst_test = tst_dat.generate_batched_data(
    batch_size=10, cv_only=False, shuffle_for_cv_only=False)


# ------ process/__main__ statement ------
# if __name__ == '__main__':
#     mydata = DataLoader(file=args.file[0],
#                         label_var=args.label_var, annotation_vars=args.annotation_vars, sample_id_var=args.sample_id_var,
#                         holdout_samples=args.holdout_samples,
#                         model_type=args.model_type, cv_only=args.cv_only, man_split=args.man_split, training_percentage=args.training_percentage,
#                         random_state=args.random_state, verbose=args.verbose)
