#!/usr/bin/env python3
"""
Current objectives:
[x] test argparse
    [x] Add groupped arguments
[ ] test output directory creation
[x] test file reading
[x] test file processing
    [x] normalization and scalling
    [x] converting to numpy arrays
[x] use convert to tf.dataset
[ ] finalize the argument check section

NOTE
All the argparser inputs are loaded from method arguments, making the class more portable, i.e. not tied to
the application.
"""
# ------ import modules ------
import argparse
import os
from utils.other_utils import AppArgParser, addBoolArg, colr, csvPath, fileDir
from utils.error_handling import error, warn

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
add_g1_arg('path', nargs=1, type=fileDir,
           help='Directory contains all input files. (Default: %(default)s)')
add_g1_arg('-o', '--output_dir', type=str,
           default='.',
           help='str. Output directory. NOTE: relative to working directory.')
# - load arguments -
args = parser.parse_args()
print(args)

# - check arguments -
if os.path.isdir(args.output_dir):
    output_dir = os.path.normpath(os.path.abspath(
        os.path.expanduser(args.output_dir)))
else:
    error('Output directory not found.')

# ------ ad-hoc test ------
print(output_dir)
