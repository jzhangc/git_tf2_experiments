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
from utils.other_utils import addBoolArg, colr, csvPath, fileDir, error, warn


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
add_g2_arg('-xr', '--x_min_max_range', type=float,
           nargs='+', default=[0.0, 1.0])


# - load arguments -
args = parser.parse_args()
print(args)

# - check arguments -

# ------ ad-hoc test ------
