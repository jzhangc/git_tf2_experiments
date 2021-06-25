"""
Current objectives:
small things for data loaders
"""

# ------ modules ------
import os
import pandas as pd


# ------ test realm ------
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, 'data/')
file = pd.read_csv(os.path.join(
    dat_dir, 'test_dat.csv'), engine='python')


outcome_var = 'group'
n_class = file[outcome_var].nunique()

n_class
