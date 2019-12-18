"""
current objectives:
1. class-based DL analysis
2. tensorflow 2.0 transition: tf.keras
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

# ------ setup working dictory ------
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, '0.data')
res_dir = os.path.join(main_dir, '1.results')
