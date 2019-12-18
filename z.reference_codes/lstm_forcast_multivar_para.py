"""
This is the practice for multivariate parallel output (output per timepoint) 
forcasting modelling. 
"""

# ------ libraries ------
import os
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint
from custom_functions.dl_custom_functions import *
from custom_functions.dl_custom_lstm_functions import *


# ------ housekeeping ------
# OMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# logging
tf.logging.set_verbosity(tf.logging.ERROR)

# ------ custom functions ------
# ------ script ------
# ---- working directory
main_dir = os.path.abspath('./')
# dat_dir = os.path.join(main_dir, 'data')
res_dir = os.path.join(main_dir, 'results')
log_dir = os.path.join(main_dir, 'log')

# ---- setup logger
logger = logging_func(filepath=os.path.join(
    log_dir, 'lstm_forcast_multivar_para.log'))

# ---- data
"""
The data strcuture is the following

One sample:
Input (columns: variables):
timepoint 1: 10, 15, 25
timepoint 2: 20, 25, 45
timepoint 3: 30, 35, 65

Output:
timepoint 1: 40, 45, 85
timepoint 2: 45, 55, 105
timepoint 3: 85, 55, 105
"""
# raw numbers
in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# cbind
in_seq1 = in_seq1.reshape(len(in_seq1), 1)
in_seq2 = in_seq2.reshape(len(in_seq2), 1)
out_seq = out_seq.reshape(len(out_seq), 1)
data = np.hstack((in_seq1, in_seq2, out_seq))

# split into samples
X, Y = split_sequences_multivar_multiY_para(
    sequences=data, n_steps_in=3, n_steps_out=2)
print(X.shape, Y.shape)  # (5, 3, 3) (5, 2, 3)
for i in range(len(X)):
    print(X[i], Y[i])
# [[10 15 25]
#  [20 25 45]
#  [30 35 65]] [[ 40  45  85]
#  [ 50  55 105]]
# [[20 25 45]
#  [30 35 65]
#  [40 45 85]] [[ 50  55 105]
#  [ 60  65 125]]
# [[ 30  35  65]
#  [ 40  45  85]
#  [ 50  55 105]] [[ 60  65 125]
#  [ 70  75 145]]
# [[ 40  45  85]
#  [ 50  55 105]
#  [ 60  65 125]] [[ 70  75 145]
#  [ 80  85 165]]
# [[ 50  55 105]
#  [ 60  65 125]
#  [ 70  75 145]] [[ 80  85 165]
#  [ 90  95 185]]

# ---- modelling
# NOTE: no reshape is needed for this one
# NOTE: simple, stacked and bidirectional models are not suited for this problem
"""
From author:
With multivariate multi-step, a vanilla or bidirectional LSTM is not suited. 
You could force it, but you will need n x m nodes in the output for n time steps 
for m time series. The time steps of each series would be flattened in this 
structure. You must interpret each of the outputs as a specific time step for a 
specific series consistently during training and prediction.
"""
# n_dense_out=n_features
enc_dec_model = encoder_decoder_lstm_m(
    n_steps=3, n_features=3, n_output=2, n_dense_out=3)
enc_dec_model_history = enc_dec_model.fit(
    x=X, y=Y, epochs=300, verbose=True)

# plotting
epochs_loss_plot(enc_dec_model_history)

# ---- predict
test_X = np.array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
test_X = test_X.reshape((1, 3, 3))
y_hat = enc_dec_model.predict(test_X)
test_Y = np.array([[90, 95, 185], [100, 105, 205]])

print('Predicting with encoder-decoder LSTM model: {}'.format(y_hat))
