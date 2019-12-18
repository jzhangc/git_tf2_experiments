"""
lstm modelling pratice for multivariate and mulitple output (per timepoint)
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
logger = logging_func(filepath=os.path.join(log_dir, 'lstm_multivar.log'))

# ---- data
"""
The data is constructed for the "multivariate multi-outcome", i.e. outcome per time point

One sample: 
Input (columns: variables):
timepoint 1: 10, 15, 25
timepoint 2: 20, 25, 45
timepoint 3: 30, 35, 65

Output:
timepoint 1: 40
timepoint 2: 45
timepoint 3: 85
"""
# raw numbers
in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# cbind
in_seq1 = in_seq1.reshape(len(in_seq1), 1)
in_seq2 = in_seq1.reshape(len(in_seq2), 1)
out_seq = out_seq.reshape(len(out_seq), 1)
data = np.hstack([in_seq1, in_seq2, out_seq])

# slice to produce the correct data
X, y = split_sequences_multivar_para(sequences=data, n_steps=3)
print(X.shape)  # (6, 3, 3)
print(y.shape)  # (6, 3)
for i in range(len(X)):
    print(X[i], y[i])
# [[10 10 25]
#  [20 20 45]
#  [30 30 65]] [40 40 85]
# [[20 20 45]
#  [30 30 65]
#  [40 40 85]] [ 50  50 105]
# [[ 30  30  65]
#  [ 40  40  85]
#  [ 50  50 105]] [ 60  60 125]
# [[ 40  40  85]
#  [ 50  50 105]
#  [ 60  60 125]] [ 70  70 145]
# [[ 50  50 105]
#  [ 60  60 125]
#  [ 70  70 145]] [ 80  80 165]
# [[ 60  60 125]
#  [ 70  70 145]
#  [ 80  80 165]] [ 90  90 185]


# ---- modelling. NOTE: the outpout is one per timepint, therefore three.
# Mode checkpoint
simple_mc = ModelCheckpoint(
    filepath=os.path.join(res_dir, 'best_simple_lstm_multivar.h5'),
    monitor='loss', mode='min', verbose=True, save_best_only=True)
bidir_mc = ModelCheckpoint(
    filepath=os.path.join(res_dir, 'best_bidir_lstm_multivar.h5'),
    monitor='loss', mode='min', verbose=True, save_best_only=True)
stacked_mc = ModelCheckpoint(
    filepath=os.path.join(res_dir, 'best_stacked_lstm_multivar.h5'),
    monitor='loss', mode='min', verbose=True, save_best_only=True)

# simple LSTM
simple_model = simple_lstm_m(n_steps=3, n_features=3, n_output=3)
simple_model_history = simple_model.fit(
    x=X, y=y, epochs=200, callbacks=None, verbose=True)
epochs_loss_plot(simple_model_history)

# bidrecitional LSTM model
bidir_model = bidirectional_lstm_m(n_steps=3, n_features=3, n_output=3)
bidir_model_history = bidir_model.fit(
    x=X, y=y, epochs=200,
    callbacks=None,
    verbose=True)
epochs_loss_plot(bidir_model_history)

# stacked LSTM model
stacked_model = stacked_lstm_m(n_steps=3, n_features=3, n_output=3)
stacked_model_history = stacked_model.fit(
    x=X, y=y, epochs=200,
    callbacks=None,
    verbose=True)
epochs_loss_plot(stacked_model_history)

# ---- prediction
# data
test_X = np.array([[70, 75, 145], [80, 85, 165], [90, 95, 185]])
test_X = test_X.reshape(1, 3, 3)
test_y = np.array([[100, 105, 205]])

# predict
y_hat = simple_model.predict(test_X)
print('Predicting with simple model: {}'.format(y_hat))
y_hat = stacked_model.predict(test_X)
print('Predicting with stacked model: {}'.format(y_hat))
y_hat = bidir_model.predict(test_X)
print('Predicting with bidirectional model: {}'.format(y_hat))
