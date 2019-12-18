"""
Multivairate LSTM RNN modelling pratices
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
The goal is to constrcut a two-variable one output with three time points
Ex. Sample 1
var: v1, v2
t1: [10, 15]
t2: [20, 25]
t3: [30, 35]

y: 65
"""
# raw numbers
in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# reshape: transpose
in_seq1 = in_seq1.reshape(len(in_seq1), 1)
in_seq2 = in_seq2.reshape(len(in_seq2), 1)
out_seq = out_seq.reshape(len(out_seq), 1)

# horizontal stack, or "cbind" (R), to make a 9x3 matrix
data = np.hstack((in_seq1, in_seq2, out_seq))
print(data)
# [[10  15  25]
#  [20  25  45]
#  [30  35  65]
#  [40  45  85]
#  [50  55 105]
#  [60  65 125]
#  [70  75 145]
#  [80  85 165]
#  [90  95 185]]
print(data.shape)  # 9, 3

X = list()
for i in range(len(data)):
    # find the end of this pattern
    end_ix = 0 + 3
    # check if we are beyond the dataset
    if end_ix > len(data):
        break
    # gather input and output parts of the pattern
    seq_x = data[0:end_ix, :-1]
    X.append(seq_x)
np.array(X)

type(seq_x)

# split data
X, y = split_sequences_multivar(sequences=data, n_steps=3)

for i in range(len(X)):
    print(X[i], y[i])
# [[10 15]
#  [20 25]
#  [30 35]] 65
# [[20 25]
#  [30 35]
#  [40 45]] 85
# [[30 35]
#  [40 45]
#  [50 55]] 105
# [[40 45]
#  [50 55]
#  [60 65]] 125
# [[50 55]
#  [60 65]
#  [70 75]] 145
# [[60 65]
#  [70 75]
#  [80 85]] 165
# [[70 75]
#  [80 85]
#  [90 95]] 185

print(X.shape)  # 7, 3, 2: seven samples, three timepoints, and two variables

# ---- modelling
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
simple_model = simple_lstm_m(n_steps=3, n_features=2)
simple_model_history = simple_model.fit(
    x=X, y=y, epochs=200, callbacks=None, verbose=True)
epochs_loss_plot(simple_model_history)

# bidrecitional LSTM model
bidir_model = bidirectional_lstm_m(n_steps=3, n_features=2)
bidir_model_history = bidir_model.fit(
    x=X, y=y, epochs=200,
    callbacks=None,
    verbose=True)
epochs_loss_plot(bidir_model_history)

# stacked LSTM model
stacked_model = stacked_lstm_m(n_steps=3, n_features=2)
stacked_model_history = stacked_model.fit(
    x=X, y=y, epochs=200,
    callbacks=None,
    verbose=True)
epochs_loss_plot(stacked_model_history)

# ---- prediction test
# data: test_y = 205
test_X = np.array([[80, 85], [90, 95], [100, 105]])
test_X = test_X.reshape(1, 3, 2)

# predict
y_hat = simple_model.predict(test_X)
print('Predicting with simple model: {:.3f}'.format(y_hat[0, 0]))
y_hat = stacked_model.predict(test_X)
print('Predicting with stacked model: {:.3f}'.format(y_hat[0, 0]))
y_hat = bidir_model.predict(test_X)
print('Predicting with bidirectional model: {:.3f}'.format(y_hat[0, 0]))
