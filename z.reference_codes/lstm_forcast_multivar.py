"""
In this section, we will provide short examples of data preparation 
and modeling for multivariate multi-step time series forecasting as a 
template to ease this challenge, specifically:

- Multiple Input Multi-Step Output.
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
    log_dir, 'lstm_forcast_multivar.log'))

# ---- data
"""
The goal is to constrcut a two-variable two output with three time points
Ex. Sample 1
var: v1, v2
t1: [10, 15]
t2: [20, 25]
t3: [30, 35]

y: 65, 85
"""
# raw numbers
in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# cbind
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
data = np.hstack((in_seq1, in_seq2, out_seq))

# construct training data
X, Y = split_sequences_multivar_multiY(
    sequences=data, n_steps_in=3, n_steps_out=2)
print(X.shape, Y.shape)  # (6, 3, 2) (6, 2)

# summarize the data
for i in range(len(X)):
    print(X[i], Y[i])
# [[10 15]
#  [20 25]
#  [30 35]] [65 85]
# [[20 25]
#  [30 35]
#  [40 45]] [ 85 105]
# [[30 35]
#  [40 45]
#  [50 55]] [105 125]
# [[40 45]
#  [50 55]
#  [60 65]] [125 145]
# [[50 55]
#  [60 65]
#  [70 75]] [145 165]
# [[60 65]
#  [70 75]
#  [80 85]] [165 185]

# ---- modelling
# NOTE: no reshape is needed for this one
simple_model = simple_lstm_m(
    n_steps=3, n_features=2, n_output=2, hidden_units=100)
simple_model_history = simple_model.fit(
    x=X, y=Y, epochs=200, verbose=True)

stacked_model = stacked_lstm_m(
    n_steps=3, n_features=2, n_output=2, hidden_units=100)
stacked_model_history = stacked_model.fit(
    x=X, y=Y, epochs=200, verbose=True)

bidir_model = bidirectional_lstm_m(
    n_steps=3, n_features=2, n_output=2, hidden_units=100)
bidir_model_history = bidir_model.fit(
    x=X, y=Y, epochs=200, verbose=True)

# y needs to be in samples, timesteps, features: n_feature for the outpout is 1
edc_Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
enc_dec_model = encoder_decoder_lstm_m(n_steps=3, n_features=2, n_output=2)
enc_dec_model_history = enc_dec_model.fit(
    x=X, y=edc_Y, epochs=200, verbose=True)

# plotting
histories = [simple_model_history, stacked_model_history,
             bidir_model_history, enc_dec_model_history]
for model_history in histories:
    epochs_loss_plot(model_history)

# ---- predict
test_X = np.array([[70, 75], [80, 85], [90, 95]])
test_X = test_X.reshape(1, 3, 2)
test_Y = np.array([185, 205]).reshape(1, 2, 1)

y_hat = simple_model.predict(test_X)
print('Predicting with simple model: {}'.format(y_hat))
y_hat = stacked_model.predict(test_X)
print('Predicting with stacked model: {}'.format(y_hat))
y_hat = bidir_model.predict(test_X)
print('Predicting with bidirectional model: {}'.format(y_hat))
y_hat = enc_dec_model.predict(x=test_X)
print('Predicting with encoder-decoder LSTM model: {}'.format(y_hat))
