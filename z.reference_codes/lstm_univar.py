"""
1st attempt to a LSTM-RNN time series modelling, including:
Univirate
Multivariate
Multi-steps
Multivariate multi-steps
"""

# ------ libraries ------
import os
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from custom_functions.dl_custom_functions import *
from custom_functions.dl_custom_lstm_functions import *


# ------ housekeeping ------
# OMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# logging
tf.logging.set_verbosity(tf.logging.ERROR)

# ------ custom functions ------
#
# ------ script ------
# ---- working directory
# set working directory
main_dir = os.path.abspath('./')
# dat_dir = os.path.join(main_dir, 'data')
res_dir = os.path.join(main_dir, 'results')
log_dir = os.path.join(main_dir, 'log')

# ---- setup logger
logger = logging_func(filepath=os.path.join(log_dir, 'lstm_univar.log'))

# ---- data
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# ---- model call back variables
# es = EarlyStopping(monitor='loss', mode='min', patience=3, verbose=True)

# ---- fit singular LSTM model
singluar_m_X, singluar_m_y = split_sequence_univar(sequence=raw_seq, n_steps=3)
for i in range(len(singluar_m_X)):
    print(singluar_m_X[i], singluar_m_y[i])
# [10 20 30] 40
# [20 30 40] 50
# [30 40 50] 60
# [40 50 60] 70
# [50 60 70] 80
# [60 70 80] 90
# [70 80 90] 100

# shape and reshape the X data from [samples, time points] to [samples, time points, features]
singluar_m_X = singluar_m_X.reshape(
    singluar_m_X.shape[0], singluar_m_X.shape[1], 1)  # n_features = 1
print(singluar_m_X)

# simple LSTM model
mc = ModelCheckpoint(os.path.join(res_dir, 'best_simple_lstm.h5'), monitor='loss',
                     mode='min', verbose=True, save_best_only=True)
simple_model = simple_lstm_m(n_steps=3, n_features=1)
simple_model_history = simple_model.fit(
    singluar_m_X, singluar_m_y, epochs=200,
    callbacks=[mc],
    verbose=True)
epochs_loss_plot(simple_model_history)

# stacked LSTM model
stacked_model = stacked_lstm_m(n_steps=3, n_features=1)
stacked_model_history = stacked_model.fit(
    singluar_m_X, singluar_m_y, epochs=200,
    callbacks=None,
    verbose=True)
epochs_loss_plot(stacked_model_history)

# bidirectional LSTM model
bidir_model = bidirectional_lstm_m(n_steps=3, n_features=1)
bidir_model_history = bidir_model.fit(
    singluar_m_X, singluar_m_y, epochs=200,
    callbacks=None,
    verbose=True)
epochs_loss_plot(bidir_model_history)

# ---- fit CNN-LSTM hybrid model
# format the data
hybrid_m_X, hybrid_m_y = split_sequence_univar(sequence=raw_seq, n_steps=4)
for i in range(len(hybrid_m_X)):
    print(hybrid_m_X[i], hybrid_m_y[i])
# [10 20 30 40] 50
# [20 30 40 50] 60
# [30 40 50 60] 70
# [40 50 60 70] 80
# [50 60 70 80] 90
# [60 70 80 90] 100

# reshape hybrid_X with shape: (samples, timepoints)
# to shape: (samples, subsequences (filter): |..|, timepoints, features)
# [|10 20| 30 40] 50
# [20 30 40 50] 60
# [30 40 50 60] 70
# [40 50 60 70] 80
# [50 60 70 80] 90
# [60 70 80 90] 100
# note: now the original four time points becomes two points upon filtering
hybrid_m_X = hybrid_m_X.reshape(hybrid_m_X.shape[0], 2, 2, 1)
print(hybrid_m_X)
hybrid_model = cnn_lstm_m(n_steps=2, n_features=1)
hybrid_model_history = hybrid_model.fit(
    hybrid_m_X, hybrid_m_y, epochs=500,
    callbacks=None,
    verbose=True)
epochs_loss_plot(hybrid_model_history)

# ---- prediction tests
test_x = np.array([70, 80, 90])
test_x = test_x.reshape(1, 3, 1)
y_hat = simple_model.predict(test_x)
print('Prediction using simple LSTM model: {:.3f}'.format(y_hat[0, 0]))
y_hat = stacked_model.predict(test_x)
print('Prediction using stacked LSTM model: {:.3f}'.format(y_hat[0, 0]))
y_hat = bidir_model.predict(test_x)
print('Prediction using bidirectional LSTM model: {:.3f}'.format(y_hat[0, 0]))

test_x = np.array([60, 70, 80, 90])
test_x = test_x.reshape(1, 2, 2, 1)
y_hat = hybrid_model.predict(test_x)
print('Prediction using CNN-LSTM model: {:.3f}'.format(y_hat[0, 0]))
