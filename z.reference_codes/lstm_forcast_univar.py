"""
lstm modelling pratice for univariate forcasting: multi-step models

A time series forecasting problem that requires a prediction of multiple time steps 
into the future can be referred to as multi-step time series forecasting.

Specifically, these are problems where the forecast horizon or interval is more 
than one time step.

Two models are shown here:
1. Vector output model
2. Encoder and decoder model
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
logger = logging_func(filepath=os.path.join(log_dir, 'lstm_forcast.log'))

# ---- data
"""
The data is constructed for "univariate multi-step modelling", i.e. univariate forcasting

training data: input timeponts 1 to 3, and predict timepoints 4 and 5
[10 20 30] [40 50]
[20 30 40] [50 60]
[30 40 50] [60 70]
[40 50 60] [70 80]
[50 60 70] [80 90]
"""
# raw numbers
raw = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# split t the data
X, Y = split_sequence_univar_multiY(sequence=raw, n_steps_in=3, n_steps_out=2)

for i in range(len(X)):
    print(X[i], Y[i])
# [10 20 30] [40 50]
# [20 30 40] [50 60]
# [30 40 50] [60 70]
# [40 50 60] [70 80]
# [50 60 70] [80 90]

# ---- modelling
# -- model one: vector output model
"""
Like other types of neural network models, the LSTM can output a vector 
directly that can be interpreted as a multi-step forecast.

This approach was seen in the previous section were one time step of 
each output time series was forecasted as a vector.
"""
# reshape X data: from (row: sample x column: timepoints) to (sample x timepoint x n_features)
singular_m_X = X.reshape(X.shape[0], X.shape[1], 1)
for i in range(len(singular_m_X)):
    print(singular_m_X[i], Y[i])
#  [20]
#  [30]] [40 50]
# [[20]
#  [30]
#  [40]] [50 60]
# [[30]
#  [40]
#  [50]] [60 70]
# [[40]
#  [50]
#  [60]] [70 80]
# [[50]
#  [60]
#  [70]] [80 90]

# modelling
simple_model = simple_lstm_m(
    n_steps=3, n_features=1, n_output=2, hidden_units=100)
simple_model_history = simple_model.fit(
    x=singular_m_X, y=Y, epochs=200, verbose=True)

stacked_model = stacked_lstm_m(
    n_steps=3, n_features=1, n_output=2, hidden_units=100)
stacked_model_history = stacked_model.fit(
    x=singular_m_X, y=Y, epochs=200, verbose=True)

bidir_model = bidirectional_lstm_m(
    n_steps=3, n_features=1, n_output=2, hidden_units=100)
bidir_model_history = bidir_model.fit(
    x=singular_m_X, y=Y, epochs=200, verbose=True)

# # NOTE: reshape to sample x sub sequences size (filter) x timpoints per filter size x n_features
# hybrid_m_X = X.reshape(X.shape[0], 2, 2, 1)
# hybird_model = cnn_lstm_m(n_steps=3, n_features=1,
#                           n_output=2, hidden_units=100)
# hybird_model_history = hybird_model.fit(x=X, y=Y, epochs=500, verbose=True)

#  plotting
histories = [simple_model_history, stacked_model_history, bidir_model_history]
for model_history in histories:
    epochs_loss_plot(model_history)


# predict
test_X = np.array([70, 80, 90])
test_X = test_X.reshape(1, 3, 1)
test_Y = np.array([100, 110])

y_hat = simple_model.predict(test_X)
print('Predicting with simple model: {}'.format(y_hat))
y_hat = stacked_model.predict(test_X)
print('Predicting with stacked model: {}'.format(y_hat))
y_hat = bidir_model.predict(test_X)
print('Predicting with bidirectional model: {}'.format(y_hat))


# -- encoder and decoder model
"""
The model was designed for prediction problems where there are both input 
and output sequences, so-called sequence-to-sequence, or seq2seq problems, 
such as translating text from one language to another.

This model can be used for multi-step time series forecasting.

As its name suggests, the model is comprised of two sub-models: 
the encoder and the decoder.

The encoder is a model responsible for reading and interpreting the input 
sequence. The output of the encoder is a fixed length vector that represents 
the model’s interpretation of the sequence. 
The encoder is traditionally a Vanilla LSTM model, although other encoder 
models can be used such as Stacked, Bidirectional, and CNN models.

The decoder uses the output of the encoder as an input.
"""

# data: here we need to reshape Y as well. Shape: sample x timepoints x n_features
edc_Y = Y.reshape(Y.shape[0], Y.shape[1], 1)

# modelling
enc_dec_model = encoder_decoder_lstm_m(n_steps=3, n_features=1, n_output=2)
enc_dec_model_history = enc_dec_model.fit(
    x=singular_m_X, y=edc_Y, epochs=200, verbose=True)

# plotting
epochs_loss_plot(enc_dec_model_history)

# predict
y_hat = enc_dec_model.predict(x=test_X)
print('Predicting with encoder-decoder LSTM model: {}'.format(y_hat))
