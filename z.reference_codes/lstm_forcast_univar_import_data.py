"""
This exercise will show how to model a LSTM RNN model with imported dataset.

The current model is to create a model where the input is the passenger number at
timepoint t the outpout is the passenger number at timepoint t+1

data setup


"""

# ------ libraries ------
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from custom_functions.dl_custom_functions import *
from custom_functions.dl_custom_lstm_functions import *

# ------ housekeeping ------
# OMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# logging
tf.logging.set_verbosity(tf.logging.ERROR)

# set the RNG
np.random.seed(7)


# ------ custom functions ------
def create_dataset(dataset, look_back=1):
    """
    Split an input numpy array into input X and output Y

    Arguments;
    dataset: input numpy array
    look_back: the number of previous time steps to use as 
               input variables to predict the next time period
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# ------ script ------
# ---- working directory
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, 'data')
res_dir = os.path.join(main_dir, 'results')
log_dir = os.path.join(main_dir, 'log')

# ---- setup logger
logger = logging_func(filepath=os.path.join(
    log_dir, 'lstm_forcast_file_import.log'))

# ---- set the RNG
np.random.seed(7)

# ---- file import
"""
Below is a sample of the first few lines of the file.
"Month","Passengers"
"1949-01",112
"1949-02",118
"1949-03",132
"1949-04",129
"1949-05",121
We can load this dataset easily using the Pandas library. 
We are not interested in the date, given that each observation is 
separated by the same interval of one month. Therefore, when we 
load the dataset we can exclude the first column.
"""
# below: only use the second column
raw = pd.read_csv(os.path.join(
    dat_dir, 'airline-passengers.csv'), usecols=[1], engine='python')

# extract the value from the pd data frame and convert them into float
data = raw.values
data = data.astype('float32')

# Min-Max normalization
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
plt.plot(data)

# split data into training and set data
train_size = int(len(data) * 0.67)
training, test = data[0:train_size, :], data[train_size:len(data), :]
print(training.shape, test.shape)

# set up X and Y for the training and test sets
training_X, training_Y = create_dataset(training)
print(training.shape, training_X.shape, training_Y.shape)
for i in range(len(training_X)):
    print(training_X[i], training_Y[i])

test_X, test_Y = create_dataset(test)
for i in range(len(test_X)):
    print(test_X[i], test_Y[i])

# reshape data into the input compatible for keras: sample x timepoints x n_features
training_X = training_X.reshape(training_X.shape[0], 1, training_X.shape[1])
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

# ---- modelling
simple_model = simple_lstm_m(
    n_steps=1, n_features=1, n_output=1, hidden_units=4)
simple_model_history = simple_model.fit(
    x=training_X, y=training_Y, epochs=100, batch_size=1, verbose=2)
epochs_loss_plot(simple_model_history)

stacked_model = stacked_lstm_m(
    n_steps=1, n_features=1, n_output=1, hidden_units=4)
stacked_model_history = stacked_model.fit(
    x=training_X, y=training_Y, epochs=100, batch_size=1, verbose=2)
epochs_loss_plot(stacked_model_history)

bidir_model = bidirectional_lstm_m(
    n_steps=1, n_features=1, n_output=1, hidden_units=4)
bidir_model_history = bidir_model.fit(
    x=training_X, y=training_Y, epochs=100, batch_size=1, verbose=2)
epochs_loss_plot(bidir_model_history)

# ---- predict and performance eval
# raw predict
training_Y_hat = simple_model.predict(training_X)
test_Y_hat = simple_model.predict(test_X)

training_Y_hat = stacked_model.predict(training_X)
test_Y_hat = stacked_model.predict(test_X)

training_Y_hat = bidir_model.predict(training_X)
test_Y_hat = bidir_model.predict(test_X)

# we need to invert the predicted resutls to restore the raw data unit (passengers)
# because the raw prediction is based on the min-max normalized data
training_Y_hat = scaler.inverse_transform(training_Y_hat)
training_Y = scaler.inverse_transform([training_Y])  # [] is necessary
test_Y_hat = scaler.inverse_transform(test_Y_hat)
test_Y = scaler.inverse_transform([test_Y])  # [] is necessary

# RMSE calculation
training_rmse = math.sqrt(mean_squared_error(
    training_Y[0], training_Y_hat[:, 0]))
print('Training RMSE: {:.2f}'.format(training_rmse))
test_rmse = math.sqrt(mean_squared_error(test_Y[0], test_Y_hat[:, 0]))
print('Test RMSE: {:.2f}'.format(test_rmse))

# ---- plot resutls
"""
Because of how the dataset was prepared, 
we must shift the predictions so that they align 
on the x-axis with the original dataset. 
"""
training_Y_hat_plot = np.empty_like(data)
training_Y_hat_plot[:, :] = np.nan
# 1 is the look_back value
training_Y_hat_plot[1:len(training_Y_hat)+1, :] = training_Y_hat

test_Y_hat_plot = np.empty_like(data)
test_Y_hat_plot[:, :] = np.nan
test_Y_hat_plot[len(training_Y_hat)+(1*2)+1:len(data)-1,
                :] = test_Y_hat  # 1*2: look_back value*2

y_yhat_plot(y=scaler.inverse_transform(data),
            training_yhat=training_Y_hat_plot, test_yhat=test_Y_hat_plot)
