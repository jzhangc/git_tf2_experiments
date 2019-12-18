"""
Note: the save and load model functionality requires HDF5 support (h5py)

Data: ../data/pima-indians-diabetes.data.csv

Description (partial):
1. Title: Pima Indians Diabetes Database

2. Sources:
   (a) Original owners: National Institute of Diabetes and Digestive and
                        Kidney Diseases
   (b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
                          Research Center, RMI Group Leader
                          Applied Physics Laboratory
                          The Johns Hopkins University
                          Johns Hopkins Road
                          Laurel, MD 20707
                          (301) 953-6231
   (c) Date received: 9 May 1990

4. Relevant Information:
      Several constraints were placed on the selection of these instances from
      a larger database.  In particular, all patients here are females at
      least 21 years old of Pima Indian heritage.  ADAP is an adaptive learning
      routine that generates and executes digital analogs of perceptron-like
      devices.  It is a unique algorithm; see the paper for details.

5. Number of Instances: 768

6. Number of Attributes: 8 plus class

7. For Each Attribute: (all numeric-valued)
   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)

8. Missing Attribute Values: Yes

9. Class Distribution: (class value 1 is interpreted as "tested positive for
   diabetes")

   Class Value  Number of instances
   0            500
   1            268

10. Brief statistical analysis:

    Attribute number:    Mean:   Standard Deviation:
    1.                     3.8     3.4
    2.                   120.9    32.0
    3.                    69.1    19.4
    4.                    20.5    16.0
    5.                    79.8   115.2
    6.                    32.0     7.9
    7.                     0.5     0.3
    8.                    33.2    11.8

"""

# ------ imnport libraries ------
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout  # fully connnected layer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_curve, auc  # calculate ROC-AUC
from matplotlib import pyplot as plt  # to plot ROC-AUC

from custom_functions.util_functions import logging_func

# ------ housekeeping ------
# logger
tf.logging.set_verbosity(tf.logging.ERROR)  # disable tensorflow ERROR message


# ------ functions ------
def dp_model():  # mode set up
    # initiate and configure the model
    """
    Three layer fully connected (Dense) NN
    Input layer: 8 input
    Layer 1: 12 neurons (relu function)
    Layer 2: 8 neurons (relu function)
    Layer 3 (output layer): one binary output (sigmoid function)
    """
    m = Sequential()  # initiate a sequential NN
    m.add(Dense(units=12,
                input_dim=training_X.shape[1], kernel_initializer="uniform", activation="relu"))
    # m.add(Dropout(0.5))
    m.add(Dense(units=8, kernel_initializer="uniform", activation="relu"))
    # m.add(Dropout(0.5))
    m.add(Dense(units=1, kernel_initializer="uniform",
                activation="sigmoid"))  # output layer
    return m


# # ------ classes ------
# class MySeqentialDense(object):
#     """
#     this is my custom class for the model and the data
#     """
#
#     def __init__(self, data):
#         self.x
#         self.y
#
#     def auc(self):
#         return auc_plot(self.x, self.y)
#
#
# ------ script ------
# set working directory
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, 'data')
res_dir = os.path.join(main_dir, 'results')
log_dir = os.path.join(main_dir, 'log')

# set logger
logger = logging_func(filepath=os.path.join(log_dir, 'fully_connected.log'))

logger.info('directory existence')
for dir in (main_dir, dat_dir, res_dir):  # check existence
    logger.debug('{}: {}'.format(dir, os.path.exists(dir)))

# set random seed
np.random.seed(7)

# load data
dataset = pd.read_csv(os.path.join(
    dat_dir, 'pima-indians-diabetes.data.csv'), header=None)
training = dataset.sample(frac=0.9, random_state=99)
validation_test = dataset.iloc[~dataset.index.isin(
    training.index), :]  # exclude the training rows by index
validation = validation_test.sample(frac=0.5, random_state=2)
test = validation_test.iloc[~validation_test.index.isin(validation.index), :]
training_X, training_Y = training.iloc[:, 0:8], training.iloc[:, 8]
val_X, val_Y, test_X, test_Y = validation.iloc[:,
                                               0:8], validation.iloc[:, 8], test.iloc[:, 0:8], test.iloc[:, 8]

logger.info('training data length')
logger.debug(len(training.index))
logger.info('validation data length')
logger.debug(len(validation.index))
logger.info('testdata length')
logger.debug(len(test_X.index))
logger.info('training data snippet: X')
logger.debug(training_X.head())
logger.info('training data snippet: Y')
logger.debug(training_Y.head())
logger.info('validation data snippet: X')
logger.debug(val_X.head())
logger.info('validation snippet: Y')
logger.debug(val_Y.head())
logger.info('test data snippet: X')
logger.debug(test_X.head())
logger.info('test data snippet: Y')
logger.debug(test_Y.head())

# compile the model
model = dp_model()  # dp_model is the cusotm function at the top
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])


# early stopping callback variable
es = EarlyStopping(monitor='val_loss', mode='min', verbose=True)
mc = ModelCheckpoint(os.path.join(res_dir, 'best_model.h5'), monitor='val_acc',
                     mode='max', verbose=True, save_best_only=True)

# fit the model
m_history = model.fit(training_X, training_Y, epochs=200, batch_size=64,
                      validation_data=(val_X, val_Y),
                      callbacks=[es, mc],
                      verbose=False)
# model_fitted = model.fit(training_X, training_Y, epochs=150, batch_size=64,
#                          validation_data=None,
#                          verbose=True)
epochs_acc_plot(model_history=m_history)
epochs_loss_plot(model_history=m_history)


# evaluate the model
best_model = load_model(os.path.join(res_dir, 'best_model.h5'))

scores = best_model.evaluate(training_X, training_Y, verbose=False)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# ROC-AUC
auc_plot(model=best_model, newdata_X=test_X, newdata_Y=test_Y)
