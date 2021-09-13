"""
this is test realm for plot functions with tf.keras models using the data from BatchDataLoader()

Objectives:
[x] extract values from tf.dataset objects
[x] ROC-AUC
    [x] calculate AUC
    [x] construct simple ROC
    [x] multiple classes
    [x] from One-Hot back to labels (figure legends)
[ ] Recall and precision curve
    [ ] Micro/Macro precision/recall values
[ ] F1 curve
[x] Update the epochsPlot function
"""


# ------ load modules ------
import os
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Flatten, Input, Layer, LeakyReLU,
                                     MaxPooling2D, Reshape, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.distribute.multi_worker_util import id_in_cluster
from tensorflow.python.keras.callbacks import History
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.python.types.core import Value

from utils.dl_utils import BatchMatrixLoader, WarmUpCosineDecayScheduler, makeGradcamHeatmap, displayGradcam
from utils.plot_utils import epochsPlotV2, rocaucPlot, lrSchedulerPlot
from utils.other_utils import flatten, warn, error
from utils.models import CnnClassifier, CnnClassifierFuncAPI

# Display
from IPython.display import Image, display
import matplotlib.cm as cm

# ------ TF device check ------
tf.config.list_physical_devices()


# ------ class ------


# ------ functions ------


# ------ data ------
# - multiclass -
tst_tf_dat = BatchMatrixLoader(filepath='./data/tf_data', target_file_ext='txt',
                               manual_labels=None, model_type='classification',
                               multilabel_classification=False, label_sep=None,
                               x_scaling='minmax', x_min_max_range=[0, 1], lower_triangular_padding=0,
                               resmaple_method='random',
                               training_percentage=0.8, verbose=False, random_state=1)
tst_tf_train, tst_tf_test = tst_tf_dat.generate_batched_data(batch_size=10)

# - multilabel with manual labels -
tst_tf_dat = BatchMatrixLoader(filepath='./data/tf_data', target_file_ext='txt',
                               manual_labels='./data/tst_annot.csv',
                               manual_labels_fileNameVar='filename', manual_labels_labelVar='label',
                               model_type='classification',
                               multilabel_classification=True, label_sep='_',
                               x_scaling='minmax', x_min_max_range=[-1, 1], lower_triangular_padding=0,
                               resmaple_method='random',
                               training_percentage=0.8, verbose=False, random_state=1)
tst_tf_train, tst_tf_test = tst_tf_dat.generate_batched_data(batch_size=10)


# ------ training ------
# -- early stop and optimizer --
# Training batch size, set small value here for demonstration purpose.
batch_size = tst_tf_dat.train_batch_n
epochs = 100

# Base learning rate after warmup.
learning_rate_base = 0.001
total_steps = int(epochs * tst_tf_dat.train_n / batch_size)

# Number of warmup epochs.
warmup_epoch = 10
warmup_steps = int(warmup_epoch * tst_tf_dat.train_n / batch_size)
warmup_batches = warmup_epoch * tst_tf_dat.train_n / batch_size

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0)

# optimizer
# optm = Adam(learning_rate=0.001, decay=0.001/80)  # decay?? lr/epoch
# no need to specify decay here as the scheduler will take care of that.
optm = Adam()

# early stop
earlystop = EarlyStopping(monitor='val_loss', patience=5)

# callback
callbacks = [warm_up_lr, earlystop]
callbacks = [warm_up_lr]


# -- model --
# - multiclass -
tst_m = CnnClassifier(initial_x_shape=(90, 90, 1), y_len=len(tst_tf_dat.labels_map_rev), multilabel=False,
                      bottleneck_dim=64, output_n=len(tst_tf_dat.lables_count),
                      output_activation='softmax')
tst_m.model().summary()

tst_m_cls = CnnClassifierFuncAPI(initial_x_shape=(90, 90, 1), y_len=len(tst_tf_dat.labels_map_rev), multilabel=False,
                                 bottleneck_dim=64, output_n=len(tst_tf_dat.lables_count),
                                 output_activation='softmax')
tst_m = tst_m_cls.model()
tst_m.summary()

# - multilabel -
tst_m = CnnClassifier(initial_x_shape=(90, 90, 1), y_len=len(tst_tf_dat.labels_map_rev), multilabel=True,
                      bottleneck_dim=64, output_n=len(tst_tf_dat.lables_count),
                      output_activation='sigmoid')
tst_m.model().summary()

tst_m_cls = CnnClassifierFuncAPI(initial_x_shape=(90, 90, 1), y_len=len(tst_tf_dat.labels_map_rev), multilabel=True,
                                 bottleneck_dim=64, output_n=len(tst_tf_dat.lables_count),
                                 output_activation='sigmoid')
tst_m = tst_m_cls.model()
tst_m.summary()

# -- training --
# - multiclass -
tst_m.compile(optimizer=optm, loss="categorical_crossentropy",
              metrics=['categorical_accuracy'])  # for mutually exclusive multiclass

# - multilabel -
tst_m.compile(optimizer=optm, loss="binary_crossentropy",
              metrics=['binary_accuracy', tf.keras.metrics.Recall()])  # for multilabel

# - fitting -
tst_m_history = tst_m.fit(tst_tf_train, epochs=epochs,
                          callbacks=callbacks,
                          validation_data=tst_tf_test)

lrSchedulerPlot(warm_up_lr)


# -- prediction --
label_dict = tst_tf_dat.labels_map_rev

# - single label multiclass -
proba, pred_class = tst_m.predict_classes(
    label_dict=label_dict, x=tst_tf_test)
# to_categorical(np.argmax(pred, axis=1), len(tst_tf_dat.lables_count))

proba, pred_class = tst_m_cls.predict_classes(
    label_dict=label_dict, x=tst_tf_test)

# - multilabel -
proba, pred_class = tst_m.predict_classes(
    label_dict=label_dict, x=tst_tf_test, proba_threshold=0.5)

proba, pred_class = tst_m_cls.predict_classes(
    label_dict=label_dict, x=tst_tf_test, proba_threshold=0.5)


# - prediction -
for i, b in tst_tf_test:
    # print(b.numpy())
    i_numpy = i.numpy()
    bn = b.numpy()
    print(i_numpy)
    print(bn)
    break

tst_pred = tst_m.predict(x=i_numpy)

tst_img = i_numpy[0, :, :, :].reshape([1, 90, 90, 1])

# tst_img = tf.keras.applications.xception.preprocess_input(tst_img)
img_size = (90, 90)
last_conv_layer_name = 'last_conv'


tst_m.layers[-1].activation = None
pred = tst_m.predict(tst_img)

pred, class_out = tst_m_cls.predict_classes(
    x=tst_img, label_dict=label_dict, proba_threshold=0.5)

heatmap = makeGradcamHeatmap(tst_img, tst_m, last_conv_layer_name)

plt.matshow(heatmap)
plt.matshow(tst_img.reshape((90, 90)))
plt.show()

try:
    last_conv_layer = next(
        x for x in tst_m2.layers[::-1] if isinstance(x, Conv2D))
except Exception as e:
    warn("not there")

last_conv_layer.name


tst_m2 = Model()


def GradCAM():
    def __init__(self, model, label_index, last_conv_layer_name=None):
        # -- initialization --
        self.model = model
        self.label_index = label_index

        if last_conv_layer_name is None:
            try:
                last_conv_layer = next(
                    x for x in model.layers[::-1] if isinstance(x, Conv2D))
                last_conv_layer_name = last_conv_layer.name
            except StopIteration as e:
                print('')

        self.last_conv_layer_name = last_conv_layer_name


# - ROC-AUC curve -
proba_threshold = 0.5
"""this is to display percentages for each class"""
# raise NotImplemented('TBC')
multilabel_res = np.zeros(proba.shape)
for i, j in enumerate(proba):
    print(f'{i}: {j}')
    sample_res = j >= proba_threshold
    for m, n in enumerate(sample_res):
        print(f'{m}: {n}')
        multilabel_res[i, m] = n
    # break

label_dict = tst_tf_dat.labels_map_rev
auc_res, _, _ = rocaucPlot(classifier=tst_m, x=tst_tf_test,
                           label_dict=label_dict, legend_pos='outside', proba_threshold=0.5)

auc_res, _, _ = rocaucPlot(classifier=tst_m, x=tst_tf_train,
                           label_dict=label_dict, legend_pos='outside', proba_threshold=0.5)


# - epochs plot function -
tst_m_history.history.keys()
epochsPlotV2(model_history=tst_m_history)

metrics_dict = tst_m_history.history
tst_args = {'loss': 'loss', 'joker': 'joker',
            'recall': "recall", 'binary_accuracy': 'binary_accuracy'}
