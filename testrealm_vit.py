"""
This is the testrealem for visual transformer (ViT): square matrix -> classes

To do:
    [ ] Build a ViT model for image classification
    [ ] Implement multihead attention visualization
    [ ] Implement GradCAM for post-hoc attention
"""


# ------ load modules ------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Display
from IPython.display import Image, display
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import History
from tensorflow.python.ops.gen_array_ops import tensor_scatter_max

from utils.dl_utils import (BatchMatrixLoader, WarmUpCosineDecayScheduler,
                            makeGradcamHeatmap, makeGradcamHeatmapV2)
from utils.models import CnnClassifier, CnnClassifierFuncAPI
from utils.other_utils import error, flatten, warn
from utils.plot_utils import epochsPlotV2, lrSchedulerPlot, rocaucPlot


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


# - multilabel -


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
