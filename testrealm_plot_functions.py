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
import time
import math

import cv2
import matplotlib.cm as cm
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


# tst_m.layers[-1].activation = None
pred = tst_m.predict(tst_img)

label_dict = tst_tf_dat.labels_map_rev
pred, class_out = tst_m_cls.predict_classes(
    x=tst_img, label_dict=label_dict, proba_threshold=0.5)

heatmap = makeGradcamHeatmap(
    tst_img, tst_m, last_conv_layer_name, pred_label_index=None)
heatmap = makeGradcamHeatmapV2(
    img_array=tst_img, model=tst_m, pred_label_index=None,
    target_layer_name=last_conv_layer_name, guided_grad=True)

plt.matshow(heatmap)
plt.matshow(tst_img.reshape((90, 90)))
plt.show()


tst_dict = tst_tf_dat.labels_map

tst_dict['all']


class GradCAM():
    def __init__(self, model, label_index_dict=None,
                 conv_last_layer=False, target_layer_name=None):
        """
        # Details:\n
            - When not None, the label_index_dict should be a dictionary where
                the keys are labels, and values are indices. One can obtain such
                dictionary from the BatchDataLoader().label_map.\n
                Example:
                {'all': 0,
                'alpha': 1,
                'beta': 2,
                'fmri': 3,
                'hig': 4,
                'megs': 5,
                'pc': 6,
                'pt': 7,
                'sc': 8}\n
        """
        # -- initialization --
        self.model = model
        if label_index_dict is not None:
            if not isinstance(label_index_dict, dict):
                raise ValueError(
                    'label_index_dict should be a dict class if not None.')
        self.label_index_dict = label_index_dict
        if target_layer_name is None:
            if conv_last_layer:
                try:
                    last_conv_layer = next(
                        x for x in model.layers[::-1] if isinstance(x, Conv2D))
                    target_layer_name = last_conv_layer.name
                except StopIteration as e:
                    print('No Conv2D layer found in the input model.')
                    raise
            else:
                target_layer_name = self._find_target_layer()
        else:
            layer_names = []
            for l in model.layers:
                layer_names.append(l.name)

            if target_layer_name not in layer_names:
                raise ValueError(
                    'Custom target_layer_name not found in model.')

        self.target_layer_name = target_layer_name

    def _find_target_layer(self):
        """find the target layer (final layer with 4D output: n, dim1, dim2, channel)"""
        for l_4d in reversed(self.model.layers):
            if len(l_4d.output_shape) == 4:
                return l_4d.name

        raise ValueError(
            'Input model has no layer with output shape=4: None, dim1, dim2, channel.')

    def compute_gradcam_heatmap(self, img_array, target_label=None, guided_grad=False):
        """
        # Arguments:\n
            image_array: np.ndarray. Normalized np.ndarray for an image of interest.\n
            target_label: None or str. Target label of interest.\n

        # Details:\n
            - When target_label=None, the method automatically uses the top predcited label.\n
        """
        if self.label_index_dict is not None and target_label is not None:
            try:
                pred_label_index = self.label_index_dict[target_label]
            except Exception as e:
                print('Check target_label.')
                raise
        else:
            pred_label_index = None

        heatmap = makeGradcamHeatmapV2(
            img_array=img_array, model=self.model,
            target_layer_name=self.target_layer_name,
            pred_label_index=pred_label_index,
            guided_grad=guided_grad)

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)


tst_cam = GradCAM(model=tst_m)


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
