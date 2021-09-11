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

from utils.dl_utils import BatchMatrixLoader, WarmUpCosineDecayScheduler
from utils.plot_utils import epochsPlotV2, rocaucPlot, lrSchedulerPlot
from utils.other_utils import flatten, warn
from utils.models import CnnClassifier

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ------ TF device check ------
tf.config.list_physical_devices()


# ------ model ------
class CnnClassifierFuncAPI():
    def __init__(self, initial_x_shape, y_len,
                 bottleneck_dim, output_n, output_activation='softmax', multilabel=False):
        """
        # Details:\n
            - Use "softmax" for binary or mutually exclusive multiclass modelling,
                and use "sigmoid" for multilabel classification.\n
            - y_len: this is the length of y.
                y_len = 1 or 2: binary classification.
                y_len >= 2: multiclass or multilabel classification.
        """
        # super(CnnClassifierFuncAPI, self).__init__()

        # -- initialization and argument check--
        self.initial_x_shape = initial_x_shape
        self.y_len = y_len
        self.bottleneck_dim = bottleneck_dim
        self.multilabel = multilabel
        self.output_n = output_n
        if multilabel and output_activation == 'softmax':
            warn(
                'Activation automatically set to \'sigmoid\' for multilabel classification.')
            self.output_activation = 'sigmoid'
        else:
            self.output_activation = output_activation

        # -- CNN model --
        model_input = tf.keras.Input(shape=self.initial_x_shape)
        # CNN encoding sub layers
        x = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.01),
                   padding='same')(model_input)  # output: 28, 28, 16
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2))(x)  # output: 14, 14, 16
        x = Conv2D(8, (3, 3), activation='relu',
                   padding='same', name='last_conv')(x)  # output: 14, 14, 8
        x = BatchNormalization()(x)
        x = LeakyReLU(name='grads_cam_dense')(x)
        # x = MaxPooling2D((2, 2))(x)  # output: 7, 7, 8
        x = MaxPooling2D((5, 5))(x)  # output: 9, 9, 8
        x = Flatten()(x)  # 7*7*8=392
        x = Dense(self.bottleneck_dim, activation='relu',
                  activity_regularizer=tf.keras.regularizers.l2(
                      l2=0.01))(x)
        x = LeakyReLU()(x)
        x = Dense(self.output_n, activation=self.output_activation)(x)

        self.m = Model(model_input, x)

    def model(self):
        return self.m

    def predict_classes(self, label_dict,
                        x, proba_threshold=None,
                        batch_size=32, verbose=1):
        """
        # Purpose:\n
            Generate class predictions for the input samples batch by batch.\n
        # Arguments:\n
            label_dict: dict. Dictionary with index (integers) as keys.\n
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).\n
            proba_threshold: None or float. The probability threshold to allocate class labels to multilabel prediction.\n
            batch_size: integer.\n
            verbose: verbosity mode, 0 or 1.\n
        # Return:\n
            Two pandas dataframes for probability results and 0/1 classification results, in this order.\n
        # Details:\n
            - For label_dict, this is a dictionary with keys as index integers.
                Example:
                {0: 'all', 1: 'alpha', 2: 'beta', 3: 'fmri', 4: 'hig', 5: 'megs', 6: 'pc', 7: 'pt', 8: 'sc'}.
                This can be derived from the "label_map_rev" attribtue from BatchDataLoader class.\n
            - For binary classification, the length of the label_dict should be 1.
                Example: {0: 'case'}. \n
        """
        # - argument check -
        if not isinstance(label_dict, dict):
            raise ValueError('label_dict needs to be a dictionary.')
        else:
            label_keys = list(label_dict.keys())

        if not all(isinstance(key, int) for key in label_keys):
            raise ValueError('The keys in label_dict need to be integers.')

        if self.multilabel and proba_threshold is None:
            raise ValueError(
                'Set proba_threshold for multilabel class prediction.')

        # - set up output column names -
        if len(label_dict) == 1:
            label_dict[0] = label_dict.pop(label_keys[0])

        res_colnames = [None]*len(label_dict)
        for k, v in label_dict.items():
            res_colnames[k] = v

        # - prediction -
        proba = self.m.predict(x, batch_size=batch_size, verbose=verbose)
        if proba.min() < 0. or proba.max() > 1.:
            warn('Network returning invalid probability values.',
                 'The last layer might not normalize predictions',
                 'into probabilities (like softmax or sigmoid would).')

        proba_res = pd.DataFrame(proba, dtype=float)
        proba_res.columns = res_colnames

        # self.proba = proba
        if self.output_activation == 'softmax':
            if proba.shape[-1] > 1:
                multiclass_res = to_categorical(
                    proba.argmax(axis=1), proba.shape[-1])
            else:
                multiclass_res = (proba > 0.5).astype('int32')

            multiclass_out = pd.DataFrame(multiclass_res, dtype=int)
            multiclass_out.columns = res_colnames

            return proba_res, multiclass_out

        elif self.output_activation == 'sigmoid':
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

            if verbose:
                idxs = np.argsort(proba)
                for i, j in enumerate(idxs):
                    print(f'Sample: {i}')
                    idx_decrease = j[::-1]  # [::-1] to make decreasing order
                    sample_proba = proba[i]
                    for n in idx_decrease:
                        print(f'\t{label_dict[n]}: {sample_proba[n]*100:.2f}%')
                # break

            multilabel_out = pd.DataFrame(multilabel_res, dtype=int)
            multilabel_out.columns = res_colnames

            return proba_res, multilabel_out
        else:
            raise NotImplemented(
                f'predict_classes method not implemented for {self.output_activation}')


# ------ functions ------
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(
        superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


# ------ data ------
# - multiclass -
tst_tf_dat = BatchMatrixLoader(filepath='./data/tf_data', target_file_ext='txt',
                               manual_labels=None, model_type='classification',
                               multilabel_classification=False, label_sep=None,
                               x_scaling='minmax', x_min_max_range=[0, 1], resmaple_method='random',
                               training_percentage=0.8, verbose=False, random_state=1)
tst_tf_train, tst_tf_test = tst_tf_dat.generate_batched_data(batch_size=10)

# - multilabel with manual labels -
tst_tf_dat = BatchMatrixLoader(filepath='./data/tf_data', target_file_ext='txt',
                               manual_labels='./data/tst_annot.csv',
                               manual_labels_fileNameVar='filename', manual_labels_labelVar='label',
                               model_type='classification',
                               multilabel_classification=True, label_sep='_',
                               x_scaling='minmax', x_min_max_range=[-1, 1], resmaple_method='random',
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
    x=tst_img, label_dict=label_dict)

heatmap = make_gradcam_heatmap(tst_img, tst_m, last_conv_layer_name)

plt.matshow(heatmap)
plt.matshow(tst_img.reshape((90, 90)))
plt.show()


jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]

heatmap = np.uint8(255 * heatmap)
jet_heatmap = jet_colors[heatmap]

# Create an image with RGB colorized heatmap
jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize(img_size)
jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

# Superimpose the heatmap on original image
superimposed_img = jet_heatmap * 0.8 + i_numpy[0, :, :, :]
superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
superimposed_img


f = np.loadtxt('/Users/jingzhang/Documents/git_repo/git_tf2_experiments/data/tf_data/all_megs_pc/all_megs_fused_mtx_v2_PC03.txt').astype('float32')

X=f
if self.x_scaling == 'max':
    X = X/X.max()
elif self.x_scaling == 'minmax':
    Min = -1
    Max = 1
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X = X_std * (Max - Min) + Min

if self.new_shape is not None:  # reshape
    X = np.reshape(X, self.new_shape)
else:
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))


X[np.arange(X.shape[0])[:,None] > np.arange(X.shape[1])] = 0
np.fill_diagonal(X, 0)


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
