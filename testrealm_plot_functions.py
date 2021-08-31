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
from tensorflow.python.keras.callbacks import History
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.python.types.core import Value

from utils.dl_utils import BatchMatrixLoader
from utils.plot_utils import epochsPlotV2
from utils.other_utils import flatten, warn

# ------ TF device check ------
tf.config.list_physical_devices()


# ------ model ------
class CnnClassifier(Model):
    def __init__(self, initial_x_shape, y_len,
                 bottleneck_dim, outpout_n, output_activation='softmax', multilabel=False):
        """
        # Details:\n
            - Use "softmax" for binary or mutually exclusive multiclass modelling,
                and use "sigmoid" for multilabel classification.\n
            - y_len: this is the length of y.
                y_len = 1 or 2: binary classification.
                y_len >= 2: multiclass or multilabel classification.
        """
        super(CnnClassifier, self).__init__()

        # - initialization and argument check-
        self.initial_x_shape = initial_x_shape
        self.y_len = y_len
        self.bottleneck_dim = bottleneck_dim
        self.multilabel = multilabel
        if multilabel and output_activation != 'softmax':
            warn(
                'Activation automatically set to \'sigmoid\' for multilabel classification.')
            self.output_activation = 'sigmoid'
        else:
            self.output_activation = output_activation

        # - CNN layers -
        # CNN encoding sub layers
        self.conv2d_1 = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.01),
                               padding='same', input_shape=initial_x_shape)  # output: 28, 28, 16
        self.bn1 = BatchNormalization()
        self.leakyr1 = LeakyReLU()
        self.maxpooling_1 = MaxPooling2D((2, 2))  # output: 14, 14, 16
        self.conv2d_2 = Conv2D(8, (3, 3), activation='relu',
                               padding='same')  # output: 14, 14, 8
        self.bn2 = BatchNormalization()
        self.leakyr2 = LeakyReLU()
        # self.maxpooling_2 = MaxPooling2D((2, 2))  # output: 7, 7, 8
        self.maxpooling_2 = MaxPooling2D((5, 5))  # output: 9, 9, 8
        self.fl = Flatten()  # 7*7*8=392
        self.dense1 = Dense(bottleneck_dim, activation='relu',
                            activity_regularizer=tf.keras.regularizers.l2(l2=0.01))
        self.encoded = LeakyReLU()
        self.dense2 = Dense(outpout_n, activation=output_activation)

    def call(self, input):
        x = self.conv2d_1(input)
        x = self.bn1(x)
        x = self.leakyr1(x)
        x = self.maxpooling_1(x)
        x = self.conv2d_2(x)
        x = self.bn2(x)
        x = self.leakyr2(x)
        x = self.maxpooling_2(x)
        x = self.fl(x)
        x = self.dense1(x)
        x = self.encoded(x)
        x = self.dense2(x)
        return x

    def model(self):
        """
        This method enables correct model.summary() results:
        model.model().summary()
        """
        x = Input(self.initial_x_shape)
        return Model(inputs=[x], outputs=self.call(x))

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
        proba = self.predict(x, batch_size=batch_size, verbose=verbose)
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
                    for m, n in enumerate(idx_decrease):
                        print(f'\t{label_dict[m]}: {sample_proba[n]*100:.2f}%')
                # break

            multilabel_out = pd.DataFrame(multilabel_res, dtype=int)
            multilabel_out.columns = res_colnames

            return proba_res, multilabel_out
        else:
            raise NotImplemented(
                f'predict_classes method not implemented for {self.output_activation}')


# ------ functions ------
def foo(**kwargs):
    print('Saving...', end='')
    time.sleep(2)
    print('Done!')


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
                               x_scaling='minmax', x_min_max_range=[0, 1], resmaple_method='random',
                               training_percentage=0.8, verbose=False, random_state=1)
tst_tf_train, tst_tf_test = tst_tf_dat.generate_batched_data(batch_size=10)


# ------ training ------
# -- early stop and optimizer --
earlystop = EarlyStopping(monitor='val_loss', patience=5)
callbacks = [earlystop]
optm = Adam(learning_rate=0.001, decay=0.001/80)  # decay?? lr/epoch

# -- model --
# - multiclass -
tst_m = CnnClassifier(initial_x_shape=(90, 90, 1), y_len=len(tst_tf_dat.labels_map_rev), multilabel=False,
                      bottleneck_dim=64, outpout_n=len(tst_tf_dat.lables_count),
                      output_activation='softmax')

# - multilabel -
tst_m = CnnClassifier(initial_x_shape=(90, 90, 1), y_len=len(tst_tf_dat.labels_map_rev), multilabel=True,
                      bottleneck_dim=64, outpout_n=len(tst_tf_dat.lables_count),
                      output_activation='sigmoid')

# - model summary -
tst_m.model().summary()

# -- training --
# - multiclass -
tst_m.compile(optimizer=optm, loss="categorical_crossentropy",
              metrics=['categorical_accuracy', tf.keras.metrics.Recall()])  # for mutually exclusive multiclass

# - multilabel -
tst_m.compile(optimizer=optm, loss="binary_crossentropy",
              metrics=['binary_accuracy', tf.keras.metrics.Recall()])  # for multilabel

# - fitting -
tst_m_history = tst_m.fit(tst_tf_train, epochs=80,
                          callbacks=callbacks,
                          validation_data=tst_tf_test)
tst_m_history.history.keys()


# -- prediciton --
label_dict = tst_tf_dat.labels_map_rev
# pred = tst_m.predict(tst_tf_test)

# - single label multiclass -
proba, pred_class = tst_m.predict_classes(
    label_dict=label_dict, x=tst_tf_test)
# to_categorical(np.argmax(pred, axis=1), len(tst_tf_dat.lables_count))

# - multilabel -
proba, pred_class = tst_m.predict_classes(
    label_dict=label_dict, x=tst_tf_test, proba_threshold=0.5)


# ------ test plot functions ------
def tstfoo(classifier, x, y=None, label_dict=None, legend_pos='inside', **kwargs):
    """
    # Purpose\n
        To calculate and plot ROC-AUC for binary or mutual multiclass classification.

    # Arguments\n
        classifier: tf.keras.model subclass. 
            These classes were created with a custom "predict_classes" method, along with other smaller custom attributes.\n
        x: tf.Dataset or np.ndarray. Input x data for prediction.\n
        y: None or np.ndarray. Only needed when x is a np.ndarray. Label information.\n
        label_dict: dict. Dictionary with index (integers) as keys.\n
        legend_pos: str. Legend position setting. Can be set to 'none' to hide legends.\n
        **kwargs: additional arguments for the classifier.predict_classes.\n

    # Return\n
        - AUC scores for all the classes.\n
        - Plot objects "fg" and "ax" from matplotlib.\n
        - Order: auc_res, fg, ax.\n

    # Details\n
        - The function will throw an warning if multilabel classifier is used.\n        
        - The output auc_res is a pd.DataFrame. 
            Column names:  'label', 'auc', 'thresholds', 'fpr', 'tpr'.
            Since the threshold contains multiple values, so as the corresponding 'fpr' and 'tpr',
            the value of these columns is a list.\n
        - For label_dict, this is a dictionary with keys as index integers.
            Example:
            {0: 'all', 1: 'alpha', 2: 'beta', 3: 'fmri', 4: 'hig', 5: 'megs', 6: 'pc', 7: 'pt', 8: 'sc'}.
            This can be derived from the "label_map_rev" attribtue from BatchDataLoader class.\n

    # Note\n
        - need to test the non-tf.Dataset inputs.\n
        - In the case of using tf.Dataset as x, y is not needed.\n
    """
    # - probability calculation -
    # more model classes are going to be added.
    if not isinstance(classifier, CnnClassifier):
        raise ValueError('The classifier should be one of \'CnnClassifier\'.')

    if not isinstance(x, (np.ndarray, tf.data.Dataset)):
        raise TypeError(
            'x needs to be either a np.ndarray or tf.data.Dataset class.')

    if isinstance(x, np.ndarray):
        if y is None:
            raise ValueError('Set y (np.ndarray) when x is np.ndarray')
        elif not isinstance(y, np.ndarray):
            raise ValueError('Set y (np.ndarray) when x is np.ndarray')
        elif y.shape[-1] != classifier.y_len:
            raise ValueError(
                'Make sure y is the same length as classifier.y_len.')

    if classifier.multilabel:
        warn('ROC-AUC for multilabel models should not be used.')

    if legend_pos not in ['none', 'inside', 'outside']:
        raise ValueError(
            'Options for legend_pos are \'none\', \'inside\' and \'outside\'.')

    if label_dict is None:
        raise ValueError('Set label_dict.')

    # - make prediction -
    proba, _ = classifier.predict_classes(x=x, label_dict=label_dict, **kwargs)
    proba_np = proba.to_numpy()

    # - set up plotting data -
    if isinstance(x, tf.data.Dataset):
        t = np.ndarray((0, proba.shape[-1]))
        for _, b in x:
            # print(b.numpy())
            bn = b.numpy()
            # print(type(bn))
            t = np.concatenate((t, bn), axis=0)
    else:
        t = y

    # - calculate AUC and plotting -
    auc_res = pd.DataFrame(
        columns=['label', 'auc', 'thresholds', 'fpr', 'tpr'])

    fg, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k--')
    for class_idx, auc_class in enumerate(proba.columns):
        fpr, tpr, thresholds = roc_curve(
            t[:, class_idx], proba_np[:, class_idx])
        auc_score = roc_auc_score(t[:, class_idx], proba_np[:, class_idx])
        auc_res.loc[class_idx] = [auc_class, auc_score,
                                  thresholds, fpr, tpr]  # store results

        ax.plot(fpr, tpr, label=f'{auc_class} vs rest: {auc_score:.3f}')
    ax.set_title('ROC-AUC')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if legend_pos == 'inside':
        ax.legend(loc='best')
    elif legend_pos == 'outside':
        ax.legend(loc='best', bbox_to_anchor=(1.01, 1.0))
    plt.show()

    return auc_res, fg, ax


# - ROC-AUC curve -
tst_t = np.ndarray((0, 10))
for _, b in tst_tf_test:
    # print(b.numpy())
    bn = b.numpy()
    # print(type(bn))
    tst_t = np.concatenate((tst_t, bn), axis=0)
tst_t.shape

proba_np = proba.to_numpy()

tst_tf_train


fg, ax = plt.subplots()
ax.plot([0, 1], [0, 1], 'k--')
auc_res = pd.DataFrame(columns=['label', 'auc', 'thresholds', 'fpr', 'tpr'])
for class_idx, auc_class in enumerate(proba.columns):
    fpr, tpr, thresholds = roc_curve(
        tst_t[:, class_idx], proba_np[:, class_idx])
    auc_score = roc_auc_score(tst_t[:, class_idx], proba_np[:, class_idx])
    auc_res.loc[class_idx] = [auc_class, auc_score, thresholds, fpr, tpr]

    ax.plot(fpr, tpr, label=f'{auc_class} vs rest: {auc_score:.3f}')
ax.set_title('ROC-AUC')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='best', bbox_to_anchor=(1.01, 1.0))
plt.show()


auc_res, _, _ = tstfoo(classifier=tst_m, x=tst_tf_test,
                       label_dict=label_dict, legend_pos='none')


if isinstance(tst_m, CnnClassifier):
    print('y')
else:
    print('n')

# - eppochs plot function -
epochsPlotV2(model_history=tst_m_history)

metrics_dict = tst_m_history.history
tst_args = {'loss': 'loss', 'joker': 'joker',
            'recall': "recall", 'binary_accuracy': 'binary_accuracy'}
