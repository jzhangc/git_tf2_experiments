"""
this is test realm for plot functions with tf.keras models using the data from BatchDataLoader()

Objectives:
[x] extract values from tf.dataset objects
[ ] ROC-AUC
    [ ] calculate AUC
    [ ] construct simple ROC
    [ ] multiple classes
    [ ] from One-Hot back to labels (figure legends)
[ ] Recall and precision curve
[ ] F1 curve
[x] Update the epochsPlot function
"""


# ------ load modules ------
import os
import math

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

from utils.dl_utils import BatchMatrixLoader
from utils.plot_utils import epochsPlot
from utils.other_utils import flatten, warn

from pylab import subplots_adjust

# ------ TF device check ------
tf.config.list_physical_devices()


# ------ model ------
class CnnClassifier(Model):
    def __init__(self, initial_shape, bottleneck_dim, outpout_n, output_activation='softmax'):
        """
        Details:\n
            - Use "softmax" for binary or mutually exclusive multiclass modelling,
                and use "sigmoid" for multilabel classification.\n
        """
        super(CnnClassifier, self).__init__()
        self.output_activation = output_activation
        self.initial_shape = initial_shape
        self.bottleneck_dim = bottleneck_dim
        # CNN encoding sub layers
        self.conv2d_1 = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.01),
                               padding='same', input_shape=initial_shape)  # output: 28, 28, 16
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
        x = Input(self.initial_shape)
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

        if self.output_activation == 'sigmoid' and proba_threshold is None:
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
def tstfoo(y, pred):
    """ROC-AUC plot function"""

    return None


def choose_subplot_dimensions(k):
    if k < 4:
        return k, 1
    elif k < 11:
        return math.ceil(k/2), 2
    else:
        # I've chosen to have a maximum of 3 columns
        return math.ceil(k/4), 4


def generate_subplots(k, row_wise=False):
    nrow, ncol = choose_subplot_dimensions(k)
    # Choose your share X and share Y parameters as you wish:
    figure, axes = plt.subplots(nrow, ncol,
                                sharex=False,
                                sharey=False)

    # Check if it's an array. If there's only one plot, it's just an Axes obj
    if not isinstance(axes, np.ndarray):
        return figure, [axes]
    else:
        # Choose the traversal you'd like: 'F' is col-wise, 'C' is row-wise
        axes = axes.flatten(order=('C' if row_wise else 'F'))

        # Delete any unused axes from the figure, so that they don't show
        # blank x- and y-axis lines
        idxes_to_turn_on_ticks = []
        for idx, ax in enumerate(axes[k:]):
            figure.delaxes(ax)

            # Turn ticks on for the last ax in each column, wherever it lands
            idx_to_turn_on_ticks = idx + k - ncol if row_wise else idx + k - 1
            idxes_to_turn_on_ticks.append(idx_to_turn_on_ticks)
        idxes_to_turn_on_ticks.append(k-1)

        idxs_to_turn_off_ticks = [elem for elem in list(
            range(k)) if elem not in idxes_to_turn_on_ticks]

        axes = axes[:k]
        return figure, axes, idxs_to_turn_off_ticks


def tstPlot(model_history,
            file=None,
            figure_size=(5, 5), **kwargs):
    """
    # Purpose\n
        The plot function for epoch history from LSTM modelling\n
    # Arguments\n
        model_history: keras.callbacks.History. Keras modelling history object, generated by model.fit process.\n
        file: string or None. (optional) Directory and file name to save the figure. Use os.path.join() to generate.\n
        figure_size: float in two-tuple/list. Figure size.\n
        kwargs: generic keyword arguments for metrics to visualize in the history object. \n
    # Return\n
        The function returns a pdf figure file to the set directory and with the set file name\n
    # Details\n
        - The loss_var and accuracy_var are keys in the history.history object.\n
    """
    # -- argument check --
    if not isinstance(model_history, History):
        raise TypeError('model_history needs to be a keras History object."')

    if len(kwargs) > 0:
        hist_metrics = []
        for _, key_val in kwargs.items():
            if key_val in model_history.history:
                hist_metrics.append(key_val)
            else:
                warn(
                    f'Input metric {key_val} not found in the model_history.\n')
                pass

    # -- set up data and plotting-
    fig, axes, idxes_to_turn_off = generate_subplots(
        len(hist_metrics), row_wise=True)

    for hist_metric, ax in zip(hist_metrics, axes):
        plot_metric = np.array(tst_dict[hist_metric])
        plot_x = np.arange(1, len(plot_metric) + 1)
        plot_val_metric = np.array(
            tst_dict['val_'+hist_metric])

        ax.plot(plot_x, plot_metric, linestyle='-',
                color='blue', label='train')
        ax.plot(plot_x, plot_val_metric, linestyle='-',
                color='red', label='validation')
        ax.set_facecolor('white')
        ax.set_title(hist_metric, color='black')
        # ax.set_xlabel('Epoch', fontsize=10, color='black')
        ax.set_ylabel(hist_metric, fontsize=10, color='black')
        ax.legend()
        ax.tick_params(labelsize=5, color='black', labelcolor='black')

        plt.setp(ax.spines.values(), color='black')

    for i in idxes_to_turn_off:
        plt.setp(axes[i].get_xticklabels(), visible=False)
    plt.xlabel('Epoch')
    plt.tight_layout()

    fig.set_facecolor('white')
    fig

    # - save output -
    if file is not None:
        full_path = os.path.normpath(os.path.abspath(os.path.expanduser(file)))
        if not os.path.isfile(full_path):
            raise ValueError('Invalid input file or input file not found.')
        else:
            plt.savefig(full_path, dpi=600,
                        bbox_inches='tight', facecolor='white')

    return fig, ax


# ------ data ------
# -- batch loader data --
tst_tf_dat = BatchMatrixLoader(filepath='./data/tf_data', target_file_ext='txt',
                               manual_labels=None, model_type='classification',
                               multilabel_classification=False, label_sep=None,
                               x_scaling='minmax', x_min_max_range=[0, 1], resmaple_method='random',
                               training_percentage=0.8, verbose=False, random_state=1)
tst_tf_train, tst_tf_test = tst_tf_dat.generate_batched_data(batch_size=10)

# below: manual labels
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
tst_m = CnnClassifier(initial_shape=(90, 90, 1),
                      bottleneck_dim=64, outpout_n=len(tst_tf_dat.lables_count),
                      output_activation='sigmoid')
tst_m.model().summary()

# -- training --
tst_m.compile(optimizer=optm, loss="categorical_crossentropy",
              metrics=['categorical_accuracy', tf.keras.metrics.Recall()])  # for mutually exclusive multiclass
tst_m.compile(optimizer=optm, loss="binary_crossentropy",
              metrics=['binary_accuracy', tf.keras.metrics.Recall()])  # for multilabel
tst_m_history = tst_m.fit(tst_tf_train, epochs=80,
                          callbacks=callbacks,
                          validation_data=tst_tf_test)


label_dict = tst_tf_dat.labels_map_rev
tst_m_history.history.keys()

# - single label multiclass -
pred = tst_m.predict(tst_tf_test)
proba, pred_class = tst_m.predict_classes(
    label_dict=label_dict, x=tst_tf_test, proba_threshold=0.5)
# to_categorical(np.argmax(pred, axis=1), len(tst_tf_dat.lables_count))

# - multilabel -
pred_class = tst_m.predict_classes(
    label_dict=label_dict, x=tst_tf_test, proba_threshold=0.5)

tst_t = np.ndarray((0, 9))
for _, b in tst_tf_test:
    # print(b.numpy())
    bn = b.numpy()
    # print(type(bn))
    tst_t = np.concatenate((tst_t, bn), axis=0)
tst_t.shape


# ------ plot function ------
epochsPlot(model_history=tst_m_history,
           accuracy_var='binary_accuracy',
           val_accuracy_var='val_binary_accuracy')

tst_dict = tst_m_history.history
tst_args = {'loss': 'loss', 'joker': 'joker',
            'recall': "recall", 'binary_accuracy': 'binary_accuracy'}

hist_metrics = []
for _, key_val in tst_args.items():
    if key_val in tst_dict:
        hist_metrics.append(key_val)
    else:
        warn(
            f'Input metric {key_val} not found in the model_history.\n')
        pass


figure, axes, idxes_to_turn_off = generate_subplots(
    len(hist_metrics), row_wise=True)

for hist_metric, ax in zip(hist_metrics, axes):
    plot_metric = np.array(tst_dict[hist_metric])
    plot_x = np.arange(1, len(plot_metric) + 1)
    plot_val_metric = np.array(
        tst_dict['val_'+hist_metric])

    ax.plot(plot_x, plot_metric, linestyle='-',
            color='blue', label='train')
    ax.plot(plot_x, plot_val_metric, linestyle='-',
            color='red', label='validation')
    ax.set_facecolor('white')
    ax.set_title(hist_metric, color='black')
    # ax.set_xlabel('Epoch', fontsize=10, color='black')
    ax.set_ylabel(hist_metric, fontsize=10, color='black')
    ax.legend()
    ax.tick_params(labelsize=5, color='black', labelcolor='black')

    plt.setp(ax.spines.values(), color='black')

for i in idxes_to_turn_off:
    plt.setp(axes[i].get_xticklabels(), visible=False)

plt.xlabel('Epoch')
plt.tight_layout()

x_variable = list(range(-5, 6))
parameters = list(range(0, 13))

figure, axes, idxes_to_turn_off = generate_subplots(
    len(parameters), row_wise=True)
for parameter, ax in zip(parameters, axes):
    ax.plot(x_variable, [x**parameter for x in x_variable])
    ax.set_title(label="y=x^{}".format(parameter))

for i in idxes_to_turn_off:
    plt.setp(axes[i].get_xticklabels(), visible=False)

plt.tight_layout()


fpr, tpr, thresholds = roc_curve(tst_t[:, 0], pred[:, 0])
auc_score = roc_auc_score(tst_t[:, 0], pred[:, 0])

plt.plot(fpr, tpr)
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
