"""utilities for plots"""

# ------ modules ------
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from utils.other_utils import warn
from utils.models import CnnClassifier
from tensorflow.python.keras.callbacks import History
from sklearn.metrics import roc_auc_score, roc_curve


# ----- functions ------
def epochsPlot(model_history,
               file=None,
               loss_var='loss', val_loss_var='val_loss',
               accuracy_var=None, val_accuracy_var=None,
               plot_title_loss='Loss', plot_title_acc='Accuracy',
               figure_size=(5, 5)):
    """
    # Purpose\n
        The plot function for epoch history from LSTM modelling\n
    # Arguments\n
        model_history: keras.callbacks.History. Keras modelling history object, generated by model.fit process.\n
        file: string or None. (optional) Directory and file name to save the figure. Use os.path.join() to generate.\n
        loss_var: string. Variable name for loss in the model history.\n
        plot_title: string. Plot title.\n
        xlabel: string. X-axis title.\n
        ylabel: string. Y-axis title.\n
        figure_size: float in two-tuple/list. Figure size.\n
    # Return\n
        The function returns a pdf figure file to the set directory and with the set file name\n
    # Details\n
        - The loss_var and accuracy_var are keys in the history.history object.\n
    """
    # -- argument check --
    if not isinstance(model_history, History):
        raise TypeError('model_history needs to be a keras History object."')

    if not all(hist_key in model_history.history for hist_key in [loss_var, val_loss_var]):
        raise ValueError(
            'Make sure both loss_var and val_loss_var exist in model_history.')

    if all(acc is not None for acc in [accuracy_var, val_accuracy_var]):
        if not all(hist_key in model_history.history for hist_key in [accuracy_var, val_accuracy_var]):
            raise ValueError(
                'Make sure both accuracy_var and val_accuracy_var exist in model_history.')
        else:
            acc_plot = True
    elif any(acc is not None for acc in [accuracy_var, val_accuracy_var]):
        try:
            raise ValueError
        except ValueError as e:
            warn('Only one of accuracy_var, val_accuracy_var are set.',
                 'Proceed with only loss plot.')
        finally:
            acc_plot = False
    else:
        acc_plot = False

    # -- prepare data --
    plot_loss = np.array(model_history.history[loss_var])  # RMSE
    plot_val_loss = np.array(model_history.history[val_loss_var])  # RMSE
    plot_x = np.arange(1, len(plot_loss) + 1)

    if acc_plot:
        plot_acc = np.array(model_history.history[accuracy_var])
        plot_val_acc = np.array(
            model_history.history[val_accuracy_var])

    # -- plotting --
    if acc_plot:  # two plots
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(plot_x, plot_loss, linestyle='-',
                   color='blue', label='train')
        ax[0].plot(plot_x, plot_val_loss, linestyle='-',
                   color='red', label='validation')
        ax[0].set_facecolor('white')
        ax[0].set_title(plot_title_loss, color='black')
        ax[0].set_xlabel('Epoch', fontsize=10, color='black')
        ax[0].set_ylabel('Loss', fontsize=10, color='black')
        ax[0].legend()
        ax[0].tick_params(labelsize=5, color='black', labelcolor='black')

        ax[1].plot(plot_x, plot_acc, linestyle='-',
                   color='blue', label='train')
        ax[1].plot(plot_x, plot_val_acc, linestyle='-',
                   color='red', label='validation')
        ax[1].set_facecolor('white')
        ax[1].set_title(plot_title_acc, color='black')
        ax[1].set_xlabel('Epoch', fontsize=10, color='black')
        ax[1].set_ylabel('Accuracy', fontsize=10, color='black')
        ax[1].legend()
        ax[1].tick_params(labelsize=5, color='black', labelcolor='black')

        plt.setp(ax[0].spines.values(), color='black')
        plt.setp(ax[1].spines.values(), color='black')
    else:
        fig, ax = plt.subplots(figsize=figure_size)
        ax.plot(plot_x, plot_loss, linestyle='-', color='blue', label='train')
        ax.plot(plot_x, plot_val_loss, linestyle='-',
                color='red', label='validation')
        ax.set_facecolor('white')
        ax.set_title(plot_title_loss, color='black')
        ax.set_xlabel('Epoch', fontsize=10, color='black')
        ax.set_ylabel('Accuracy', fontsize=10, color='black')
        ax.legend()
        ax.tick_params(labelsize=5, color='black', labelcolor='black')

        plt.setp(ax.spines.values(), color='black')

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


def choose_subplot_dimensions(k):
    """
    # Purpose\n
        To generate subplot dimensions for matplotlib.\n

    # Arguments\n
        k: int. Number of subplots.\n

    # Return\n
        Number of rows and columns (int, int) as subplot dimensions.\n

    # Details\n
        - If k < 4, one column.\n
        - If k < 11, two columns.\n
        - If k >= 11, three columns.\n 
    """
    if k < 4:
        return k, 1
    elif k < 11:
        return math.ceil(k/2), 2
    else:
        # I've chosen to have a maximum of 3 columns
        return math.ceil(k/4), 4


def generate_subplots(k):
    """
    # Purpose\n
        Generate matplotlib figure and axes with k number of subplots.\n

    # Arguments\n
        k: int. Number of subplots.\n

    # Return\n
        - If more than one metric, function returns figure, axes, idxs_to_turn_off_ticks\n
        - If only one metric, function returns figure, axes\n
    """
    nrow, ncol = choose_subplot_dimensions(k)
    # - set up initial figures and axes -
    figure, axes = plt.subplots(nrow, ncol,
                                sharex=False,
                                sharey=False)

    # - check if it's an array -
    if not isinstance(axes, np.ndarray):
        return figure, [axes]
    else:
        axes = axes.flatten(order='C')  # 'C' is row-wise
        # delete any unused axes from the figure
        for ax in axes[k:]:
            figure.delaxes(ax)

        # extract indices for the axes to show/hide tick labels
        idxes_to_turn_on_ticks = []
        for idx in range(ncol):
            idx_to_turn_on_ticks = idx + k - ncol
            idxes_to_turn_on_ticks.append(idx_to_turn_on_ticks)
        idxs_to_turn_off_ticks = [elem for elem in list(
            range(k)) if elem not in idxes_to_turn_on_ticks]

        # finalize axes
        axes = axes[:k]
        return figure, axes, idxs_to_turn_off_ticks


def epochsPlotV2(model_history,
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
    metrics_dict = model_history.history

    # - set up metrics names -
    hist_metrics = []
    if len(kwargs) > 0:
        for _, key_val in kwargs.items():
            if key_val in model_history.history:
                hist_metrics.append(key_val)
            else:
                warn(
                    f'Input metric {key_val} not found in the model_history.')
                pass
    else:
        for key_val in model_history.history.keys():
            hist_metrics.append(key_val)

        hist_metrics = [x for x in hist_metrics if 'val_' not in x]

    if len(hist_metrics) == 0:
        raise ValueError('No valid metrics found to plot.')

    # -- set up data and plotting-
    if len(hist_metrics) == 1:
        fig, axes = generate_subplots(
            len(hist_metrics))
    else:
        fig, axes, idxes_to_turn_off = generate_subplots(
            len(hist_metrics))

    for hist_metric, ax in zip(hist_metrics, axes):
        plot_metric = np.array(metrics_dict[hist_metric])
        plot_x = np.arange(1, len(plot_metric) + 1)

        try:
            plot_val_metric = np.array(
                metrics_dict['val_'+hist_metric])
            ax.plot(plot_x, plot_val_metric, linestyle='-',
                    color='red', label='validation')
        except:
            warn(f'{hist_metric} on validation data not found.')
        finally:
            ax.plot(plot_x, plot_metric, linestyle='-',
                    color='blue', label='train')
            ax.set_facecolor('white')
            ax.set_title(hist_metric, color='black')
            # ax.set_xlabel('Epoch', fontsize=10, color='black')
            ax.set_ylabel(hist_metric, fontsize=10, color='black')
            ax.legend()
            ax.tick_params(labelsize=5, color='black', labelcolor='black')

            plt.setp(ax.spines.values(), color='black')

    if len(hist_metrics) > 1:
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
            print(f'Saveing plot as {file}...', end='')
            plt.savefig(full_path, dpi=600,
                        bbox_inches='tight', facecolor='white')
            print('Done!')

    return fig, ax


def rocaucPlot(classifier, x, y=None, label_dict=None, legend_pos='inside', **kwargs):
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
