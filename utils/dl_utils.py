"""utilities for deep learning (excluding models)

To be implemented:
    [ ] logger
"""

# ------ modules ------
import os
import random

import cv2
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import Image, display
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D

from utils.data_utils import (adjmatAnnotLoader, adjmatAnnotLoaderV2,
                              getSelectedDataset, labelMapping, labelOneHot)
from utils.other_utils import FileError, VariableNotFoundError, warn


# ------ functions ------
def cosinDecayWithWarmup(global_step,
                         learning_rate_base,
                         total_steps,
                         warmup_learning_rate=0.0,
                         warmup_steps=0,
                         hold_base_rate_steps=0):
    """
    # Purpose\n
    Cosine decay schedule with warm up period.\n

    # Arguments\n
        global_step {int} -- global step.\n
        learning_rate_base {float} -- base learning rate.\n
        total_steps {int} -- total number of training steps.\n

    # Keyword Arguments\n
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})\n
        warmup_steps {int} -- number of warmup steps. (default: {0})\n
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})\n

    # Returns\n
        a float representing learning rate.\n

    # Details\n
        Cosine annealing learning rate as described in:
        Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
        ICLR 2017. https://arxiv.org/abs/1608.03983\n

        In this schedule, the learning rate grows linearly from warmup_learning_rate 
            to learning_rate_base for warmup_steps, then transitions to a cosine decay
            schedule.\n

        Stole from here:
            https://github.com/Tony607/Keras_Bag_of_Tricks/blob/master/warmup_cosine_decay_scheduler.py\n

    # Raises\n
        ValueError: if warmup_learning_rate is larger than learning_rate_base,
            or if warmup_steps is larger than total_steps.\n
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


def getImgArray(img_path, size):
    """
    Not used here. Stole from the grad-CAM page from keras.io\n
    """
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def makeGradcamHeatmap(img_array, model, target_layer_name, pred_label_index=None):
    """
    Purpose:\n
        heatmap gerneation for GradCAM, from keras.io.\n

    Details:\n
        - The pred_label_index is unsorted. 
            One can get the info from from the "label_map_rev" attribtue from BatchDataLoader class.
            Example (dict keys are indices): 
            {0: 'all', 1: 'alpha', 2: 'beta', 3: 'fmri', 4: 'hig', 5: 'megs', 6: 'pc', 7: 'pt', 8: 'sc'}\n
    """
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

    # tf.maximu(heatmap, 0) is to apply relu to the heatmap (remove negative values)
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # -- resize to the image dimension --
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmap = cv2.resize(heatmap.numpy(), (w, h))

    return heatmap.numpy()


def makeGradcamHeatmapV2(img_array, model, target_layer_name, pred_label_index=None, guided_grad=False, eps=1e-8):
    """
    Purpose:\n
        V2 of heatmap gerneation for GradCAM, with guided grad functionality.\n

    Details:\n
        - The pred_label_index is unsorted. 
            One can get the info from from the "label_map_rev" attribtue from BatchDataLoader class.
            Example (dict keys are indices): 
            {0: 'all', 1: 'alpha', 2: 'beta', 3: 'fmri', 4: 'hig', 5: 'megs', 6: 'pc', 7: 'pt', 8: 'sc'}\n
    """
    # -- construct our gradient model by supplying (1) the inputs
    # to our pre-trained model, (2) the output of the (presumably)
    # final 4D layer in the network, and (3) the output of the
    # softmax activations from the model --
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(target_layer_name).output, model.output])

    # -- record operations for automatic differentiation --
    with tf.GradientTape() as tape:
        # cast the image tensor to a float-32 data type, pass the
        # image through the gradient model, and grab the loss
        # associated with the specific class index
        inputs = tf.cast(img_array, tf.float32)
        (target_layer_output, preds) = grad_model(inputs)

        # the function automatically calculate for the top predicted
        # label when pred_label_index=None
        if pred_label_index is None:
            pred_label_index = tf.argmax(preds[0])

        loss = preds[:, pred_label_index]

    # -- use automatic differentiation to compute the gradients --
    grads = tape.gradient(loss, target_layer_output)

    # -- use guided grad or not --
    if guided_grad:
        # compute the guided gradients
        casttarget_layer_output = tf.cast(target_layer_output > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = casttarget_layer_output * castGrads * grads

        # the guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        # guidedGrads = guidedGrads[0]
        grads = guidedGrads[0]
    else:
        grads = grads[0]

    # -- compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    # from (dim1, dim2, c) to (c) --
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # -- the convolution have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch --
    target_layer_output = target_layer_output[0]
    cam = tf.reduce_sum(tf.multiply(weights, target_layer_output), axis=-1)
    heatmap = cam

    # -- normalize the heatmap such that all values lie in the range
    # [0, 1], and optionally scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer --
    heatmap = tf.maximum(heatmap, 0)  # relu heatmap
    heatmap = heatmap / tf.math.reduce_max(heatmap)  # scale to [0,]
    # numer = heatmap - np.min(heatmap)
    # denom = (heatmap.max() - heatmap.min()) + eps
    # heatmap = numer / denom
    # heatmap = (heatmap * 255).astype("uint8")  # convert back heatmap into 255 scale

    # -- grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions --
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmap = cv2.resize(heatmap.numpy(), (w, h))

    # return the resulting heatmap
    return heatmap


def displayGradcam(image_array, heatmap, cam_path=None, alpha=0.4):
    """
    # Arguments:\n
        cam_path: None or string. The file name with directory name.\n

    # Details:\n
        - The superimposed image will be not saved when cam_path=None.\n
    """
    # argument check
    img = image_array
    if len(img.shape) == 4:
        img = img.reshape(img.shape[1:])

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    img = np.unit8(255 * img)

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
    if cam_path is not None:
        superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


# ------ classes -------
class GradCAM():
    def __init__(self, model, pred_label_index=None, conv_last_layer=False, last_layer_name=None):
        # -- initialization --
        self.model = model
        self.pred_label_index = pred_label_index
        if last_layer_name is None:
            if conv_last_layer:
                try:
                    last_conv_layer = next(
                        x for x in model.layers[::-1] if isinstance(x, Conv2D))
                    last_layer_name = last_conv_layer.name
                except StopIteration as e:
                    print('No Conv2D layer found in the input model.')
                    raise
            else:
                last_layer_name = self._find_target_layer()
        else:
            layer_names = []
            for l in model.layers:
                layer_names.append(l.name)

            if last_layer_name not in layer_names:
                raise ValueError('Custom last_layer_name not found in model.')

        self.last_layer_name = last_layer_name

    def _find_target_layer(self):
        """find the target layer (final layer with 4D output: n, dim1, dim2, channel)"""
        for l_4d in reversed(self.model.layers):
            if len(l_4d.output_shape) == 4:
                return l_4d.name

        raise ValueError(
            'Input model has no layer with output shape=4: None, dim1, dim2, channel.')

    def compute_gradcam_heatmap(self, img_array):
        heatmap = makeGradcamHeatmapV2(
            img_array=img_array, model=self.model, target_layer_name=self.last_layer_name,
            pred_label_index=self.pred_label_index)
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


class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    """
    Cosine decay with warmup learning rate scheduler\n

    Stole from here:
        https://github.com/Tony607/Keras_Bag_of_Tricks/blob/master/warmup_cosine_decay_scheduler.py\n
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """
        Purpose:\n
            Constructor for cosine decay with warmup learning rate scheduler.\n
        Arguments:\n
            learning_rate_base: float. Base learning rate.\n
            total_steps: int. Total number of training steps.\n
        Keyword Arguments:\n
            global_step_init: int. Initial global step, e.g. from previous checkpoint.\n
            warmup_learning_rate: float. Initial learning rate for warm up. (default: 0.0)\n
            warmup_steps: int. Number of warmup steps. (default: 0)\n
            hold_base_rate_steps: int. Optional number of steps to hold base learning rate
                                        before decaying. (default: 0)\n
            verbose: int. 0: quiet, 1: update messages. (default: 0)\n
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosinDecayWithWarmup(global_step=self.global_step,
                                  learning_rate_base=self.learning_rate_base,
                                  total_steps=self.total_steps,
                                  warmup_learning_rate=self.warmup_learning_rate,
                                  warmup_steps=self.warmup_steps,
                                  hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


class BatchMatrixLoader(object):
    """
    # Purpose\n
        Data loader for batch (out of memory) loading of matrices.

    # Initialization arguments\n
        filepath: str. Input file root file path.\n
        new_shape: tuple of int, or None. Optional new shape for the input data. When None, the first two dimensions are not changed.\n
        target_file_ext: str or None. Optional extension of the files to scan. When None, the data loader scans all files.\n
        manual_labels: pd.DataFrame or None. Optional file label data frame. When None, the loader's _parse_file() method automatically
            parses subfolder's name as file labels. Cannot be None when model_type='regression'.\n
        manual_labels_fileNameVar: string or None. Required for manual_labels, variable name in annotFile for file names.\n
        manual_labels_labelVar: string or None. Required for manual_labels, variable nam ein annotFile for lables.\n
        label_sep: str or None.  Optional str to separate label strings. When none, the loader uses the entire string as file labels.
        model_type: str. Model (label) type. Options are "classification", "regression" and "semisupervised".\n
        multilabel_classification: bool. If the classification is a "multilabel" type. Only effective when model_type='classification'.\n
        x_scaling: str. If and how to scale x values. Options are "none", "max" and "minmax".\n
        x_min_max_range: two num list. Only effective when x_scaling='minmax', the range for the x min max scaling.\n
        lower_triangular_paddin: None or int. The value to pad the lower triangular of the input systematic input matrix, including the diagonal.\n
        resampole_method: str. Effective when cv_only is not True. Train/test split method. Options are "random" and "stratified".\n
        training_percentage: num. Training data set percentage.\n
        verbose: bool. verbose.\n
        randome_state: int. randome state.\n

    # Details\n
        - This data loader is designed for matrices (similar to AxB resolution pictures).\n
        - For semisupervised model, the "label" would be the input data itself. This is typically used for autoencoder-decoder.\n
        - It is possible to stack matrix with A,B,N, and use new_shape argument to reshape the data into A,B,N shape.\n
        - For filepath, one can set up each subfolder as data labels. In such case, the _parse_file() method will automatically
            parse the subfolder name as labales for the files inside.\n
        - When using manual label data frame, make sure to only have one variable for labels, EVEN IF for multilabel modelling.
            In the case of multilabel modelling, the label string should be multiple labels separated by a separator string, which
            is set by the label_sep argument.\n
        - When multilabel, make sure to set up label_sep argument.\n
        - For multilabels, a mixture of continuous and discrete labels are not supported.\n
        - For x_min_max_range, a two tuple is required. Order: min, max. \n
        - No padding is applied when lower_triangular_padding=None.\n
        - It is noted that for regression, multilabel modelling is automatically supported via multiple labels in the manual label data frame.
            Therefore, for regression, manual_labels argument cannot be None.\n
        - When resample_method='random', the loader randomly draws samples according to the split percentage from the full data.
            When resample_method='stratified', the loader randomly draws samples according to the split percentage within each label.
            Currently, the "balanced" method, i.e. drawing equal amount of samples from each label, has not been implemented.\n
        - For manual_labels, the CSV file needs to have at least two columns: one for file names (no path, but with extension), one for labels.
            For regression modelling, the label column is the outcome.\n
    """

    def __init__(self, filepath,
                 new_shape=None,
                 target_file_ext=None,
                 model_type='classification', multilabel_classification=False, label_sep=None,
                 manual_labels=None, manual_labels_fileNameVar=None, manual_labels_labelVar=None,
                 x_scaling="none", x_min_max_range=[0, 1], lower_triangular_padding=None,
                 resmaple_method="random",
                 training_percentage=0.8,
                 verbose=True, random_state=1):
        """Initialization"""
        # - argument check -
        # for multilabel modelling label separation
        if model_type == 'classification':
            if multilabel_classification:
                if label_sep is None:
                    raise ValueError(
                        'set label_sep for multilabel classification.')
                else:
                    self.label_sep = label_sep
            else:
                if label_sep is not None:
                    warn('label_sep ignored when multilabel_class=False')
                    self.label_sep = None
                else:
                    self.label_sep = label_sep

        # - model information -
        self.model_type = model_type
        self.multilabel_class = multilabel_classification
        self.filepath = filepath
        self.target_ext = target_file_ext
        self.manual_labels = manual_labels
        self.manual_labels_fileNameVar = manual_labels_fileNameVar
        self.manual_labels_labelVar = manual_labels_labelVar
        # self.pd_labels_var_name = pd_labels_var_name  # deprecated argument
        # self.label_sep = label_sep
        self.new_shape = new_shape

        if model_type == 'semisupervised':
            self.semi_supervised = True
        else:
            self.semi_supervised = False

        # - processing -
        self.x_scaling = x_scaling
        self.x_min_max_range = x_min_max_range
        if lower_triangular_padding is not None and not isinstance(lower_triangular_padding, int):
            raise ValueError(
                'lower_triangular_padding needs to be an int if not None.')
        self.lower_triangular_padding = lower_triangular_padding
        # - resampling -
        self.resample_method = resmaple_method
        self.train_percentage = training_percentage
        self.test_percentage = 1 - training_percentage

        # - random state and other settings -
        self.rand = random_state
        self.verbose = verbose

        # - load paths -
        self.filepath_list, self.labels_list, self.lables_count, self.labels_map, self.labels_map_rev, self.encoded_labels = self._get_file_annot()

    def _parse_file(self):
        """
        - parse file path to get file path annotation and label information\n
        - set up manual label information\n
        """
        if self.manual_labels is None:
            if self.model_type == 'classification':
                # file_annot, labels = adjmatAnnotLoader(
                #     self.filepath, targetExt=self.target_ext)
                file_annot, labels = adjmatAnnotLoaderV2(
                    self.filepath, targetExt=self.target_ext)
            elif self.model_type == 'regression':
                raise ValueError(
                    'Set manual_labels when model_type=\"regression\".')
            elif self.model_type == 'semisupervised':
                file_annot, _ = adjmatAnnotLoaderV2(
                    self.filepath, targetExt=self.target_ext)
                labels = None
            else:
                raise NotImplemented('Unknown model type.')
        else:
            if self.model_type == 'semisupervised':
                raise ValueError(
                    'Manual labels not supported for \'semisupervised\' model type.')
            elif self.model_type in ('classification', 'regression'):
                try:
                    file_annot, labels = adjmatAnnotLoaderV2(
                        self.filepath, targetExt=self.target_ext, autoLabel=False,
                        annotFile=self.manual_labels,
                        fileNameVar=self.manual_labels_fileNameVar, labelVar=self.manual_labels_labelVar)
                except VariableNotFoundError as e:
                    print(
                        'Filename variable or label variable names not found in manual_labels file.')
                    raise
                except FileNotFoundError as e:
                    raise
                except FileError as e:
                    raise
            else:
                raise NotImplemented('Unknown model type.')

        # if self.model_type == 'classification':
        #     if self.manual_labels is None:
        #         # file_annot, labels = adjmatAnnotLoader(
        #         #     self.filepath, targetExt=self.target_ext)
        #         file_annot, labels = adjmatAnnotLoaderV2(
        #             self.filepath, targetExt=self.target_ext)
        #     else:
        #         try:
        #             file_annot, labels = adjmatAnnotLoaderV2(
        #                 self.filepath, targetExt=self.target_ext, autoLabel=False,
        #                 annotFile=self.manual_labels,
        #                 fileNameVar=self.manual_labels_fileNameVar, labelVar=self.manual_labels_labelVar)
        #         except VariableNotFoundError as e:
        #             print(e)
        #         except FileNotFoundError as e:
        #             print(e)

        # elif self.model_type == 'regression':
        #     if self.manual_labels is None:
        #         raise ValueError(
        #             'Set manual_labels when model_type=\"regression\".')
        #     else:
        #         try:
        #             file_annot, labels = adjmatAnnotLoaderV2(
        #                 self.filepath, targetExt=self.target_ext, autoLabel=False,
        #                 annotFile=self.manual_labels,
        #                 fileNameVar=self.manual_labels_fileNameVar, labelVar=self.manual_labels_labelVar)
        #         except VariableNotFoundError as e:
        #             print(e)
        #         except FileNotFoundError as e:
        #             print(e)

        # else:  # semisupervised
        #     file_annot, _ = adjmatAnnotLoader(
        #         self.filepath, targetExt=self.target_ext)
        #     labels = None

        return file_annot, labels

    def _get_file_annot(self, **kwargs):
        file_annot, labels = self._parse_file(**kwargs)

        if self.model_type == 'classification':
            labels_list, lables_count, labels_map, labels_map_rev = labelMapping(
                labels=labels, sep=self.label_sep)
            # if self.multilabel_class:
            #     if self.label_sep is None:
            #         raise ValueError(
            #             'set label_sep for multilabel classification.')

            #     labels_list, lables_count, labels_map, labels_map_rev = labelMapping(
            #         labels=labels, sep=self.label_sep)
            # else:
            #     if self.label_sep is not None:
            #         warn('label_sep ignored when multilabel_class=False')

            #     labels_list, lables_count, labels_map, labels_map_rev = labelMapping(
            #         labels=labels, sep=None)
            encoded_labels = labelOneHot(labels_list, labels_map)
        else:
            labels_list, lables_count, labels_map, labels_map_rev = None, None, None, None
            encoded_labels = labels

        try:
            filepath_list = file_annot['path'].to_list()
        except KeyError as e:
            print('Failed to load files. Hint: check target extension or directory.')
            raise

        return filepath_list, labels_list, lables_count, labels_map, labels_map_rev, encoded_labels

    def _x_data_process(self, x_array):
        """NOTE: reshaping to (_, _, 1) is mandatory"""
        # - variables -
        if isinstance(x_array, np.ndarray):  # this check can be done outside of the class
            X = x_array
        else:
            raise TypeError('data processing function should be a np.ndarray.')

        if self.x_scaling == 'max':
            X = X/X.max()
        elif self.x_scaling == 'minmax':
            Min = self.x_min_max_range[0]
            Max = self.x_min_max_range[1]
            X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            X = X_std * (Max - Min) + Min

        if self.lower_triangular_padding is not None:
            X[np.arange(X.shape[0])[:, None] > np.arange(
                X.shape[1])] = self.lower_triangular_padding
            np.fill_diagonal(X, self.lower_triangular_padding)

        if self.new_shape is not None:  # reshape
            X = np.reshape(X, self.new_shape)
        else:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X

    def _map_func(self, filepath: tf.Tensor, label: tf.Tensor, processing=False):
        # - read file and assign label -
        fname = filepath.numpy().decode('utf-8')
        f = np.loadtxt(fname).astype('float32')
        lb = label

        # - processing if needed -
        if processing:
            f = self._x_data_process(f)

        f = tf.convert_to_tensor(f, dtype=tf.float32)
        return f, lb

    def _map_func_semisupervised(self, filepath: tf.Tensor, processing=False):
        # - read file and assign label -
        fname = filepath.numpy().decode('utf-8')
        f = np.loadtxt(fname).astype('float32')

        # - processing if needed -
        if processing:
            f = self._x_data_process(f)

        f = tf.convert_to_tensor(f, dtype=tf.float32)
        lb = f
        return f, lb

    def _fixup_shape(self, f: tf.Tensor, lb: tf.Tensor):
        """requires further testing, this is for classification"""
        f.set_shape([None, None, f.shape[-1]])
        lb.set_shape(lb.shape)  # number of class
        return f, lb

    def _fixup_shape_semisupervised(self, f: tf.Tensor, lb: tf.Tensor):
        """requires further testing, only for semisupervised for testing"""
        f.set_shape([None, None, f.shape[-1]])
        lb.set_shape([None, None, lb.shape[-1]])
        return f, lb

    def _data_resample(self, total_data, n_total_sample, encoded_labels):
        """
        NOTE: regression cannot use stratified splitting\n
        NOTE: "stratified" (keep class ratios) is NOT the same as "balanced" (make class ratio=1)\n
        NOTE: "balanced" mode will be implemented at a later time\n
        NOTE: depending on how "balanced" is implemented, the if/else block could be simplified\n
        """
        # _, encoded_labels, _, _ = self._get_file_annot()
        X_indices = np.arange(n_total_sample)

        if self.model_type != 'classification' and self.resample_method == 'stratified':
            raise ValueError(
                'resample_method=\'stratified\' can only be set when model_type=\'classification\'.')

        if self.semi_supervised:  # only random is supported
            X_train_indices, X_test_indices = train_test_split(
                X_indices, test_size=self.test_percentage, stratify=None, random_state=self.rand)
        else:
            if self.resample_method == 'random':
                X_train_indices, X_test_indices, _, _ = train_test_split(
                    X_indices, encoded_labels, test_size=self.test_percentage, stratify=None, random_state=self.rand)
            elif self.resample_method == 'stratified':
                X_train_indices, X_test_indices, _, _ = train_test_split(
                    X_indices, encoded_labels, test_size=self.test_percentage, stratify=encoded_labels, random_state=self.rand)
            else:
                raise NotImplementedError(
                    '\"balanced\" resmapling method has not been implemented.')

        train_ds, train_n = getSelectedDataset(total_data, X_train_indices)
        test_ds, test_n = getSelectedDataset(total_data, X_test_indices)

        return train_ds, train_n, test_ds, test_n

    def generate_batched_data(self, batch_size=4, cv_only=False, shuffle_for_cv_only=True):
        """
        # Purpose\n
            To generate working data in batches. The method also creates a series of attributes that store 
                information like batch size, number of batches etc (see details)\n

        # Arguments\n
            batch_size: int. Batch size for the tf.dataset batches.\n
            cv_only: bool. When True, there is no train/test split.\n
            shuffle_for_cv_only: bool. Effective when cv_only=True, if to shuffle the order of samples for the output data.\n

        # Details\n
            - When cv_only=True, the loader returns only one tf.dataset object, without train/test split.
                In such case, further cross validation resampling can be done using followup resampling functions.
                However, it is not to say train/test split data cannot be applied with further CV operations.\n
            - As per tf.dataset behaviour, self.train_set_map and self.test_set_map do not contain data content. 
                Instead, these objects contain data map information, which can be used by tf.dataset.batch() tf.dataset.prefetch()
                methods to load the actual data content.\n
        """
        self.batch_size = batch_size
        self.cv_only = cv_only
        self.shuffle_for_cv_only = shuffle_for_cv_only

        # # - load paths -
        # filepath_list, labels_list, lables_count, labels_map_rev, encoded_labels = self._get_file_annot()

        if self.semi_supervised:
            total_ds = tf.data.Dataset.from_tensor_slices(self.filepath_list)
        else:
            total_ds = tf.data.Dataset.from_tensor_slices(
                (self.filepath_list, self.encoded_labels))

        # below: tf.dataset.cardinality().numpy() always displays the number of batches.
        # the reason this can be used for total sample size is because
        # tf.data.Dataset.from_tensor_slices() reads the file list as one file per batch
        self.n_total_sample = total_ds.cardinality().numpy()

        # return total_ds, self.n_total_sample  # test point

        # - resample data -
        self.train_batch_n = 0
        if self.cv_only:
            self.train_n = self.n_total_sample

            if self.semi_supervised:
                self.train_set_map = total_ds.map(lambda x: tf.py_function(self._map_func_semisupervised, [x, True], [tf.float32, tf.float32]),
                                                  num_parallel_calls=tf.data.AUTOTUNE)
                self.train_set_map = self.train_set_map.map(
                    self._fixup_shape_semisupervised, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                self.train_set_map = total_ds.map(lambda x: tf.py_function(self._map_func, [x, True], [tf.float32, tf.uint8]),
                                                  num_parallel_calls=tf.data.AUTOTUNE)
                self.train_set_map = self.train_set_map.map(
                    self._fixup_shape, num_parallel_calls=tf.data.AUTOTUNE)

            if self.shuffle_for_cv_only:  # check this
                self.train_set_map = self.train_set_map.shuffle(
                    random.randint(2, self.n_total_sample), seed=self.rand)
            self.test_set_map, self.test_n, self.test_batch_n = None, None, None
        else:
            train_ds, self.train_n, test_ds, self.test_n = self._data_resample(
                total_ds, self.n_total_sample, self.encoded_labels)

            if self.semi_supervised:
                self.train_set_map = train_ds.map(lambda x: tf.py_function(self._map_func_semisupervised, [x, True], [tf.float32, tf.float32]),
                                                  num_parallel_calls=tf.data.AUTOTUNE)
                self.train_set_map = self.train_set_map.map(
                    self._fixup_shape_semisupervised, num_parallel_calls=tf.data.AUTOTUNE)

                self.test_set_map = test_ds.map(lambda x: tf.py_function(self._map_func_semisupervised, [x, True], [tf.float32, tf.float32]),
                                                num_parallel_calls=tf.data.AUTOTUNE)
                self.test_set_map = self.test_set_map.map(
                    self._fixup_shape_semisupervised, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                self.train_set_map = train_ds.map(lambda x, y: tf.py_function(self._map_func, [x, y, True], [tf.float32, tf.uint8]),
                                                  num_parallel_calls=tf.data.AUTOTUNE)
                self.train_set_map = self.train_set_map.map(
                    self._fixup_shape, num_parallel_calls=tf.data.AUTOTUNE)

                self.test_set_map = test_ds.map(lambda x, y: tf.py_function(self._map_func, [x, y, True], [tf.float32, tf.uint8]),
                                                num_parallel_calls=tf.data.AUTOTUNE)
                self.test_set_map = self.test_set_map.map(
                    self._fixup_shape, num_parallel_calls=tf.data.AUTOTUNE)

            self.test_batch_n = 0

        # # - export attributes -
        # self.filepath_list = filepath_list
        # self.labels_list = labels_list
        # self.lables_count = lables_count
        # self.labels_map_rev = labels_map_rev
        # self.encoded_labels = encoded_labels

        # - set up batch and prefeching -
        # NOTE: the train_set and test_set are tensorflow.python.data.ops.dataset_ops.PrefetchDataset type
        train_batched = self.train_set_map.batch(
            self.batch_size).cache().prefetch(tf.data.AUTOTUNE)
        for _ in train_batched:
            self.train_batch_n += 1

        if self.test_set_map is not None:
            test_batched = self.test_set_map.batch(
                self.batch_size).cache().prefetch(tf.data.AUTOTUNE)
            for _ in test_batched:
                self.test_batch_n += 1
        else:
            test_batched = None

        # - retain real data shapes -
        for a, b in train_batched.take(1):  # only take one samples
            self.x_shape = a.numpy().shape[1:]  # [1:]: [0] is sample number
            self.y_shape = b.numpy().shape[1:]

        return train_batched, test_batched


class SingleCsvMemLoader(object):
    """
    # Purpose\n
        In memory data loader for single file CSV.\n
    # Arguments\n
        file: str. complete input file path.\n
        label_var: list of strings. Variable name for label(s). 
        annotation_vars: list of strings. Column names for the annotation variables in the input dataframe, EXCLUDING label variable.
        sample_id_var: str. variable used to identify samples.\n
        model_type: str. model type, classification or regression.\n
        n_classes: int. number of classes when model_type='classification'.\n
        training_percentage: float, betwen 0 and 1. percentage for training data.\n
        random_state: int. random state.\n
        verbose: bool. verbose.\n
    # Methods\n
        __init__: initialization.\n
        _label_onehot_encode: one hot encoding for labels.\n
        _x_minmax: min-max normalization for x data.\n        
    # Public class attributes\n
        Below are attributes read from arguments
            self.model_type
            self.n_classes
            self.file
            self.label_var
            self.annotation_vars
            self.cv_only
            self.holdout_samples
            self.training_percentage
            self.rand: int. random state
        self.labels_working: np.ndarray. Working labels. For classification, working labels are one hot encoded.
        self.filename: str. input file name without extension
        self.raw: pandas dataframe. input data
        self.raw_working: pandas dataframe. working input data
        self.complete_annot_vars: list of strings. column names for the annotation variables in the input dataframe, INCLUDING label variable
        self.n_features: int. number of features
        self.le: sklearn LabelEncoder for classification study
        self.label_mapping: dict. Class label mapping codes, when model_type='classification'.\n
    # Class property\n
        modelling_data: dict. data for model training. data is split if necessary.
            No data splitting for the "CV only" mode.
            returns a dict object with 'training' and 'test' items.\n
    # Details\n
        - label_var: Multilabels are supported.
            For classification, multilabels are supported via separater strings. Example: a_b_c = a, b, c.
            For regression, multiple variable names are accepted as multilabels. 
            The loader has sanity checks for this, i.e. classification type can only have one variable name in the list. \n
        - For multilabels, a mixture of continuous and discrete labels are not supported.\n
    """

    def __init__(self, file,
                 label_var, annotation_vars, sample_id_var,
                 minmax=True,
                 model_type='classification',
                 label_string_sep=None,
                 cv_only=False, shuffle_for_cv_only=True,
                 holdout_samples=None,
                 training_percentage=0.8,
                 resample_method='random',
                 random_state=1, verbose=True):
        """initialization"""
        # - random state and other settings -
        self.rand = random_state
        self.verbose = verbose

        # - model and data info -
        self.model_type = model_type
        # convert to a list for trainingtestSpliterFinal() to use

        if model_type == 'classification':
            if len(label_var) > 1:
                raise ValueError(
                    'label_var can only be len of 1 when model_type=\'classification\'')
            else:
                self.label_var = label_var[0]  # "delist"
                # self.y_var = [self.label_var]  # might not need this anymore
                self.complete_annot_vars = annotation_vars + [self.label_var]
        else:
            self.label_var = label_var
            self.complete_annot_vars = annotation_vars + label_var

        self.label_sep = label_string_sep
        self.annotation_vars = annotation_vars
        self._n_annot_col = len(self.complete_annot_vars)

        # - args.file is a list. so use [0] to grab the string -
        self.file = file
        self._basename, self._file_ext = os.path.splitext(file)

        # - resampling settings -
        self.cv_only = cv_only
        self.shuffle_for_cv_only = shuffle_for_cv_only
        self.resample_method = resample_method
        self.sample_id_var = sample_id_var
        self.holdout_samples = holdout_samples
        self.training_percentage = training_percentage
        self.test_percentage = 1 - training_percentage
        self.minmax = minmax

        # - parse file -
        self.raw = pd.read_csv(self.file, engine='python')
        if self.cv_only and self.shuffle_for_cv_only:
            self.raw_working = shuffle(self.raw.copy(), random_state=self.rand)
        else:
            self.raw_working = self.raw.copy()  # value might be changed

        self.n_features = int(
            (self.raw_working.shape[1] - self._n_annot_col))  # pd.shape[1]: ncol
        self.total_n = self.raw_working.shape[0]
        if model_type == 'classification':
            self.n_class = self.raw[label_var].nunique()
        else:
            self.n_class = None
        self.x = self.raw_working[self.raw_working.columns[
            ~self.raw_working.columns.isin(self.complete_annot_vars)]].to_numpy()
        self.labels = self.raw_working[self.label_var].to_numpy()

    def _label_onehot_encode(self, labels):
        """one hot encoding for labels. labels: should be a np.ndarray"""
        labels_list, labels_count, labels_map, labels_map_rev = labelMapping(
            labels, sep=self.label_sep)

        onehot_encoded = labelOneHot(labels_list, labels_map)

        return onehot_encoded, labels_count, labels_map_rev

    def _x_minmax(self, x_array):
        """NOTE: reshaping to (_, _, 1) is mandatory"""
        # - variables -
        if isinstance(x_array, np.ndarray):  # this check can be done outside of the classs
            X = x_array
        else:
            raise TypeError('data processing function should be a np.ndarray.')

        # - minmax -
        Min = 0
        Max = 1
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X = X_std * (Max - Min) + Min

        return X

    def generate_batched_data(self, batch_size=4):
        """
        # Purpose\n
            Generate batched data\n
        # Arguments\n
            batch_size: int.\n
            cv_only: bool. If to split data into training and holdout test sets.\n
            shuffle_for_cv_only: bool. Effective when cv_only=True, if to shuffle the order of samples for the output data.\n
        """
        # print("called setter") # for debugging
        if self.model_type == 'classification':  # one hot encoding
            self.labels_working, self.labels_count, self.labels_rev = self._label_onehot_encode(
                self.labels)
        else:
            self.labels_working, self.labels_count, self.labels_rev = self.labels, None, None

        if self.minmax:
            self.x_working = self._x_minmax(self.x)

        # - data resampling -
        self.train_batch_n = 0
        if self.cv_only:  # only training is stored
            # training set prep
            self._training_x = shuffle(self.x_working, random_state=self.rand)
            self._training_y = self.labels_working
            self.train_n = self.total_n

            # test set prep
            self._test_x, self._test_y = None, None
            self.test_n = None
            self.test_batch_n = None
        else:  # training and holdout test data split
            X_indices = np.arange(self.total_n)
            if self.resample_method == 'random':
                X_train_indices, X_test_indices, self._training_y, self._test_y = train_test_split(
                    X_indices, self.labels_working, test_size=self.test_percentage, stratify=None, random_state=self.rand)
            elif self.resample_method == 'stratified':
                X_train_indices, X_test_indices, self._training_y, self._test_y = train_test_split(
                    X_indices, self.labels_working, test_size=self.test_percentage, stratify=self.labels_working, random_state=self.rand)
            else:
                raise NotImplementedError(
                    '\"balanced\" resmapling method has not been implemented.')

            self._training_x, self._test_x = self.x_working[
                X_train_indices], self.x_working[X_test_indices]
            self.train_n, self.test_n = len(
                X_train_indices), len(X_test_indices)
            self.test_batch_n = 0

        # - set up final training and test set -
        self.train_ds = tf.data.Dataset.from_tensor_slices(
            (self._training_x, self._training_y))

        if self.cv_only:
            self.test_ds = None
        else:
            self.test_ds = tf.data.Dataset.from_tensor_slices(
                (self._test_x, self._test_y))

        # - set up batches -
        train_batched = self.train_ds.batch(
            batch_size).cache().prefetch(tf.data.AUTOTUNE)
        for _ in train_batched:
            self.train_batch_n += 1

        if self.test_ds is not None:
            test_batched = self.test_ds.batch(batch_size).cache().prefetch(
                tf.data.AUTOTUNE)
            for _ in test_batched:
                self.test_batch_n += 1
        else:
            test_batched = None

        return train_batched, test_batched
