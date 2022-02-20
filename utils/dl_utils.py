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
from typing import Union, Optional
from IPython.display import Image, display
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import Callback

from utils.data_utils import (adjmatAnnotLoader, adjmatAnnotLoaderV2,
                              getSelectedDataset, labelMapping, labelOneHot)
from utils.error_handling import FileError, VariableNotFoundError, warn


# ------ functions ------
class MultilabelConfusionMatrixHistory(Callback):
    """custom callback to recrod multilabel confusion matrix per epoch

    Usage:
        metrics_callback = MultilabelConfusionMatrixHistory()
        model.fit(..., callback=[metrics_callback])
        metrics_callback.return_matrices()

    Note: to use scikit-learn metrics functions
    Note: currently (tf2 keras version), custom callback can not access training and validation data from tf.dataset.Dataset
    """

    def on_train_begin(self, logs=None):
        self.confusion = []

    def on_epoch_end(self, epoch, logs=None):
        """calculate confusion matrix at each epoch"""
        # print(f'validation x type: {type(self.validation_data[0])}')
        # print(f'validation y type: {type(self.validation_data[0])}')
        val_x = self.validation_data[0]  # would not work
        val_y = self.validation_data[1]  # would not work
        pred = self.model.predict(val_x)

        confusion = multilabel_confusion_matrix(val_y, pred)
        self.confusion.append(confusion)

        return

    def return_confusion_matrices(self):
        return self.confusion


def cosinDecayWithWarmup(global_step: int,
                         learning_rate_base: float,
                         total_steps: int,
                         warmup_learning_rate: float = 0.0,
                         warmup_steps: int = 0,
                         hold_base_rate_steps: int = 0):
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


def getImgArray(img_path: str, size: int):
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


def makeGradcamHeatmap(img_array, model, target_layer_name: str, pred_label_index: Optional[int] = None):
    """
    # Purpose:\n
        heatmap gerneation for GradCAM, from keras.io.\n

    # Details:\n
        - The pred_label_index is unsorted. 
            One can get the info from from the "label_map_rev" attribtue from BatchDataLoader class.
            Example (dict keys are indices): 
            {0: 'all', 1: 'alpha', 2: 'beta', 3: 'fmri', 4: 'hig', 5: 'megs', 6: 'pc', 7: 'pt', 8: 'sc'}\n
    """
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            target_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_label_index is None:
            pred_label_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_label_index]

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

    return heatmap


def makeGradcamHeatmapV2(img_array, model, target_layer_name, pred_label_index: Optional[int] = None,
                         guided_grad: bool = False, eps: float = 1e-8):
    """
    # Purpose:\n
        V2 of heatmap gerneation for GradCAM, with guided grad functionality.\n

    # Details:\n
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
    heatmap = heatmap / tf.math.reduce_max(heatmap)  # scale to [0, 1]
    # numer = heatmap - np.min(heatmap)
    # denom = (heatmap.max() - heatmap.min()) + eps
    # heatmap = numer / denom
    # heatmap = (heatmap * 255).astype("uint8")  # convert back heatmap into 255 scale

    # -- grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions --
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmap = cv2.resize(heatmap.numpy(), (w, h))

    # -- return the resulting heatmap --
    return heatmap


def displayGradcam(image_array, heatmap, cam_path: Optional[str] = None,
                   alpha: float = 0.4):
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
    def __init__(self, model, label_index_dict: Optional[dict] = None,
                 conv_last_layer: bool = False,
                 target_layer_name: Optional[str] = None):
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
        # Purpose:\n
            Constructor for cosine decay with warmup learning rate scheduler.\n
        # Arguments:\n
            learning_rate_base: float. Base learning rate.\n
            total_steps: int. Total number of training steps.\n
        # Keyword Arguments:\n
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
