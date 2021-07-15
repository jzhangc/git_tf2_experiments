"""utilities for deep learning (excluding models)"""

# ------ modules ------
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.data_utils import adjmatAnnotLoader, labelMapping, labelOneHot, getSelectedDataset
from sklearn.model_selection import train_test_split


# ------ classes -------
class BatchDataLoader(object):
    """
    # Purpose\n
        Data loader for batch (out of memory) loading of matrices.

    # Initialization arguments\n
        filepath: str. Input file root file path.\n
        new_shape: tuple of int, or None. Optional new shape for the input data. When None, the first two dimensions are not changed.\n
        target_file_ext: str or None. Optional extension of the files to scan. When None, the data loader scans all files.\n
        manual_labels: pd.DataFrame or None. Optional file label data frame. When None, the loader's _parse_file() method automatically
            parses subfolder's name as file labels. Cannot be None when model_type='regression'.\n
        label_sep: str or None.  Optional str to separate label strings. When none, the loader uses the entire string as file labels.
        pd_labelse_bar_name: list of str or None. Set when manual_labels is not None, variable name for file labels.\n
        model_type: str. Model (label) type. Options are "classification" and "regression".\n
        multilabel_classification: bool. If the classifiation is a "multilabel" type. Only effective when model_type='classification'.\n
        x_scaling: str. If and how to scale x values. Options are "none", "max" and "minmax".\n
        x_min_max_range: two num list. Only effective when x_scaling='minmax', the range for the x min max scaling.\n
        resampole_method: str. Effective when cv_only is not True. Train/test split method. Options are "random" and "stratified".
        training_percentage: num. Training data set percentage.\n
        verbose: bool. verbose.\n 
        randome_state: int. randome state.\n

    # Details\n
        - This data loader is designed for matrices (similar to AxB resolution pictures).\n
        - It is possible to stack matrix with A,B,N, and use new_shape argument to reshape the data into A,B,N shape.\n
        - For filepath, one can set up each subfolder as data labels. In such case, the _parse_file() method will automatically
            parse the subfolder name as labales for the files inside.\n
        - When using manual label data frame, make sure to only have one variable for labels, EVEN IF for multilabel modelling.
            In the case of multilabel modelling, the label string should be multiple labels separated by a separator string, which
            is set by the label_sep argument.\n
        - When multilabel, make sure to set up label_sep argument.\n
        - For x_min_max_range, a two tuple is required. Order: min, max. \n
        - It is noted that for regression, multilabel modelling is automatically supported via multiple labels in the maual label data frame.
            Therefore, for regression, manual_labels argument cannot be None.\n
        - When resample_method='random', the loader randomly draws samples according to the split percentage from the full data.
            When resample_method='stratified', the loader randomly draws samples accoridng to the split percentage within each label.
            Currently, the "balanced" method, i.e. drawing equal amount of samples from each label, has not been implemented.\n
    """

    def __init__(self, filepath,
                 new_shape=None,
                 target_file_ext=None,
                 manual_labels=None, label_sep=None, pd_labels_var_name=None,
                 model_type='classification', multilabel_classification=False,
                 x_scaling="none", x_min_max_range=[0, 1],
                 resmaple_method="random",
                 training_percentage=0.8,
                 verbose=True, random_state=1):
        """Initialization"""
        # model information
        self.model_type = model_type
        self.multilabel_class = multilabel_classification
        self.filepath = filepath
        self.target_ext = target_file_ext
        self.manual_labels = manual_labels
        self.pd_labels_var_name = pd_labels_var_name
        self.label_sep = label_sep
        self.new_shape = new_shape

        # processing
        self.x_scaling = x_scaling
        self.x_min_max_range = x_min_max_range

        # resampling
        self.resample_method = resmaple_method
        self.train_percentage = training_percentage
        self.test_percentage = 1 - training_percentage

        # random state and other settings
        self.rand = random_state
        self.verbose = verbose

    def _parse_file(self):
        """
        - parse file path to get file path annotatin and, optionally, label information\n
        - set up manual label information\n
        """

        if self.model_type == 'classification':
            file_annot, labels = adjmatAnnotLoader(
                self.filepath, targetExt=self.target_ext)
        else:  # regeression
            if self.manual_labels is None:
                raise ValueError(
                    'Set manual_labels when model_type=\"regression\".')
            file_annot, _ = adjmatAnnotLoader(
                self.filepath, targetExt=self.target_ext, autoLabel=False)

        if self.manual_labels is not None:  # update labels to the manually set array
            if isinstance(self.manual_labels, pd.DataFrame):
                if self.pd_labels_var_name is None:
                    raise TypeError(
                        'Set pd_labels_var_name when manual_labels is a pd.Dataframe.')
                else:
                    try:
                        labels = self.manual_labels[self.pd_labels_var_name].to_numpy(
                        )
                    except Exception as e:
                        print(
                            'Manual label parsing failed. Hint: check if pd_labels_var_name is present in the maual label data frame.')
            elif isinstance(self.manual_labels, np.ndarray):
                labels = self.manual_labels
            else:
                raise TypeError(
                    'When not None, manual_labels needs to be pd.Dataframe or np.ndarray.')

            labels = self.manual_labels

        return file_annot, labels

    def _get_file_annot(self, **kwargs):
        file_annot, labels = self._parse_file(**kwargs)

        if self.model_type == 'classification':
            if self.multilabel_class:
                if self.label_sep is None:
                    raise ValueError(
                        'set label_sep for multilabel classification.')

                labels_list, lables_count, labels_map, labels_map_rev = labelMapping(
                    labels=labels, sep=self.label_sep)
            else:
                labels_list, lables_count, labels_map, labels_map_rev = labelMapping(
                    labels=labels, sep=None)
            encoded_labels = labelOneHot(labels_list, labels_map)
        else:
            encoded_labels = labels
            lables_count, labels_map_rev = None, None

        try:
            filepath_list = file_annot['path'].to_list()
        except KeyError as e:
            print('Failed to load files. Hint: check target extension or directory.')

        return filepath_list, encoded_labels, lables_count, labels_map_rev

    def _x_data_process(self, x_array):
        """NOTE: reshaping to (_, _, 1) is mandatory"""
        # - variables -
        if isinstance(x_array, np.ndarray):  # this check can be done outside of the classs
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

    def _data_resample(self, total_data, n_total_sample, encoded_labels):
        """
        NOTE: regression cannot use stratified splitting\n
        NOTE: "stratified" (keep class ratios) is NOT the same as "balanced" (make class ratio=1)\n
        NOTE: "balanced" mode will be implemented at a later time\n
        NOTE: depending on how "balanced" is implemented, the if/else block could be implified\n
        """
        # _, encoded_labels, _, _ = self._get_file_annot()
        X_indices = np.arange(n_total_sample)
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

    def generate_batched_data(self, batch_size=4, cv_only=False, shuffle=True):
        """
        # Purpose\n
            To generate working data in batches. The method also creates a series of attributes that store 
                information like batch size, number of batches etc (see details)\n

        # Arguments\n
            batch_size: int. Batch size for the tf.dataset batches.\n
            cv_only: bool. When True, there is no train/test split.\n
            shuffle: bool. Effective when cv_only=True, if to shuffle the order of samples for the output data.\n

        # Details\n
            - When cv_only=True, the loader returns only one tf.dataset object, without train/test split.
                In such case, further cross validation resampling can be done using followup resampling functions.
                However, it is not to say train/test split data cannot be applied with further CV operations.\n        
        """
        self.batch_size = batch_size
        self.cv_only = cv_only
        self.shuffle = shuffle

        # - load paths -
        filepath_list, encoded_labels, self.lables_count, self.labels_map_rev = self._get_file_annot()
        total_ds = tf.data.Dataset.from_tensor_slices(
            (filepath_list, encoded_labels))
        # below: tf.dataset.cardinality().numpy() always displays the number of batches.
        # the reason this can be used for total sample size is because
        # tf.data.Dataset.from_tensor_slices() reads the file list as one file per batch
        self.n_total_sample = total_ds.cardinality().numpy()

        # return total_ds, self.n_total_sample  # test point

        # - resample data -
        self.train_set_batch_n = 0
        if self.cv_only:
            self.train_set_map = total_ds.map(lambda x, y: tf.py_function(self._map_func, [x, y, True], [tf.float32, tf.uint8]),
                                              num_parallel_calls=tf.data.AUTOTUNE)
            self.train_n = self.n_total_sample
            if self.shuffle:  # check this
                self.train_set_map = self.train_set_map.shuffle(
                    random.randint(2, self.n_total_sample), seed=self.rand)
            self.test_set_map, self.test_n, self.test_set_batch_n = None, None, None
        else:
            train_ds, self.train_n, test_ds, self.test_n = self._data_resample(
                total_ds, self.n_total_sample, encoded_labels)
            self.train_set_map = train_ds.map(lambda x, y: tf.py_function(self._map_func, [x, y, True], [tf.float32, tf.uint8]),
                                              num_parallel_calls=tf.data.AUTOTUNE)
            self.test_set_map = test_ds.map(lambda x, y: tf.py_function(self._map_func, [x, y, True], [tf.float32, tf.uint8]),
                                            num_parallel_calls=tf.data.AUTOTUNE)
            self.test_set_batch_n = 0

        # - set up batch and prefeching -
        # NOTE: the train_set and test_set are tensorflow.python.data.ops.dataset_ops.PrefetchDataset type
        train_set_batched = self.train_set_map.batch(
            self.batch_size).cache().prefetch(tf.data.AUTOTUNE)
        for _ in train_set_batched:
            self.train_set_batch_n += 1

        if self.test_set is not None:
            test_set_batched = self.test_set_map.batch(
                self.batch_size).cache().prefetch(tf.data.AUTOTUNE)
            for _ in test_set_batched:
                self.test_set_batch_n += 1

        return train_set_batched, test_set_batched
