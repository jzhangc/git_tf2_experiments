"""utilities for deep learning (excluding models)"""

# ------ modules ------
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.data_utils import adjmatAnnotLoader, adjmatAnnotLoaderV2, labelMapping, labelOneHot, getSelectedDataset
from utils.other_utils import VariableNotFoundError, FileError, warn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# ------ classes -------
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
        resampole_method: str. Effective when cv_only is not True. Train/test split method. Options are "random" and "stratified".
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
        - It is noted that for regression, multilabel modelling is automatically supported via multiple labels in the manual label data frame.
            Therefore, for regression, manual_labels argument cannot be None.\n
        - When resample_method='random', the loader randomly draws samples according to the split percentage from the full data.
            When resample_method='stratified', the loader randomly draws samples according to the split percentage within each label.
            Currently, the "balanced" method, i.e. drawing equal amount of samples from each label, has not been implemented.\n
        - For manual_labels, the CSV file needs to have at least two columns: one for file names (no path, but with extension), one for labels.
            For regression modelling, the
    """

    def __init__(self, filepath,
                 new_shape=None,
                 target_file_ext=None,
                 model_type='classification', multilabel_classification=False, label_sep=None,
                 manual_labels=None, manual_labels_fileNameVar=None, manual_labels_labelVar=None,
                 x_scaling="none", x_min_max_range=[0, 1],
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

        # - resampling -
        self.resample_method = resmaple_method
        self.train_percentage = training_percentage
        self.test_percentage = 1 - training_percentage

        # - random state and other settings -
        self.rand = random_state
        self.verbose = verbose

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
                    raise VariableNotFoundError(
                        'Filename variable or label variable names not found in manual_labels file.')
                except FileNotFoundError as e:
                    raise e
                except FileError as e:
                    raise e
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

        # - load paths -
        filepath_list, encoded_labels, self.lables_count, self.labels_map_rev = self._get_file_annot()

        if self.semi_supervised:
            total_ds = tf.data.Dataset.from_tensor_slices(filepath_list)
        else:
            total_ds = tf.data.Dataset.from_tensor_slices(
                (filepath_list, encoded_labels))

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
                total_ds, self.n_total_sample, encoded_labels)

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
