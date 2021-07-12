#!/usr/bin/env python3
"""
Current objectives:
[ ] Test argparse
    [ ] Add groupped arguments
[ ] Test output directory creation
[x] Test file reading
[x] Test file processing
    [x] normalization and scalling
    [x] converting to numpy arrays
[ ] Implement "balanced" data resampling
[x] Implement data resampling for cross-validation (maybe move this out of the dataloader)

NOTE
All the argparser inputs are loaded from method arguments, making the class more portable, i.e. not tied to
the application.
"""
# ------ import modules ------
import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.other_utils import error, warn, flatten, addBoolArg, outputDir, fileDir, colr
from utils.data_utils import adjmatAnnotLoader, labelMapping, labelOneHot, getSelectedDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ------ system classes ------
class AppArgParser(argparse.ArgumentParser):
    """
    This is a sub class to argparse.ArgumentParser.
    Purpose
        The help page will display when (1) no argumment was provided, or (2) there is an error
    """

    def error(self, message, *lines):
        string = "\n{}ERROR: " + message + "{}\n" + \
            "\n".join(lines) + ("{}\n" if lines else "{}")
        print(string.format(colr.RED_B, colr.RED, colr.ENDC))
        self.print_help()
        sys.exit(2)


# ------ GLOBAL variables -------
__version__ = '0.1.0'
AUTHOR = 'Jing Zhang, PhD'
DESCRIPITON = """
{}--------------------------------- Description -------------------------------------------
Data loader for batch adjacency matrix CSV file data table for deep learning.
The loaded CSV files are stored in a 3D numpy array, with size: .
This loader also handels the following:
    1. Data resampling, e.g. traning/test split, cross validation
    2. Data normalization
-----------------------------------------------------------------------------------------{}
""".format(colr.YELLOW, colr.ENDC)

# ------ augment definition ------
# - set up parser and argument groups -
parser = AppArgParser(description=DESCRIPITON,
                      epilog='Written by: {}. Current version: {}\n\r'.format(
                          AUTHOR, __version__),
                      formatter_class=argparse.RawTextHelpFormatter)
parser._optionals.title = "{}Help options{}".format(colr.CYAN_B, colr.ENDC)

arg_g1 = parser.add_argument_group(
    '{}Input and output{}'.format(colr.CYAN_B, colr.ENDC))
arg_g2 = parser.add_argument_group(
    '{}Resampling and normalization{}'.format(colr.CYAN_B, colr.ENDC))
arg_g3 = parser.add_argument_group(
    '{}Modelling{}'.format(colr.CYAN_B, colr.ENDC))
arg_g4 = parser.add_argument_group('{}Other{}'.format(colr.CYAN_B, colr.ENDC))

add_g1_arg = arg_g1.add_argument  # input/output
add_g2_arg = arg_g2.add_argument  # processing and resampling
add_g3_arg = arg_g3.add_argument  # modelling
add_g4_arg = arg_g4.add_argument  # others

# - add arugments to the argument groups -
# g1: inpout and ouput
add_g1_arg('path', nargs=1, type=fileDir,
           help='Directory contains all input files. (Default: %(default)s)')
add_g1_arg('-il', '--manual_label_file', type=str,
           help='Optional one manual labels CSV file. (Default: %(default)s)')
add_g1_arg('-te', '--target_file_ext', type=str, default=None,
           help='str. When manual labels are provided and imported as a pandas dataframe, the label variable name for this pandas dataframe. (Default: %(default)s)')
add_g1_arg('-lv', '--pd_labels_var_name', type=str, default=None,
           help='str. When manual labels are provided and imported as a pandas dataframe, the label variable name for this pandas dataframe. (Default: %(default)s)')
add_g1_arg('-ls', '--label_string_sep', type=str, default=None,
           help='str. Separator to separate label string, to create multilabel labels. (Default: %(default)s)')
addBoolArg(parser=arg_g1, name='y_scale', input_type='flag', default=False,
           help='str. If to min-max scale label for regression study. (Default: %(default)s)')
add_g1_arg('-o', '--output_dir', type=fileDir,
           default='.',
           help='str. Output directory. NOTE: not an absolute path, only relative to working directory -w/--working_dir.')

# g2: processing and resampling
add_g2_arg('-ns', '--new_shape', type=str, default=None,
           help='str. Optional new shape tuple. (Default: %(default)s)')

# g4: others
add_g2_arg('-se', '--random_state', type=int,
           default=1, help='int. Random state. (Default: %(default)s)')
addBoolArg(parser=arg_g4, name='verbose', input_type='flag', default=False,
           help='Verbose or not. (Default: %(default)s)')

# - load arguments -
args = parser.parse_args()
print(args)

# ------- check arguments -------
# below: we use custom script to check this file (not csvPath function) for custom error messages.
if args.manual_label_file is not None:
    if os.path.isfile(args.manual_label_file):
        # return full_path
        _, file_ext = os.path.splitext(args.manual_label_file)
        if file_ext != '.csv':
            error('Manual label file needs to be .csv type.')
        else:
            manual_label_file = os.path.normpath(os.path.abspath(
                os.path.expanduser(args.manual_label_file)))
    else:
        error('Invalid manual label file or file not found.')
else:
    manual_label_file = args.manual_label_file


# ------ loacl classes ------
class BatchDataLoader(object):
    """
    # Purpose\n
        Data loader for batch (out of memory) loading of matrices.
    # Initialization arguments\n
        filepath: str. Input file root file path.
        new_shape: tuple of int, or None. Optional new shape for the input data. When None, the first two dimensions are not changed.
        target_file_ext: str or None. Optional extension of the files to scan. When None, the data loader scans all files.
        manual_labels: pd.DataFrame or None. Optional file label data frame. When None, the loader's _parse_file() method automatically
            parses subfolder's name as file labels. Cannot be None when model_type='regression'.
        label_sep: str or None.  Optional str to separate label strings. When none, the loader uses the entire string as file labels.
        pd_labelse_bar_name: list of str or None. Set when manual_labels is not None, variable name for file labels.
        model_type: str. Model (label) type. Options are "classification" and "regression".
        multilabel_classification: bool. If the classifiation is a "multilabel" type. Only effective when model_type='classification'.
        x_scaling: str. If and how to scale x values. Options are "none", "max" and "minmax".
        x_min_max_range. two num tuple. Only effective when x_scaling='minmax', the range for the x min max scaling.
        cv_only: bool. When True, there is no train/test split.
        shuffle: bool. Effective when cv_only=True, if to shuffle the order of samples for the output data.
        resampole_method: str. Effective when cv_only is not True. Train/test split method. Options are "random" and "stratified".
        training_percentage: num. Training data set percentage.
        batch_size: int. Batch size for the tf.dataset batches.

    # Details\n
        1. This data loader is designed for matrices (similar to _, _ resolution pictures).
        2. It is possible to stack matrix with _,_,_N, and use new_shape argument to reshape the data into _,_,N shape.
        3. For filepath, one can set up each subfolder as data labels. In such case, the _parse_file() method will automatically
            parse the subfolder name as labales for the files inside.
        4. When using manual label data frame, make sure to only have one variable for labels, EVEN IF for multilabel modelling.
            In the case of multilabel modelling, the label string should be multiple labels separated by a separator string, which
            is set by the label_sep argument.
        5. When multilabel, make sure to set up label_sep argument.
        6. It is noted that for regression, multilabel modelling is automatically supported via multiple labels in the maual label data frame.
            Therefore, for regression, manual_labels argument cannot be None.
        7. When cv_only=True, the loader returns only one tf.dataset object, without train/test split.
            In such case, further cross validation resampling can be done using followup resampling functions.
            However, it is not to say train/test split data cannot be applied with further CV operations.
        8. When resample_method='random', the loader randomly draws samples according to the split percentage from the full data.
            When resample_method='stratified', the loader randomly draws samples accoridng to the split percentage within each label.
            Currently, the "balanced" method, i.e. drawing equal amount of samples from each label, has not been implemented.
    """

    def __init__(self, filepath,
                 new_shape=None,
                 target_file_ext=None,
                 manual_labels=None, label_sep=None, pd_labels_var_name=None,
                 model_type='classification', multilabel_classification=False,
                 x_scaling="none", x_min_max_range=(0, 1),
                 cv_only=False, shuffle=True,
                 resmaple_method="random",
                 training_percentage=0.8,
                 batch_size=None,
                 verbose=True, random_state=1):
        """
        # Arguments\n
            resample_method: str. options: "random", "stratified" and "balanced".
            x_scaling: str. Options are "none", "max", or "minmax". Default is None (i.e. not scaling data).
            x_min_max_range: two tuple. set when x_scaling="minmax", (min, max) range.

        # Details\n
            1. resample_method is automatically set to "random" when model_type='regression'.
        """
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
        self.batch_size = batch_size

        # resampling
        self.resample_method = resmaple_method
        self.train_percentage = training_percentage
        self.test_percentage = 1 - training_percentage
        self.cv_only = cv_only
        self.shuffle = shuffle

        # random state and other settings
        self.rand = random_state
        self.verbose = verbose

    def _parse_file(self):
        """
        1. parse file path to get file path annotatin and, optionally, label information
        2. set up manual label information
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
                        error('Manual label parsing failed.',
                              'check if pd_labels_var_name is present in the maual label data frame.')
            elif isinstance(self.manual_labels, np.ndarray):
                labels = self.manual_labels
            else:
                raise TypeError(
                    'When not None, manual_labels needs to be pd.Dataframe or np.ndarray.')

            labels = self.manual_labels

        return file_annot, labels

    def _get_file_annot(self):
        file_annot, labels = self._parse_file()

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

        filepath_list = file_annot['path'].to_list()

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

    def _data_resample(self, total_data, n_total_sample):
        """
        NOTE: regression cannot use stratified splitting
        NOTE: "stratified" (keep class ratios) is NOT the same as "balanced" (make class ratio=1)
        NOTE: "balanced" mode will be implemented at a later time
        NOTE: depending on how "balanced" is implemented, the if/else block could be implified
        """
        _, encoded_labels, _, _ = self._get_file_annot()
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

    def load_data(self, batch_size, shuffle=False):
        # - load paths -
        filepath_list, encoded_labels, self.lables_count, self.labels_map_rev = self._get_file_annot()
        total_ds = tf.data.Dataset.from_tensor_slices(
            (filepath_list, encoded_labels))
        self.n_total_sample = total_ds.cardinality().numpy()

        # - resample data and output-
        if self.cv_only:
            train_set = total_ds.map(lambda x, y: tf.py_function(self._map_func, [x, y, True], [tf.float32, tf.uint8]),
                                     num_parallel_calls=tf.data.AUTOTUNE)
            self.train_n = self.n_total_sample
            test_set = None
            self.test_n = None

            if self.shuffle:  # check this
                train_set = train_set.shuffle()
        else:
            train_ds, self.train_n, test_ds, self.test_n = self._data_resample(
                total_ds, self.n_total_sample)
            train_set = train_ds.map(lambda x, y: tf.py_function(self._map_func, [x, y, True], [tf.float32, tf.uint8]),
                                     num_parallel_calls=tf.data.AUTOTUNE)
            test_set = test_ds.map(lambda x, y: tf.py_function(self._map_func, [x, y, True], [tf.float32, tf.uint8]),
                                   num_parallel_calls=tf.data.AUTOTUNE)

        return train_set, test_set


# ------ process/__main__ statement ------
# if __name__ == '__main__':
#     mydata = DataLoader(file=args.file[0],
#                         label_var=args.label_var, annotation_vars=args.annotation_vars, sample_id_var=args.sample_id_var,
#                         holdout_samples=args.holdout_samples,
#                         model_type=args.model_type, cv_only=args.cv_only, man_split=args.man_split, training_percentage=args.training_percentage,
#                         random_state=args.random_state, verbose=args.verbose)
