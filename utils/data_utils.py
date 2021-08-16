"""utilities for data input and processing"""

# ------ modules ------
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.other_utils import flatten
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ------ functions -------
def getSingleCsvDataset(csv_path, label_var, column_to_exclude=None,
                        batch_size=5, **kwargs):
    """
    # Purpose\n
        Write in a single CSV file into a tf.dataset object.\n

    # Argument\n
        csv_path: str. File path.\n 
        label_var: str. Variable name.\n
        column_to_exclude: None, or list of str. A list of the variable names to exclude.\n
        batch_size: int. Batch size.\n

    # Return\n
        Two items (in following order): tf.dataset, feature list.\n

    # Details\n
        - pd.read_csv is used to read in the header of the CSV file.\n
        - label_var only supports one label, i.e. only binary and multi-class are supported.\n
        - length of the output tf.dataset is number of batches, NOT samples.\n
    """
    # - write in only the header information -
    csv_header = pd.read_csv(csv_path, index_col=0,
                             nrows=0).columns.tolist()

    # - subset columns and establish feature list -
    if column_to_exclude is not None:
        column_to_include = [
            element for element in csv_header if element not in column_to_exclude]
        feature_list = column_to_include
    else:
        column_to_include = None
        feature_list = csv_header

    # - set up tf.dataset -
    ds = tf.data.experimental.make_csv_dataset(
        csv_path,
        batch_size=batch_size,
        label_name=label_var,
        na_value='?',
        num_epochs=1,
        ignore_errors=True,
        select_columns=column_to_include,
        **kwargs)

    return ds, feature_list


def scanFiles(basePath, validExts=None, contains=None):
    """
    # Purpose\n
        Scan subdirs and extract file paths.\n

    # Arguments\n
        basePath: str. Directory path to scan.\n
        validExts: str. (Optional) File extension to target.\n
        contains: str. String included in the file name to scan.\n

    # Return\n
        A multi-line string containing file paths.\n

    # Details\n
        - The function scans both root and sub directories.\n

        - This is a modified version of imutils.list_files,
            in which the function no longer verifies if the
            file is a image. Instead, it optionally only grabs
            files with the pre-set extension.\n 

        - When validExts=None, the function extracts all files.\n
    """
    if not os.path.isdir(basePath):
        raise FileNotFoundError(f'Directory not found: {basePath}')

    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind('.'):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                filePath = os.path.join(rootDir, filename)
                yield filePath  # yield is "return" without terminating the function


def sameFileCheck(dir, **kwargs):
    """check if dir or sub dirs contain duplicated filenames."""
    filepaths = list(scanFiles(dir, **kwargs))
    filenames = []

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        filenames.append(filename)

    dup = [k for k, v in Counter(filenames).items() if v > 1]

    return dup


def findFilePath(tgt_filename, dir):
    """find specific file in a dir and return full path"""
    for rootDir, dirNames, filenames in os.walk(dir):
        if tgt_filename in filenames:
            filePath = os.path.join(rootDir, tgt_filename)
            yield filePath


def adjmatAnnotLoader(dir, autoLabel=True, targetExt=None):
    """
    # Purpose\n
        Scan and extract file paths (export as pandas data frame).\n 
        Optionally, the function can also construct file labels using
            folder names and exports as a numpy array.\n 

    # Arguments\n
        path: str. The root directory path to scan.\n
        autoLabel: bool. If to automatically construct file labels using folder names.\n
        targetExt: str. Optionally set target file extension to extract.

    # Return\n
        Pandas data frame containing all file paths. Optionally, a numpy array with all
            file labels. Order: file_path, labels.\n

    # Details\n
        - When targetExt=None, the function scans root and sub directories.\n 
        - The targetExt string should exclude the "." symbol, e.g. 'txt' instead of '.txt'.\n
        - The function returns None for "labels" when autoLabel=False.\n
    """
    adjmat_paths = list(scanFiles(dir, validExts=targetExt))
    file_annot = pd.DataFrame()

    labels = []
    for i, adjmat_path in tqdm(enumerate(adjmat_paths), total=len(adjmat_paths)):
        # os.path.sep returns "/" which is used for str.split
        label = adjmat_path.split(os.path.sep)[-2]
        file_annot.loc[i, 'path'] = adjmat_path
        labels.append(label)

    labels = np.array(labels)
    if autoLabel:
        file_annot['label'] = labels
        return file_annot, labels
    else:
        return file_annot, None


def adjmatAnnotLoaderV2(dir, targetExt=None, autoLabel=True, annotFile=None, fileNameVar=None, labelVar=None):
    """
    # Purpose\n
        Scan and extract file paths (export as pandas data frame).\n
        Optionally, the function can also construct file labels using
            folder names and exports as a numpy array.\n

    # Arguments\n
        path: str. The root directory path to scan.\n
        targetExt: str. Optionally set target file extension to extract.\n
        autoLabel: bool. If to automatically construct file labels using folder names.\n
        annotFile: str or None. Required when autoLabel=False, a csv file for file names and labels.\n
        fileNameVar: str or None. Required when autoLabel=False, variable name in annotFile for file names.\n
        labelVar: str or None. Required when autoLabel=False, variable nam ein annotFile for lables.\n

    # Return\n
        Pandas data frame containing all file paths. Optionally, a numpy array with all
            file labels. Order: file_path, labels.\n

    # Details\n
        - When targetExt=None, the function scans root and sub directories.\n
        - The targetExt string should exclude the "." symbol, e.g. 'txt' instead of '.txt'.\n
        - The function returns None for "labels" when autoLabel=False.\n
        - When autoLabel=False, the sub folder is not currently supported.
            Sub folder support is not impossible. It is just too complicated to implement in a timely fashion. 
            This means all data files should be in one folder, i.e. dir.\n
        - When autoLabel=False, the CSV file should at least two columens, one for file name and one for the corresponding labels.\n
        - For regression modelling and when autoLabel=False, labelVar is used for the outcome variable.\n
    """
    # -- check arguments for autoLabel=False --
    if autoLabel == False:
        if (any(annotF is None for annotF in [annotFile, fileNameVar, labelVar])):
            raise ValueError(
                'Set annotFile, fileNameVar and labelVar when autoLabel=False.')
        else:
            annotFile_path = os.path.normpath(
                os.path.abspath(os.path.expanduser(annotFile)))

            if os.path.isfile(annotFile_path):
                # return full_path
                _, file_ext = os.path.splitext(annotFile_path)
                if file_ext != '.csv':
                    raise ValueError('annotFile needs to be .csv type.')
            else:
                raise ValueError('Invalid annotFile or annotFile not found.')

            annot_pd = pd.read_csv(annotFile_path, engine='python')
            if not all(annot_var in annot_pd.columns for annot_var in [fileNameVar, labelVar]):
                raise ValueError(
                    'fileNameVar and labelVar should both be present in the annotFile')

    # -- scan files --
    adjmat_paths = list(scanFiles(dir, validExts=targetExt))
    file_annot = pd.DataFrame()

    # -- labels --
    if autoLabel:
        labels = []
        for i, adjmat_path in tqdm(enumerate(adjmat_paths), total=len(adjmat_paths)):
            # os.path.sep returns "/" which is used for str.split
            label = adjmat_path.split(os.path.sep)[-2]
            file_annot.loc[i, 'path'] = adjmat_path
            labels.append(label)

        labels = np.array(labels)
        file_annot['label'] = labels
    else:  # manual label
        # Check duplicated files
        dup = sameFileCheck(dir=dir, validExts=targetExt)
        if len(dup) > 0:
            raise ValueError(
                f'File duplicates found when autoLabel=False: {dup}')

        labels = annot_pd[labelVar].to_numpy()
        manual_filenames = annot_pd[fileNameVar].to_list()
        manual_filename_paths = []
        for manual_filename in tqdm(manual_filenames):
            manual_filename_paths.append(
                list(findFilePath(manual_filename, dir)))
        manual_filename_paths = flatten(manual_filename_paths)

        file_annot['filename'] = annot_pd[fileNameVar]
        file_annot['path'] = manual_filename_paths
        file_annot['label'] = annot_pd[labelVar]

    return file_annot, labels


def labelMapping(labels, sep=None, pd_labels_var_name=None):
    """
    # Purpose\n
        Extract elements from a string collection using a pre-set seperator as labels (multiclass/multilabel/binary).\n

    # Arguments\n
        labels: pandas DataFrame or numpy ndarray. Input label string collections.\n
        sep: str. Separator string. Default is ' '.\n
        pd_labels_var_name: str. Set when labels is a pandas DataFrame, the variable/column name for label string collection.\n

    # Return\n
        One list and three dictionaries (in the following order): labels_list, labels_count, labels_map, labels_map_rev\n
        labels_list: a list with separated labels\n
        labels_map: key is labels, with int series as values\n
        labels_map_rev: key is int series, with key as values\n
        labels_count: key is labels, with counts as values\n

    # Details\n
        When sep=None, the function will no splice the label strings\n
    """

    # - argument check -
    if isinstance(labels, pd.DataFrame):
        if pd_labels_var_name is None:
            raise TypeError(
                'Set pd_labels_var_name when labels is a pandas DataFrame.')
        else:
            lbs = labels[pd_labels_var_name].to_numpy()
    elif isinstance(labels, np.ndarray):
        lbs = labels
    else:
        raise TypeError(
            'labels need to be ether a pandas DataFrame or numpy ndarray.')

    # - initial variables -
    if sep is not None:
        sep = str(sep)
        sep = sep

    # - map labels -
    labels_collect = set()
    labels_list = list()
    for i in range(len(lbs)):
        # convert sep separated label strings into an array of tags
        tags = lbs[i].split(sep)
        # add tags to the set of labels
        # NOTE: set does not allow duplicated elements
        labels_collect.update(tags)
        labels_list.append(tags)

    # - count labels -
    labels_count_list = flatten(labels_list)
    labels_count = {i: labels_count_list.count(i) for i in labels_count_list}

    # set (no order) needs to be converted to list (order) to be sorted.
    labels_collect = list(labels_collect)
    labels_collect.sort()

    # set a label set dictionary for indexing
    labels_map = {labels_collect[i]: i for i in range(len(labels_collect))}
    labels_map_rev = {i: labels_collect[i] for i in range(len(labels_collect))}

    return labels_list, labels_count, labels_map, labels_map_rev


def labelOneHot(labels_list, labels_map):
    """
    # Purpose\n
        One hot encode for labels (multiclass/multilabel/binary).\n

    # Arguments\n
        labels_list: list of strings. Input labels collection in the form of a list of strings.\n
        labels_map: dict. A map of labels.\n 

    # Return\n
        A numpy array with one hot encoded multiple labels, and can be used as the "y" input for tensorflow models.\n

    # Details\n
        - labels_list can be created by the function multilabel_mapping from utils.data_utils.\n

        - Example for labels_map (can be created by the function multilabel_mapping from utils.data_utils):
            >>> labels_map\n
            {'all': 0,
            'alpha': 1,
            'beta': 2,
            'fmri': 3,
            'hig': 4,
            'megs': 5,
            'pc': 6,
            'pt': 7,
            'sc': 8}
    """
    # - argument check -
    if not isinstance(labels_list, list):
        raise TypeError('labels_list needs to be a list object.')

    if not isinstance(labels_map, dict):
        raise TypeError('labels_map needs to be a dict type.')

    # - one hot encoding -
    one_hot_encoded = list()
    for i in range(len(labels_list)):
        encoding = np.zeros(len(labels_map), dtype='uint8')
        for sample in labels_list[i]:  # one sample is a vector of labels
            # mark 1 for each tag in the vector
            encoding[labels_map[sample]] = 1

        one_hot_encoded.append(encoding)

    one_hot_encoded = np.asarray(one_hot_encoded, dtype='uint8')

    return one_hot_encoded


def labelOneHotRev(onehot_labels, labels_map_rev):
    """
    # Purpure\n
        Convert a list of one hot encoded labels back to a list the labels.

    # Arguments\n
        onehot_labels: list or np.ndarray. A list of one hot encoded labels.
        labels_map_rev: dict. A dict of enumerated labels. 

    # Details\n
        The labels_map_rev input can be obtained from labelMapping(). \n

            Example:
            >>> labels_map_rev\n
            {0: 'all_megs_pc',
            1: 'all_megs_pt',
            2: 'sc_fmri_alpha_beta_pc',
            3: 'sc_fmri_alpha_beta_pt',
            4: 'sc_fmri_beta_hig_pc',
            5: 'sc_fmri_beta_hig_pt',
            6: 'sc_fmri_beta_pc',
            7: 'sc_fmri_beta_pt',
            8: 'sc_fmri_megs_pc',
            9: 'sc_fmri_megs_pt'}
    """
    # - check arguments -
    if not isinstance(labels_map_rev, dict):
        raise TypeError('labels_map_rev needs to be a dict.')

    if not isinstance(onehot_labels, (list, np.ndarray)):
        raise TypeError(
            'onehot_labels needs to be ether a list or a np.ndarray.')

    # - conversion -
    converted_labels = list()
    for i, label in labels_map_rev.items():
        for onehot in onehot_labels:
            if onehot[i]:
                print(label)
                converted_labels.append(label)

    return converted_labels


def getSelectedDataset(ds, X_indices_np):
    """
    modified from https://www.kaggle.com/tt195361/splitting-tensorflow-dataset-for-validation
    """
    # Make a tensor of type tf.int64 to match the one by Dataset.enumerate().
    X_indices_ts = tf.constant(X_indices_np, dtype=tf.int64)

    def is_index_in(index, rest):
        # Returns True if the specified index value is included in X_indices_ts.
        #
        # '==' compares the specified index value with each values in X_indices_ts.
        # The result is a boolean tensor, looks like [ False, True, ..., False ].
        # reduce_any() returns True if True is included in the specified tensor.
        return tf.math.reduce_any(index == X_indices_ts)

    def drop_index(index, rest):
        return rest

    # Dataset.enumerate() is similar to Python's enumerate().
    # The method adds indices to each elements. Then, the elements are filtered
    # by using the specified indices. Finally unnecessary indices are dropped.
    selected_ds = ds.enumerate().filter(is_index_in).map(drop_index)

    # calculate number of samples
    # cardinality would not work on "selected_ds"
    n = 0
    for e in selected_ds:
        n += 1

    return selected_ds, n


def trainingtestSplitterFinal(data, model_type='classification',
                              training_percent=0.8, random_state=None,
                              man_split=False, man_split_colname=None,
                              man_split_testset_value=None,
                              x_standardization=True,
                              x_min_max_scaling=False,
                              x_scale_column_to_exclude=None,
                              y_min_max_scaling=False,
                              y_column=None,
                              min_max_scale_range=(0, 1)):
    """
    # Purpose\n
        This is a training_test_splitter, with data standardization and normalization functionalities
    # Return\n
        Pandas DataFrame (for now) for training and test data.
        Scalers for training and test data sets are also returned, if applicable.
        Order: training (np.array), test (np.array), training_standard_scaler_X, training_minmax_scaler_X, training_scaler_Y
    # Arguments\n
        data: Pandas DataFrame. Input data.\n
        model_type: string. Options are "classification" and "regression".\n
        man_split: boolean. If to manually split the data into training/test sets.\n
        man_split_colname: string. Set only when man_split=True, the identity variable name for the column to use for manual splitting.\n
        man_split_testset_value: list. Set only when man_split=True, the identity variable values for test set.\n
        training_percent: float. percentage of the full data to be the training.\n
        random_state: int. seed for resampling RNG.\n
        x_standardization: boolean. if to center scale (z score standardization) the input X data.\n
        x_scale_column_to_exclude: list. the name of the columns.
                                to remove from the X columns for scaling.
                                makes sure to also include the y column(s.)\n
        y_column: list. column(s) to use as outcome for scaling.\n
        y_min_max_scaling: boolean. For regression study, if to do a Min_Max scaling to outcome.\n
        y_scale_range: two-tuple. the Min_Max range.\n
    # Details\n
        The data normalization is applied AFTER the training/test splitting
        "Standardization" is z score standardization ("center and scale"): (x - mean(x))/SD
        "min_max_scaling" is min-max normalization
        When x_standardization=True, the test data is standardized using training data mean and SD.
        When y_min_max_scaling=True, the test data is scaled using training data min-max parameters.
        For X, both standardization and min-max normalization can be applied.
        For Y, only min-max normalization can be chosen. 
        The z score standardization is used for X standardization.
    # Examples\n
        - w normalization\n
            training, test, training_scaler_X, training_scaler_Y = training_test_splitter_final(
                data=raw, random_state=1,
                man_split=True, man_split_colname='subject', man_split_testset_value=selected_features[0],
                x_standardization=True,
                x_scale_column_to_exclude=['subject', 'PCL', 'group'],
                y_min_max_scaling=True,
                y_column=['PCL'], y_scale_range=(0, 1))

        - w/o normalization\n
            training, test, _, _ = training_test_splitter_final(
                data=raw, random_state=1,
                man_split=True, man_split_colname='subject', man_split_testset_value=selected_features[1],
                x_standardization=False,
                y_min_max_scaling=False)
    """
    # argument check
    if not isinstance(data, pd.DataFrame):
        raise TypeError('Input needs to be a pandas DataFrame.')

    if x_standardization:
        if not isinstance(x_scale_column_to_exclude, list):
            raise ValueError(
                'x_scale_column_to_exclude needs to be a list.')
    if y_min_max_scaling:
        if not isinstance(y_column, list):
            raise ValueError(
                'y_column needs to be a list.')

    if man_split:
        if (not man_split_colname) or (not man_split_testset_value):
            raise ValueError(
                'set man_split_colname and man_split_testset_value when man_split=True.')
        else:
            if not isinstance(man_split_colname, str):
                raise ValueError('man_split_colname needs to be a string.')
            if not isinstance(man_split_testset_value, list):
                raise ValueError(
                    'man_split_colvaue needs to be a list.')
            if not all(test_value in list(data[man_split_colname]) for test_value in man_split_testset_value):
                raise ValueError(
                    'One or all man_split_test_value missing from the splitting variable.')

    # split
    if man_split:
        # .copy() to make it explicit that it is a copy, to avoid Pandas SettingWithCopyWarning
        training = data.loc[~data[man_split_colname].isin(
            man_split_testset_value), :].copy()
        test = data.loc[data[man_split_colname].isin(
            man_split_testset_value), :].copy()
    else:
        # training = data.sample(frac=training_percent,
        #                        random_state=random_state)
        # test = data.iloc[~data.index.isin(training.index), :].copy()

        if model_type == 'classification':  # stratified resampling
            train_idx, test_idx = list(), list()
            stf = StratifiedShuffleSplit(
                n_splits=1, train_size=training_percent, random_state=random_state)
            for train_index, test_index in stf.split(data, data[y_column]):
                train_idx.append(train_index)
                test_idx.append(test_index)
            training, test = data.iloc[train_idx[0],
                                       :].copy(), data.iloc[test_idx[0], :].copy()
        else:
            training = data.sample(frac=training_percent,
                                   random_state=random_state)
            test = data.iloc[~data.index.isin(training.index), :].copy()

    # normalization if needed
    # set the variables
    training_standard_scaler_X, training_minmax_scaler_X, training_scaler_Y = None, None, None
    if x_standardization:
        if all(selected_col in data.columns for selected_col in x_scale_column_to_exclude):
            training_standard_scaler_X = StandardScaler()
            training[training.columns[~training.columns.isin(x_scale_column_to_exclude)]] = training_standard_scaler_X.fit_transform(
                training[training.columns[~training.columns.isin(x_scale_column_to_exclude)]])
            test[test.columns[~test.columns.isin(x_scale_column_to_exclude)]] = training_standard_scaler_X.transform(
                test[test.columns[~test.columns.isin(x_scale_column_to_exclude)]])

            if x_min_max_scaling:
                training_minmax_scaler_X = MinMaxScaler(
                    feature_range=min_max_scale_range)
                training[training.columns[~training.columns.isin(x_scale_column_to_exclude)]] = training_minmax_scaler_X.fit_transform(
                    training[training.columns[~training.columns.isin(x_scale_column_to_exclude)]])
                test[test.columns[~test.columns.isin(x_scale_column_to_exclude)]] = training_minmax_scaler_X.transform(
                    test[test.columns[~test.columns.isin(x_scale_column_to_exclude)]])
        else:
            print(
                'Not all columns are found in the input X. Proceed without X standardization. \n')

    if model_type == 'regression':
        if y_min_max_scaling:
            if all(selected_col in data.columns for selected_col in y_column):
                training_scaler_Y = MinMaxScaler(
                    feature_range=min_max_scale_range)
                training[training.columns[training.columns.isin(y_column)]] = training_scaler_Y.fit_transform(
                    training[training.columns[training.columns.isin(y_column)]])
                test[test.columns[test.columns.isin(y_column)]] = training_scaler_Y.transform(
                    test[test.columns[test.columns.isin(y_column)]])
            else:
                print(
                    'Y column to scale not found. Proceed without Y scaling. \n')

    return training, test, training_standard_scaler_X, training_minmax_scaler_X, training_scaler_Y
