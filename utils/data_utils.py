"""data utilities"""

# ------ modules ------
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ------ functions -------
def scan_files(basePath, validExts=None, contains=None):
    """
    # Purpose:
        Scan subdirs and extract file paths.

    # Arguments:
        basePath: str. Directory path to scan.
        validExts: str. (Optional) File extension to target.
        contains: str. String included in the file name to scan. 

    # Return
        A multi-line string containing file paths

    # Details:
        1. The function scans both root and sub directories.

        2. This is a modified version of imutils.list_files,
            in which the function no longer verifies if the
            file is a image. Instead, it optionally only grabs
            files with the pre-set extension. 

        3. When validExts=None, the funciton extracts all files.
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
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                filePath = os.path.join(rootDir, filename)
                yield filePath  # yield is "return" without terminating the function


def adjmat_annot_loader(dir, autoLabel=True, targetExt=None):
    """
    # Purpose:
        Scan and extract file paths. Optionally, the function can also construct file labels using
            folder names. 

    # Arguments:
        path: str. The root directory path to scan.
        autoLabel: bool. If to automatically construct file labels using folder names.
        targetExt: str. Optionally set target file extension to extract.

    # Return:
        Pandas data frame containing all file paths. Optionially, a numpy array with all
            file labels. Order: file_path, labels.

    # Details:
        When targetExt=None, the function scans root and sub directories. 
    """
    adjmat_paths = list(scan_files(dir, validExts=targetExt))
    file_annot = pd.DataFrame()

    labels = []
    for i, adjmat_path in tqdm(enumerate(adjmat_paths), total=len(adjmat_paths)):
        # os.path.sep returns "/" which is used for str.split
        label = adjmat_path.split(os.path.sep)[-2]
        file_annot.loc[i, 'path'] = adjmat_path
        labels.append(label)

    if autoLabel:
        return file_annot, labels
    else:
        return file_annot, None


def training_test_spliter_final(data, model_type='classification',
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
    # Purpose:
        This is a training_test_spliter, with data standardization and normalization functionalities
    # Return:
        Pandas DataFrame (for now) for training and test data.
        Scalers for training and test data sets are also returned, if applicable.
        Order: training (np.array), test (np.array), training_standard_scaler_X, training_minmax_scaler_X, training_scaler_Y
    # Arguments:
        data: Pandas DataFrame. Input data.
        model_type: string. Options are "classification" and "regression". 
        man_split: boolean. If to manually split the data into training/test sets.
        man_split_colname: string. Set only when fixed_split=True, the variable name for the column to use for manual splitting.
        man_split_testset_value: list. Set only when fixed_split=True, the splitting variable values for test set.
        training_percent: float. percentage of the full data to be the training
        random_state: int. seed for resampling RNG
        x_standardization: boolean. if to center scale (z score standardization) the input X data
        x_scale_column_to_exclude: list. the name of the columns
                                to remove from the X columns for scaling.
                                makes sure to also inlcude the y column(s)
        y_column: list. column(s) to use as outcome for scaling
        y_min_max_scaling: boolean. For regression study, if to do a Min_Max scaling to outcome
        y_scale_range: two-tuple. the Min_Max range.
    # Details:
        The data normalization is applied AFTER the training/test splitting
        "Standardization" is z score standardization ("center and scale"): (x - mean(x))/SD
        "min_max_scaling" is min-max nomalization
        When x_standardization=True, the test data is standardized using training data mean and SD.
        When y_min_max_scaling=True, the test data is scaled using training data min-max parameters.
        For X, both standardiation and min-max normalization can be applied.
        For Y, only min-max normalization can be chosen. 
        The z score standardization is used for X standardization.
    # Examples
    1. with normalization
        training, test, training_scaler_X, training_scaler_Y = training_test_spliter_final(
            data=raw, random_state=1,
            man_split=True, man_split_colname='subject', man_split_testset_value=selected_features[0],
            x_standardization=True,
            x_scale_column_to_exclude=['subject', 'PCL', 'group'],
            y_min_max_scaling=True,
            y_column=['PCL'], y_scale_range=(0, 1))
    2. without noralization
        training, test, _, _ = training_test_spliter_final(
            data=raw, random_state=1,
            man_split=True, man_split_colname='subject', man_split_testset_value=selected_features[1],
            x_standardization=False,
            y_min_max_scaling=False)
    """
    # argument check
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input needs to be a pandas DataFrame.")

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

    if model_type == "regression":
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
