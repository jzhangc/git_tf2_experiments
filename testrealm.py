"""
Current objectives:
small things for data loaders
"""

# ------ modules ------
import os
import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer


# ------ function -------
def list_files(basePath, validExts=None, contains=None):
    """
    Scan subdirs and extract file names. 
    This is a modified version of imutils.list_files,
        in which the function no longer verifies if the
        file is a image. Instead, it optionally only grabs
        files with the pre-set extension. 
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


def adjmat_annot_loader(path):
    adjmat_paths = list(list_files(path))
    adjmat_annot = pd.DataFrame()
    labels = []
    for i, adjmat_path in tqdm(enumerate(adjmat_paths), len(adjmat_paths)):
        label = os.path.splitext(adjmat_path)[-2]
        adjmat_annot.iloc[i, 'image_path'] = adjmat_path
        labels.append(label)

    return adjmat_annot


# ------ test realm ------
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, 'data/tf_data')

tst = adjmat_annot_loader(dat_dir)

for (rootDir, dirNames, filenames) in os.walk(dat_dir):
    # loop over the filenames in the current directory
    for filename in filenames:
        print(filename)
        break


# ------ ref ------
# # https://debuggercafe.com/creating-efficient-image-data-loaders-in-pytorch-for-deep-learning/
# # get all the image paths
# image_paths = list(paths.list_images('../input/natural-images/natural_images'))
# # create an empty DataFrame
# data = pd.DataFrame()
# labels = []
# for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
#     label = image_path.split(os.path.sep)[-2]
#     data.loc[i, 'image_path'] = image_path
#     labels.append(label)

# labels = np.array(labels)
# # one hot encode
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)
# print(f"The first one hot encoded labels: {labels[0]}")
# print(f"Mapping an one hot encoded label to its category: {lb.classes_[0]}")
# print(f"Total instances: {len(labels)}")
# for i in range(len(labels)):
#     index = np.argmax(labels[i])
#     data.loc[i, 'target'] = int(index)
# # shuffle the dataset
# data = data.sample(frac=1).reset_index(drop=True)
# # save as csv file
# data.to_csv('../input/data.csv', index=False)
# # pickle the label binarizer
# joblib.dump(lb, '../outputs/lb.pkl')
# print('Save the one-hot encoded binarized labels as a pickled file.')
# print(data.head())

# # read the data.csv file and get the image paths and labels
# df = pd.read_csv('../input/data.csv')
# X = df.image_path.values
# y = df.target.values

# (xtrain, xtest, ytrain, ytest) = (train_test_split(X, y,
#                                 test_size=0.25, random_state=42))

# - tf.dataset reference: https://cs230.stanford.edu/blog/datapipeline/ -
