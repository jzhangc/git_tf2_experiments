"""
Current objectives:
small things for data loaders
"""

# ------ modules ------
import os
import pandas as pd
import numpy as np
from utils.data_utils import adjmat_annot_loader, multilabel_mapping, multilabel_one_hot
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer


# ------ function -------


# ------ test realm ------
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, 'data/tf_data')

file_annot, labels = adjmat_annot_loader(dat_dir, targetExt='txt')
file_annot['path'][0]
file_annot.loc[0:1]

# - below: create one hot encoding for multiclass labels -
lb_binarizer = LabelBinarizer()
labels_binary = lb_binarizer.fit_transform(labels)

# - below: create one hot encoding for multilabel labels -
labels_list, lables_count, labels_map, labels_map_rev = multilabel_mapping(
    labels=labels, sep='_')

# one hot encoding
encoded_labels = multilabel_one_hot(
    labels_list=labels_list, labels_map=labels_map)


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
# - real TF2 data loader example: https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py -
