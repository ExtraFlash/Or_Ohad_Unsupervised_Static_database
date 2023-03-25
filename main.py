import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn as skl
from sklearn.model_selection import train_test_split
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import utils
import os
from sklearn.utils import shuffle

matplotlib.rcParams['figure.figsize'] = (20, 10)

# Press the green button in the gutter to run the script.
#if __name__ == '__main__':

# importing the data
tracks = utils.load('data/tracks.csv')
features = utils.load('data/features.csv')

large_tracks = tracks['set', 'subset'] <= 'large'
X = features.loc[large_tracks, 'mfcc']
y = tracks.loc[large_tracks, ('track', 'genre_top')]

# merging with external variable
data = pd.concat([X, y], axis=1)
data = data[data.isnull().any(axis=1) == False]

# shuffling the data
data = shuffle(data, random_state=utils.RANDOM_STATE)
data.reset_index(inplace=True, drop=True)

# changing column names from tuples to strings [0:139, genre_top]
column_names = []
for i in range(data.shape[1] - 1):
    column_names.append(f'{i}')
column_names.append('genre_top')

data.columns = column_names

# data size: (49598,). train: (32238,). test: (17360,). 70%.

# splitting to train data and test data
train_data = data.iloc[:utils.TRAIN_DATA_SIZE]
test_data = data.iloc[utils.TRAIN_DATA_SIZE:]

data.to_csv('data/data.csv')
train_data.to_csv('data/train_data.csv')
test_data.to_csv('data/test_data.csv')














#X = X[y.isnull().any(axis=1) == False]
#y = y[y.isnull().any(axis=1) == False]




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
