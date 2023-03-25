import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import utils
import clustering_methods
from sklearn.metrics import silhouette_score
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import IncrementalPCA

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import utils
import clustering_methods
from sklearn.metrics import silhouette_score
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.manifold import TSNE
import random

from matplotlib import style

matplotlib.rc('font', family='Times New Roman')
import matplotlib.cm as cm


def plot_2d():
    n_clusters = utils.CATEGORIES_AMOUNT
    n_dims = clustering_methods_optimal_dims_num['minibatchkmeans']

    ipca = IncrementalPCA(n_components=n_dims)
    ipca_test_data = ipca.fit_transform(X_test)

    minibatch, labels = clustering_methods.miniBatchKmeans(data=ipca_test_data,
                                                           n_clusters=n_clusters)
    print(set(labels))
    # [green, yellow, ...]

    #colors_different_genres = cm.rainbow(np.linspace(0, 1, utils.CATEGORIES_AMOUNT))
    colors_different_genres = ['green', 'darkblue', 'purple', 'maroon', 'brown', 'grey', 'tan',
                               'crimson', 'darkseagreen', 'peru',
                               'red', 'blue', 'olive', 'steelblue', 'palevioletred', 'darkorange']

    clustering_different_colors = ['lime', 'cornflowerblue', 'violet', 'rosybrown', 'lightcoral', 'lightgrey', 'navajowhite',
                                   'pink', 'palegreen', 'sandybrown',
                                   'coral', 'cornflowerblue', 'beige', 'lightsteelblue', 'lavenderblush', 'moccasin']
    colors__genres_dict = {}
    for i, genre in enumerate(utils.GENRES):
        colors__genres_dict[genre] = colors_different_genres[i]

    samples_colors_genres_list = []
    for genre in y_test.tolist():
        color = colors__genres_dict[genre]
        samples_colors_genres_list.append(color)



    data_2d = TSNE(n_components=2).fit_transform(ipca_test_data)

    #clustering_different_colors = ["red", "blue", "green", "magenta"]
    clustering_colors_dict = {key:val for key,val in enumerate(clustering_different_colors)}

    samples_colors_clustering_list = []
    for label in labels:
        color = clustering_colors_dict[label]
        samples_colors_clustering_list.append(color)

    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=samples_colors_clustering_list, marker='o', s=100)
    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=samples_colors_genres_list, marker='.', s=15)
    plt.xlabel("t-SNE Component 1", fontsize=16)
    plt.ylabel("t-SNE Component 2", fontsize=16)
    plt.savefig('../figures/Fig2', pad_inches=0.2, bbox_inches="tight")
    plt.show()





if __name__ == '__main__':
    data = pd.read_csv('../data/test_data.csv')
    #sample_idx = random.sample(range(data.shape[0]), 5000) # sample 200 unique indices for sampling
    X_test = data.drop(utils.GENRE_TOP_NAME, axis=1)
    y_test = data[utils.GENRE_TOP_NAME]
    #X_test = X_test.iloc[sample_idx]
    #y_test = y_test.iloc[sample_idx]
    #X_test = X_test.iloc[list(range(200))]
    #y_test = y_test.iloc[list(range(200))]

    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM
    clustering_methods_names_list = utils.CLUSTERING_METHODS_NAMES_LIST
    clustering_methods_plot_names_list = utils.CLUSTERING_METHODS_PLOT_NAMES_DICT

    fig = plt.figure(figsize=(10, 10))

    plot_2d()