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


def find_num_of_clusters_silhouette(clustering_method_name):
    max_score = 0
    best_clusters_num = 0
    df = pd.DataFrame(columns=clusters_num_list)
    for clusters_num in clusters_num_list:
        silhouette_scores = []
        cvs = utils.get_cvs(n_rows=X.shape[0], cv=5)
        for i,cv in enumerate(cvs):
            print(f'{clustering_method_name}, n_clusters: {clusters_num}, cv: {i}')
            X_train = X.iloc[cv]

            clustering_function = clustering_methods_functions_dict[clustering_method_name]
            clustering_method, labels = clustering_function(X_train, n_clusters=clusters_num)
            score = silhouette_score(X_train, labels)
            silhouette_scores.append(score)
        mean = np.mean(silhouette_scores)
        if mean > max_score:
            max_score = mean
            best_clusters_num = clusters_num
        df.loc[0, clusters_num] = mean
    df.to_csv(f'data/silhouette_best_clusters_num_train_data/{clustering_method_name}')
    print(f'Best amount of clusters for {clustering_method_name} is: {best_clusters_num}, score: {max_score}')


if __name__ == '__main__':
    data = pd.read_csv('data/train_data.csv')
    X = data.drop(utils.GENRE_TOP_NAME, axis=1)
    # y = data[[utils.GENRE_TOP_NAME]]

    dims_num_list = [5, 10, 20, 25, 50, 100, 120]
    clusters_num_list = list(range(2, utils.CATEGORIES_AMOUNT + 1))  # from 2 to the number of categories

    clustering_methods_names_list = utils.CLUSTERING_METHODS_NAMES_LIST
    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT

    # find optimal number of clusters for all clustering methods
    for clustering_method_name_ in clustering_methods_names_list:
        find_num_of_clusters_silhouette(clustering_method_name_)
