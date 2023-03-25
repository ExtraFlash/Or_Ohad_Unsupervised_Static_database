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


def find_num_of_dims_silhouette(clustering_method_name):
    max_score = 0
    best_dims_num = 0
    df = pd.DataFrame(columns=dims_num_list)

    for dims_num in dims_num_list:
        print(f'{clustering_method_name}, dims_num = {dims_num}')
        silhouette_scores = []

        cvs = utils.get_cvs(n_rows=X.shape[0], cv=5)
        for cv in cvs:
            X_train = X.iloc[cv]

            # perform dims reduction
            ipca = IncrementalPCA(n_components=dims_num)
            ipca_train_data = ipca.fit_transform(X_train)

            # perform clustering
            clustering_function = clustering_methods_functions_dict[clustering_method_name]
            clustering_method, labels = clustering_function(data=ipca_train_data,
                                                            n_clusters=clustering_methods_optimal_clusters_num[
                                                                clustering_method_name])
            score = silhouette_score(X_train, labels)
            silhouette_scores.append(score)

        mean = np.mean(silhouette_scores)
        if mean > max_score:
            max_score = mean
            best_dims_num = dims_num
        df.loc[0, dims_num] = mean
    df.to_csv(f'data/silhouette_best_dims_num_train_data/{clustering_method_name}')
    print(f'Best amount of dims for {clustering_method_name} is: {best_dims_num}, score: {max_score}')

if __name__ == '__main__':
    data = pd.read_csv('data/train_data.csv')
    X = data.drop(utils.GENRE_TOP_NAME, axis=1)
    # y = data[[utils.GENRE_TOP_NAME]]

    dims_num_list = [5, 10, 20, 25, 50, 100, 120]

    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM

    clustering_methods_names_list = utils.CLUSTERING_METHODS_NAMES_LIST
    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT

    # find optimal number of dims for all clustering methods
    for clustering_method_name_ in clustering_methods_names_list:
        find_num_of_dims_silhouette(clustering_method_name_)
