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
import scipy
from scipy.stats import f_oneway
from scipy.stats import ttest_rel


def save_silhouette_cvs_scores():
    df = pd.DataFrame(index=list(range(cv_amount)), columns=clustering_methods_names_list)

    cvs = utils.get_cvs(n_rows=X_test.shape[0], cv=cv_amount)
    for cv_idx, cv in enumerate(cvs):
        print(f'Starting cv: {cv_idx}/{cv_amount}')
        X_cv = X_test.iloc[cv]
        #print(X_cv.shape)
        for clustering_method_name in clustering_methods_names_list:
            dims_num = clustering_methods_optimal_dims_num[clustering_method_name]
            clusters_num = clustering_methods_optimal_clusters_num[clustering_method_name]

            # perform dims reduction
            ipca = IncrementalPCA(n_components=dims_num)
            ipca_test_data = ipca.fit_transform(X_cv)

            # perform clustering
            clustering_function = clustering_methods_functions_dict[clustering_method_name]
            clustering_method, labels = clustering_function(data=ipca_test_data,
                                                            n_clusters=clusters_num)
            score = silhouette_score(X_cv, labels)
            df.loc[cv_idx, clustering_method_name] = score
    df.to_csv('../data/silhouette_cvs_for_anova_test_data')


def perform_anova_test():
    scores_df = pd.read_csv('../data/silhouette_cvs_for_anova_test_data')

    # list of lists of cvs per clustering method
    scores_cvs_data = []

    for clustering_method_name in clustering_methods_names_list:
        clustering_scores = scores_df[clustering_method_name].tolist()
        scores_cvs_data.append(clustering_scores)

    print(f_oneway(*scores_cvs_data))


def perform_ttest(clustering_method_name1, clustering_method_name2):
    scores_df = pd.read_csv('../data/silhouette_cvs_for_anova_test_data')
    scores1 = scores_df[clustering_method_name1]
    scores2 = scores_df[clustering_method_name2]

    print(ttest_rel(scores1, scores2, alternative='less'))



if __name__ == '__main__':
    cv_amount = 30
    data = pd.read_csv('../data/test_data.csv')
    print(data.shape)
    X_test = data.drop(utils.GENRE_TOP_NAME, axis=1)
    # y = data[[utils.GENRE_TOP_NAME]]

    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM

    clustering_methods_names_list = utils.CLUSTERING_METHODS_NAMES_LIST
    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT
    #save_silhouette_cvs_scores()

    # now we have 30 silhouette scores for each clustering method from test data
    # perform_anova_test()

    # we got pvalue=1.59e-26 for the ANOVA test

    # now we perform t test (two-sample t-test on two dependent/related samples)
    # between the 2 best clustering methods
    # (agglomerative, minibatchkmeans)
    # perform_ttest('birch', 'minibatchkmeans')


