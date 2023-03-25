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
from sklearn.metrics import adjusted_mutual_info_score


def perform_anova_test():

    mi_scores_per_clustering_list = []
    for clustering_methods_name in clustering_methods_names_list:
        df = pd.read_csv(f'../data/MI/clustering_methods/{clustering_methods_name}')
        mi_scores = df[utils.GENRE_TOP_NAME].tolist()
        mi_scores_per_clustering_list.append(mi_scores)

    print(f_oneway(*mi_scores_per_clustering_list))


def save_scores():
    # MI_optimal_per_clustering_test_data
    df_mi_optimal_per_clustering_test = pd.DataFrame(columns=clustering_methods_names_list)
    for clustering_methods_name in clustering_methods_names_list:
        df = pd.read_csv(f'../data/MI/clustering_methods/{clustering_methods_name}')
        mi_scores = df[utils.GENRE_TOP_NAME].tolist()
        mean_score = np.mean(mi_scores)
        df_mi_optimal_per_clustering_test.loc[0, clustering_methods_name] = mean_score
    df_mi_optimal_per_clustering_test.to_csv('../data/MI_optimal_per_clustering_test_data')


def perform_ttest(clustering_method_name1, clustering_method_name2):
    df_scores1 = pd.read_csv(f'../data/MI/clustering_methods/{clustering_method_name1}')
    df_scores2 = pd.read_csv(f'../data/MI/clustering_methods/{clustering_method_name2}')

    scores1 = df_scores1[utils.GENRE_TOP_NAME].tolist()
    scores2 = df_scores2[utils.GENRE_TOP_NAME].tolist()

    print(ttest_rel(scores1, scores2, alternative='greater'))


if __name__ == '__main__':
    cv_amount = 30

    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_names_list = utils.CLUSTERING_METHODS_NAMES_LIST
    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT

    # perform_anova_test()
    #save_scores()
    perform_ttest('kmeans', 'minibatchkmeans')


