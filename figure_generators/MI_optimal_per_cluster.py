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

from matplotlib import style

matplotlib.rc('font', family='Times New Roman')


def plot_bar():
    style.use('ggplot')

    plt.figure(figsize=(8, 11))
    barWidth = 0.6

    left = np.array(range(4))
    height = []
    for clustering_method_name in clustering_methods_names_list:
        score = plot_data.loc[0, clustering_method_name]
        height.append(score)

    plt.ylim([0, 1])

    tick_label = []
    for clustering_method_name in clustering_methods_names_list:
        tick_label.append(clustering_methods_plot_names_list[clustering_method_name])

    plt.bar(left, height, color=utils.CLUSTERING_COLORS, width=barWidth, tick_label=tick_label)

    plt.xlabel('Clustering methods')
    # naming the y-axis
    plt.ylabel('MI score')
    # plot title
    plt.title('MI between clustering methods and Genres')
    plt.savefig(f'../figures/MI_optimal_per_cluster', pad_inches=0.2, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    plot_data = pd.read_csv('../data/MI_optimal_per_clustering_test_data', index_col=0)

    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM

    clustering_methods_names_list = utils.CLUSTERING_METHODS_NAMES_LIST

    clustering_methods_plot_names_list = utils.CLUSTERING_METHODS_PLOT_NAMES_DICT

    plot_bar()