from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
import numpy as np

import utils


# functions return [clustering_method], [labels]
def gmm(data, n_clusters=16):
    gmm_method = GaussianMixture(n_components=n_clusters)
    #gmm_method = GaussianMixture(n_components=n_clusters, random_state=utils.RANDOM_STATE)
    gmm_method.fit(data)
    return gmm_method, gmm_method.predict(data)


def kmeans(data, n_clusters=16):
    kmeans_method = KMeans(n_clusters=n_clusters)
    #kmeans_method = KMeans(n_clusters=n_clusters, random_state=utils.RANDOM_STATE)
    kmeans_method.fit(data)
    return kmeans_method, kmeans_method.labels_


def birch(data, n_clusters=16):
    #np.random.seed(utils.RANDOM_STATE)
    birch_method = Birch(n_clusters=n_clusters)
    birch_method.fit(data)
    return birch_method, birch_method.labels_


def agglomerative(data, n_clusters=16):
    #np.random.seed(utils.RANDOM_STATE)
    agglomerative_method = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative_method.fit(data)
    return agglomerative_method, agglomerative_method.labels_


def miniBatchKmeans(data, n_clusters=16):
    mini_batch_kmeans_method = MiniBatchKMeans(n_clusters=n_clusters)
    #mini_batch_kmeans_method = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    mini_batch_kmeans_method.fit(data)
    return mini_batch_kmeans_method, mini_batch_kmeans_method.labels_


CLUSTERING_METHODS_FUNCTIONS_DICT = {'gmm': gmm,
                                     'kmeans': kmeans,
                                     'birch': birch,
                                     'minibatchkmeans': miniBatchKmeans}

CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM = {'gmm': 4,
                                           'kmeans': 7,
                                           'birch': 5,
                                           'minibatchkmeans': 4}

CLUSTERING_METHODS_OPTIMAL_DIMS_NUM = {'gmm': 5,
                                       'kmeans': 100,
                                       'birch': 5,
                                       'minibatchkmeans': 5}

# Best amount of dims for gmm is: 120, score: 0.18611671804222438
# Best amount of dims for kmeans is: 10, score: 0.23985148210301155
# Best amount of dims for birch is: 20, score: 0.22020764741729484
# Best amount of dims for agglomerative is: 50, score: 0.2523696168273341
# Best amount of dims for minibatchkmeans is: 50, score: 0.27259829699984534
