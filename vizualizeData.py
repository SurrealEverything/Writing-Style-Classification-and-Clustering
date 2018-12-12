#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 02:31:31 2018

@author: gabriel
"""
from mpl_toolkits import mplot3d
from ipywidgets import interact, fixed
import matplotlib.pyplot as plt
from dataPreprocessing import processData
from sklearn.manifold.t_sne import TSNE
from sklearn.decomposition import PCA


def isInteractive():
    import __main__ as main
    return not hasattr(main, '__file__')


def plot_3D(elev, azim, X, y, title):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y, s=50)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(title)


def plotInteractive(X, y, title):
    interact(plot_3D, elev=(-90, 90), azim=(-180, 180),
             X=fixed(X), y=fixed(y), title=fixed(title))


def plotStationary(X, y, title):
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(title)
    plt.show()


def vizualizeData(X_svm, y_svm, y_kmeans, y_dbscan, y_agglomerative):

    interactive = isInteractive()

    # X_PCA and true y
    X_PCA, y, classes = processData(encodeLabels=True, testSet=0, reduceDim=3)

    # X_TSNE
    X, y, classes = processData(encodeLabels=True, testSet=0)
    X_TSNE = TSNE(n_components=3).fit_transform(X)

    # X_PCA and X_TSNE for svm
    X_svm_PCA = PCA(n_components=3).fit_transform(X_svm)
    X_svm_TSNE = TSNE(n_components=3).fit_transform(X_svm)


    if interactive:
        plotInteractive(X_PCA, y, 'PCA')
        plotInteractive(X_TSNE, y, 'TSNE')

        plotInteractive(X_svm_PCA, y_svm, 'Support Vector Machines PCA')
        plotInteractive(X_svm_TSNE, y_svm, 'Support Vector Machines TSNE')

        plotInteractive(X_PCA, y_kmeans, 'K-means Clustering PCA')
        plotInteractive(X_TSNE, y_kmeans, 'K-means Clustering TSNE')

        plotInteractive(X_PCA, y_dbscan, 'DBSCAN Clustering PCA')
        plotInteractive(X_TSNE, y_dbscan, 'DBSCAN Clustering TSNE')

        plotInteractive(X_PCA, y_agglomerative, 'Agglomerative Clustering PCA')
        plotInteractive(X_TSNE, y_agglomerative,
                        'Agglomerative Clustering TSNE')

    else:
        plotStationary(X_PCA, y, 'PCA')
        plotStationary(X_TSNE, y, 'TSNE')

        plotStationary(X_svm_PCA, y_svm, 'Support Vector Machines PCA')
        plotStationary(X_svm_TSNE, y_svm, 'Support Vector Machines TSNE')

        plotStationary(X_PCA, y_kmeans, 'K-means Clustering PCA')
        plotStationary(X_TSNE, y_kmeans, 'K-means Clustering TSNE')

        plotStationary(X_PCA, y_dbscan, 'DBSCAN Clustering PCA')
        plotStationary(X_TSNE, y_dbscan, 'DBSCAN Clustering TSNE')

        plotStationary(X_PCA, y_agglomerative, 'Agglomerative Clustering PCA')
        plotStationary(X_TSNE, y_agglomerative, 'Agglomerative Clustering TSNE')
