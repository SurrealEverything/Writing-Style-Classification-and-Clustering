#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 01:38:52 2018

@author: gabriel
"""
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             homogeneity_score, completeness_score,
                             fowlkes_mallows_score, v_measure_score,
                             calinski_harabaz_score)
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


def evaluateClustering(X_true, y_true, y_pred, model, classes):

    print('\n', model.upper())

    ars = adjusted_rand_score(y_true, y_pred)
    print('Adjusted random score: ', ars)

    ss = silhouette_score(X_true, y_pred)
    print('Silhouette score: ', ss)

    hs = homogeneity_score(y_true, y_pred)
    print('Homogeneity score: ', hs)

    cs = completeness_score(y_true, y_pred)
    print('Completeness score: ', cs)

    vms = v_measure_score(y_true, y_pred)
    print('V-measure score: ', vms)

    fms = fowlkes_mallows_score(y_true, y_pred)
    print('Fowlkes-Mallows score: ', fms)

    chs = calinski_harabaz_score(X_true, y_pred)
    print('Calinski-Harabaz score: ', chs)

    cm = contingency_matrix(y_true, y_pred)
    orderContingencyMatrix(cm)

    acc = accuracy(cm)
    print('Supervised-like Accuracy: ', acc)

    mean = meanScore(X_true, y_true, y_pred)
    print('Mean Score: ', mean)

    plotContingencyMatrix(cm, model, classes)


def accuracy(matrix):
    """Simillarity to a perfect confusion matrix"""
    n, m = matrix.shape
    if n != m:
        return 0

    correctSum = 0
    for i in range(min(n, m)):
        val = matrix[i, i]
        correctSum += min(val, 20)

    totalSum = n*n
    acc = correctSum/totalSum
    return acc


def orderContingencyMatrix(matrix):
    """Orders contingency matrix to look like a confusion matrix
    (can be improved with dynamic programming)"""
    n, m = matrix.shape
    for i in range(min(n, m)):
        idx = np.argmax(matrix[i, i:])
        idx += i
        temp = np.copy(matrix[:, idx])
        matrix[:, idx] = matrix[:, i]
        matrix[:, i] = temp


def plotContingencyMatrix(matrix, model, classes):

    fig, ax = plt.subplots()

    ax.matshow(matrix, cmap=plt.cm.Blues)

    n, m = matrix.shape
    thresh = matrix.max() / 2.
    for i, j in product(range(n), range(m)):
        c = matrix[i, j]
        ax.text(j, i, str(c), va='center', ha='center',
                color="white" if matrix[i, j] > thresh else "black")

    tick_marks_n = np.arange(n)
    tick_marks_m = np.arange(m)
    plt.xticks(tick_marks_m)
    plt.yticks(tick_marks_n, classes)

    plt.tight_layout()

    plt.ylabel('Labels')
    plt.xlabel('Clusters')

    plt.title(model+' Contingency Matrix', y=1.2)

    plt.show()


def meanScore(X_true, y_true, y_pred):
    ars = adjusted_rand_score(y_true, y_pred)
    ss = silhouette_score(X_true, y_pred)
    hs = homogeneity_score(y_true, y_pred)
    cs = completeness_score(y_true, y_pred)
    vms = v_measure_score(y_true, y_pred)
    fms = fowlkes_mallows_score(y_true, y_pred)
    acc = computeSupLikeAcc(y_true, y_pred)
    mean = (ars+ss+hs+cs+vms+fms+acc)/7
    return mean


def computeSupLikeAcc(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    orderContingencyMatrix(cm)
    return accuracy(cm)
