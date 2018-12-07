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
import pandas as pd
import matplotlib.pyplot as plt


def evaluateClustering(X_true, y_true, y_pred, model):

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
    print('Contingency Matrix:\n', cm)

    # Create a DataFrame with labels and varieties as columns: df
    df = pd.DataFrame({'Labels': y_true, 'Clusters': y_pred})
    # Create crosstab: ct
    ct = pd.crosstab(df['Labels'], df['Clusters'])
    # Plot non-normalized confusion matrix
    print('Cross-tabulation:\n', ct)
