#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 01:38:52 2018

@author: gabriel
"""
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             homogeneity_score, completeness_score)


def evaluateClustering(y_true, y_pred, model):

    print('\n', model.upper())

    ars = adjusted_rand_score(y_true, y_pred)
    print('Adjusted random score: ', ars)

    ss = silhouette_score(y_true, y_pred)
    print('Silhouette score: ', ss)

    hs = homogeneity_score(y_true, y_pred)
    print('Homogeneity score: ', hs)

    cs = completeness_score(y_true, y_pred)
    print('Completeness score: ', cs)

    # work in progress
