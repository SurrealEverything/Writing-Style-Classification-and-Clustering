#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 04:36:52 2018

@author: gabriel
"""
from sklearn.cluster import KMeans


def kmeans(X, n_clusters=20, init='k-means++'):

    model = KMeans(n_clusters=n_clusters, init=init)

    model.fit(X)

    y = model.labels_

    return y
