#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 04:36:52 2018

@author: gabriel
"""
from sklearn.cluster import KMeans


def kmeans(X_train):

    model = KMeans(n_clusters=20)

    model.fit(X_train)

    y = model.labels_

    return y
