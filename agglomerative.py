#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 04:45:07 2018

@author: gabriel
"""
from sklearn.cluster import AgglomerativeClustering


def agglomerative(X_train):

    model = AgglomerativeClustering()

    model.fit(X_train)

    y = model.labels_

    return y
