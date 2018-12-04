#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 04:40:29 2018

@author: gabriel
"""
from sklearn.cluster import DBSCAN


def dbscan(X_train):

    model = DBSCAN()

    model.fit(X_train)

    y = model.labels_

    return y
