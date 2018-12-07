#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 20:52:19 2018

@author: gabriel
"""
from dataPreprocessing import processData
from svm import svm
from kmeans import kmeans
from dbscan import dbscan
from agglomerative import agglomerative
from evaluateClassification import evaluateClassification
from evaluateClustering import evaluateClustering
"""
# loading pre-processed data
X_train, X_test, y_train, y_test, classes = processData(vectorizerName='count')

y_svm = svm(X_train, X_test, y_train)
evaluateClassification(y_test, y_svm, 'Support Vector Machines', classes)
"""
X, y, classes = processData(testSet=0, reduceDim=2000)

y_kmeans = kmeans(X)
evaluateClustering(X, y, y_kmeans, 'K-means Clustering')

y_dbscan = dbscan(X)
evaluateClustering(X, y, y_dbscan, 'DBSCAN Clustering')

y_agglomerative = agglomerative(X)
evaluateClustering(X, y, y_agglomerative, 'Agglomerative Clustering')
