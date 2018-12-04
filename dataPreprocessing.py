#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:36:11 2018

@author: gabriel
"""
# import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA


def processData(
        filename='Dumitrescu Gabriel-Horia.csv',
        vectorizerName='tfid',
        randomSeed=0,
        testSet=1,
        reduceDim=0):

    # load the data
    df = pd.read_csv(filename)

    # drop unnecessary column
    df = df.drop(['Unnamed: 0'], axis=1)

    # dataset
    X = df.values.ravel()
    classes = df.columns.values.tolist()
    y = classes * 20

    # encoding string labels into numeric values
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)  # [:2000]

    # text embedding
    if vectorizerName == 'count':
        vectorizer = CountVectorizer()
    elif vectorizerName == 'tfid':
        vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X).toarray()  # [:, :2000]

    if reduceDim:
        pca = PCA(n_components=reduceDim)
        pca.fit(X)
        X = pca.components_

    if testSet:
        # split data into representative train/test sets
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                     random_state=randomSeed)
        train_index, test_index = next(sss.split(X, y))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        return X_train, X_test, y_train, y_test, classes

    else:
        return X, y, classes
