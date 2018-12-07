#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 20:54:04 2018

@author: gabriel
"""
from sklearn.svm import SVC
from gridSearch import gridSearch


def svm(X_train, X_test, y_train, method=1):

    if method == 0:

        model = SVC()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

    if method == 1:
        # params from gridSearch
        model = SVC(C=30, gamma=1)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

    elif method == 2:

        model = SVC()

        param_grid = [
                    {
                        'C': [0.001, 0.01, 0.1, 1, 10, 30, 70],
                        'kernel': ['rbf', 'linear', 'sigmoid'],
                        'gamma': [0.001, 0.01, 0.1, 1],
                    },
                    {
                        'C': [0.001, 0.01, 0.1, 1, 10, 30, 70],
                        'kernel': ['poly'],
                        'degree': [1, 3],
                        'gamma': [0.001, 0.01, 0.1, 1],
                    }
                    ]

        grid_search = gridSearch(X_train, y_train, model, param_grid)

        y_pred = grid_search.predict(X_test)

    return y_pred
