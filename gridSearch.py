#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 22:17:22 2018

@author: gabriel
"""
import numpy as np
from time import time
from sklearn.model_selection import GridSearchCV


def report(results, stop, n_top=1):
    """Utility function to report best scores"""
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("\nMean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

    print("GridSearchCV took %.2f seconds." % stop)


def writeParamsToFile(modelName, paramsGd, stop, best_score):
    params = 't: ' + str(int(stop)) + ', '
    params += 's: ' + str(best_score) + ': '
    params += modelName + '('
    for key, val in paramsGd.items():
        if isinstance(val, str):
            strVal = val.__repr__()
        else:
            strVal = str(val)
        params += key + ' = ' + strVal + ', '
        params = params[:-2]
        params += ')\n'

        f = open("bestParams.txt", "a")
        f.write(params)


def gridSearch(X_train, y_train, model, param_grid):

    grid_search = GridSearchCV(model, param_grid, cv=3,
                               error_score=np.nan, n_jobs=1, verbose=0)

    start = time()

    grid_search.fit(X_train, y_train)

    stop = time() - start

    report(grid_search.cv_results_, stop)

    writeParamsToFile(model.__class__.__name__, grid_search.best_params_,
                      stop, grid_search.best_score_)

    return grid_search
