#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:36:11 2018

@author: gabriel
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
import gensim.models
from stemming.porter2 import stem
import nltk
from nltk.corpus import stopwords
import string
from pathlib import Path


def processData(
        fileName='Dumitrescu Gabriel-Horia.csv',
        pathName='/home/gabriel/Spyder Projects/ML/Tema 2 - Writing Style/',
        stemming=True,
        encodeLabels=True,
        vectorizerName='tfid',
        randomSeed=0,
        testSet=1,
        reduceDim=0):

    # load the data
    df = pd.read_csv(fileName)

    # drop unnecessary column
    df = df.drop(['Unnamed: 0'], axis=1)

    # dataset
    X = df.values.ravel()
    classes = df.columns.values.tolist()
    y = classes * 20
    y = np.asarray(y)

    if stemming:
        X = stemData(X, pathName)

    if encodeLabels:
        # encoding string labels into numeric values
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

    # text embedding
    if vectorizerName == 'count':
        vectorizer = CountVectorizer(strip_accents='ascii')
        X = vectorizer.fit_transform(X).toarray()
    elif vectorizerName == 'tfid':
        vectorizer = TfidfVectorizer(strip_accents='ascii')
        X = vectorizer.fit_transform(X).toarray()
    elif vectorizerName == 'word2vec':
        X = convertWords2Vec(X, pathName)

    if reduceDim:
        pca = PCA(n_components=reduceDim)
        X = pca.fit_transform(X)
        # y = pca.transform(y.reshape(-1, 1))

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


def convertWords2Vec(X, pathName):
    """functie care converteste X din cuvinte in word2vec"""

    XVEC = Path(pathName+'XVEC.csv')
    if XVEC.is_file():
        X_vec = pd.read_csv(r"XVEC.csv")
        X_vec = X_vec.drop(["Unnamed: 0"], axis=1)
        X_vec = X_vec.iloc[:, 0].values.flatten()
        print("Vectorizing word2vec from file")
        return X_vec

    assert gensim.models.word2vec.FAST_VERSION > -1
    splitX = []
    for i in range(X.shape[0]):
        splitX.append(X[i].split(" "))
    print('Splitting done')
    model = gensim.models.Word2Vec(splitX, size=6, window=1, min_count=50000)
    model.train(splitX, total_examples=1, epochs=1)
    print('Training done')
    X_vec_lis = []
    for book in splitX:
        book_vec = []
        for word in book:
            if word in model.wv.vocab:
                word_vec = model.wv[word]
                book_vec.append(word_vec)
        X_vec_lis.append(book_vec)
    print('Vectorization done')
    maxi = len(max(X_vec_lis, key=len))
    print(len(X_vec_lis), maxi, len(X_vec_lis[0][0]))
    X_vec = np.zeros([len(X_vec_lis), maxi, len(X_vec_lis[0][0])])
    for i, j in enumerate(X_vec_lis):
        X_vec[i][0:len(j)] = j

    nsamples, nx, ny = X_vec.shape
    X_vec = X_vec.reshape((nsamples, nx*ny))
    print('Ndarray convertion done')

    # saving XVEC
    pdX = pd.DataFrame(X_vec)
    pdX.to_csv("XVEC.csv")

    return X_vec


def stemData(X, pathName):
    """Stemms the data"""

    # check if stemmed data is stored to save time
    XSTEM = Path(pathName+'XSTEM.csv')
    if XSTEM.is_file():
        X_stem = pd.read_csv(r"XSTEM.csv")
        X_stem = X_stem.drop(["Unnamed: 0"], axis=1)
        X_stem = X_stem.iloc[:, 0].values.flatten()
        print("Stemming from file")
        return X_stem

    for i in range(0, len(X)):
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        X[i] = str(X[i]).translate(translator)

    X_stem = [[stem(word.lower()) for word in fragment.split(" ")] for fragment in X]

    nltk.download('stopwords')

    X_stem = [[word for word in word_list if word not in stopwords.words('english')] for word_list in X_stem]

    X_stem = [' '.join(sentence) for sentence in X_stem]

    X_stem = [sentence.replace('\n',' ') for sentence in X_stem]

    X_stem = np.asarray(X_stem)

    # saving XSTEM
    pdX = pd.DataFrame(X_stem)
    pdX.to_csv("XSTEM.csv")

    return X_stem