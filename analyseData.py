#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 19:38:24 2018

@author: gabriel
"""
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from dataPreprocessing import processData
import numpy as np


def get_top_n_words(corpus, n=None):
    """List the top n words in a vocabulary according to occurrence in a text corpus.
    (Excluding the most common words of the english language)"""
    vec = CountVectorizer(max_df=0.9).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def plotDistribution(freq, author):
    freq=list(map(list, zip(*freq)))

    word_pos = np.arange(len(freq[0]))

    fig, ax = plt.subplots()
    plt.bar(word_pos, freq[1], align='center', alpha=0.5)
    plt.xticks(word_pos, freq[0])

    plt.ylabel('Count')
    plt.title('Most frequent words of ' + author)

    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    plt.show()


def analyseData():
    """Plots most frequent words in the dataset for each author"""
    X, y, classes = processData(encodeLabels=False, vectorizerName='None', testSet=0)
    # assert X.shape[0] == 400 and len(y) == 400 and len(classes) == 20
    for i, author in enumerate(classes):
        authorBooks=[]
        for j in range(20):
            idx=(i+1)*(j+1)-1
            authorBooks.append(X[idx])
        freq=get_top_n_words(authorBooks, n=10)
        plotDistribution(freq, author)
        #print(author, freq, '\n')