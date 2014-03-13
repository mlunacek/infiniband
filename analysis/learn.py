#!/usr/bin/env python

from __future__ import print_function

import os
import glob
import re
import json
import pandas as pd
from functools import partial
import itertools
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score    
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

    
d_path = '/lustre/janus_scratch/molu8455/infiniband/data/parsed'

REGEX = re.compile(r" ")
def tokenize(text):
    return [tok.strip().lower() for tok in REGEX.split(text)]

def get_corpus(files, threshold=20000):
    df = pd.concat(map(lambda f: pd.read_csv(f), files))
    df.dropna(inplace=True)
    corpus = df['0'].values
    y = df['res'].astype(float).values
    
    y[y < threshold] = 0
    y[y > threshold] = 1
    return y, corpus

def create_files(prefix_list):
    for prefix in prefix_list:
        yield glob.glob(os.path.join(d_path, prefix + '*.csv'))

# Parameters
def create_vectorizers(ngrams):
    for ngram in ngrams:
        yield CountVectorizer(min_df=0, 
                                tokenizer=tokenize, 
                                stop_words=None, 
                                ngram_range=(1,ngram))
        yield TfidfVectorizer(min_df=0, 
                                tokenizer=tokenize, 
                                stop_words=None, 
                                ngram_range=(1,ngram))

def create_chi2(kvalues):
    for k in kvalues:
        yield SelectKBest(k=k, score_func=chi2)


def create_set(**kwargs):

    v = CountVectorizer(tokenizer=tokenize)
    d = dict()
    d['kfold'] = kwargs.get('kfolds',[5])
    d['est'] = kwargs.get('est',[LogisticRegression()])
    pre = kwargs.get('pre',['cables'])
    d['vec'] = kwargs.get('vecorizers', [v])
    d['files'] = list(create_files(pre))

    chi_lst = kwargs.get('chilist', None)
    d['chi'] = list()
    if chi_lst:
        d['chi'] = list(create_chi2(chi_lst))
    d['chi'].append(None)
    return expandgrid(d)

def expandgrid(source):
    labels, terms = zip(*source.items())
    d = [dict(zip(labels, term)) for term in itertools.product(*terms)]
    return d

def roc_info(y,pred):
    fpr, tpr, thresholds = roc_curve(y, pred)
    #print(len(fpr), len(tpr), len(thresholds))
    return None

def evaluate_fold(clf, X_train, y_train, X_test, y_test):
    """
    This is the business section
    """
    tmp = dict()
    tmp['X_train.shape'] = X_train.shape
    tmp['X_test.shape'] = X_test.shape
    try:
        pred_test = clf.predict_proba(X_test)
        pred_train = clf.predict_proba(X_train)
        tmp['roc'] = roc_info(y_test, pred_test[:,1])   
        tmp['roc_area'] = roc_auc_score(y_test, pred_test[:,1])
        pred_test = clf.predict(X_test)
        pred_train = clf.predict(X_train)
        tmp['f1_test'] = f1_score(y_test, pred_test, pos_label=1)        
        tmp['f1_train'] = f1_score(y_train, pred_train, pos_label=1) 

    except (AttributeError, NotImplementedError):
        pred_test = clf.predict(X_test)
        pred_train = clf.predict(X_train)
        tmp['roc'] = roc_info(y_test, pred_test)
        tmp['roc_area'] = roc_auc_score(y_test, pred_test)
        tmp['f1_test'] = f1_score(y_test, pred_test, pos_label=1)        
        tmp['f1_train'] = f1_score(y_train, pred_train, pos_label=1) 

    return tmp

def evaluate(test):
    y, corpus = get_corpus(test['files'])
    X = test['vec'].fit_transform(corpus)
    #print('X.shape', X.shape)

    clf = test['est']
    
    data = dict()
    data['vec'] = str(test['vec'])
    data['est'] = str(clf)
    data['kfold'] = str(test['kfold'])
    data['chi'] = str(test['chi'])
    data['files'] = test['files']

    data['f1_train'] = 0.
    data['f1_test'] = 0.
    data['roc_area'] = 0.
    # iterator over each set
    for train_index, test_index in StratifiedShuffleSplit(y, test['kfold'], test_size=0.2):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if test['chi']:
            X_train = test['chi'].fit_transform(X_train, y_train)
            X_test = test['chi'].transform(X_test)

        try:
            clf.fit(X_train, y_train)
            tmp = evaluate_fold(clf, X_train, y_train, X_test, y_test)
            data['f1_train'] += tmp['f1_train']/float(test['kfold'])
            data['f1_test'] += tmp['f1_test']/float(test['kfold'])
            data['roc_area'] += tmp['roc_area']/float(test['kfold'])
            data['X_train.shape'] = tmp['X_train.shape']
            data['X_test.shape'] = tmp['X_test.shape']

        except TypeError, e:
            print(e)

    # average the folds

    return data

def create_tests():

    vecorizers = [ ]
    vecorizers.extend(list(create_vectorizers([1,2])))

    estimators = [] 
    cvalues = np.logspace(-2, 2, 20)
    for c in cvalues:
        estimators.append(LogisticRegression(C=c))
        estimators.append(LinearSVC(C=c))

    # smoothing = np.linspace(0.25, 0.75, 20)
    # for s in smoothing:
    #     estimators.append(MultinomialNB(alpha=s))


    tests = create_set( vecorizers=vecorizers,
                        chilist=[1000, 2000], 
                        pre=['cables','switches','ports'],
                        est=estimators,
                        kfolds=[5,10])

    return tests

if __name__ == '__main__':
      

    tests = create_tests()
    #map(print, tests)

    # test = create_set()
    # print(evaluate(test[0]))
    print('number of tests =', len(tests))
    #results = map(evaluate, tests[:1])

    # with open('results.json','w') as outfile:
    #     for r in results:
    #         outfile.write(json.dumps(r)+'\n')


