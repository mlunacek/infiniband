# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Learning
# 
# IDEA:
# 
# - Since we isolate a large degreee of failure from individual features, we can use a learning model to find the set of features that have the greatest impact on our system.
# - The following will generate important features.
#     - Logistic regression
#     - Decision trees
#     - Naive Bayes
#    

# <codecell>

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# <markdowncell>

# ## Setup
# 
# Get the corpus.

# <codecell>

filename = '/lustre/janus_scratch/molu8455/infiniband/data/combined/all_cables.csv'
df = pd.read_csv(filename)
df.dropna(inplace=True)
df['norm'] = df.res.values/np.min(df.res.values)

# <markdowncell>

# Clean up the figures.

# <codecell>

mpl.rcParams['figure.figsize'] = 10,6

def clean(ax):
    for s in ['top','right']:
        ax.spines[s].set_visible(False)
    
    ax.tick_params(direction='out', length=10, pad=12,
                   width=1., colors='grey',
                   bottom='on', top='off', left='on', right='off')
    ax.grid(False)

# <markdowncell>

# Create the feature vector.

# <codecell>

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

REGEX = re.compile(r" ")
def tokenize(text):
    return [ str(tok.strip()) for tok in REGEX.split(text)]

vec = CountVectorizer(min_df=0, 
                tokenizer=tokenize, 
                stop_words=None, 
                ngram_range=(1,1))

corpus = df['path'].values
print len(corpus)
X = vec.fit_transform(corpus)
print X.shape

# <markdowncell>

# Create the resonse vector, about half the data.

# <codecell>

y = np.zeros(X.shape[0])
y[ df['norm'].values > 2 ] = 1
print 'percent', y.sum()/y.shape[0]

# <markdowncell>

# ## Fit
# 
# - Use a 5 fold cross validation method.
# - Try logistic regression

# <codecell>

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

# <codecell>

from sklearn.metrics import f1_score    
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def roc_info(y, pred):
    fpr, tpr, thresholds = roc_curve(y, pred)
    N = len(fpr)/20
    return fpr[::N], tpr[::N]

def evaluate_fold(clf, X_train, y_train, X_test, y_test):
    tmp = dict()
    
    pred_test = clf.predict_proba(X_test)
    pred_train = clf.predict_proba(X_train)
    tmp['roc'] = roc_info(y_test, pred_test[:,1])   
    tmp['roc_area'] = roc_auc_score(y_test, pred_test[:,1])
    
    pred_test = clf.predict(X_test)
    pred_train = clf.predict(X_train)
    tmp['f1_test'] = f1_score(y_test, pred_test, pos_label=1)        
    tmp['f1_train'] = f1_score(y_train, pred_train, pos_label=1) 
    
    print tmp['roc_area'], tmp['f1_train'], tmp['f1_test']
    return tmp

# <codecell>

results = []

for train_index, test_index in StratifiedShuffleSplit(y, 5, test_size=0.2):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # fit..
    clf.fit(X_train, y_train)
    tmp = evaluate_fold(clf, X_train, y_train, X_test, y_test)
    results.append(tmp)

# <markdowncell>

# ## Evaluate

# <codecell>

def plot_results(results):
    fig, ax = plt.subplots()
    
    for tmp in results:
        fpr, tpr  = tmp['roc']
        roc_auc = tmp['roc_area']
        ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic ROC')
    ax.legend(loc="lower right")
    clean(ax)
    fig.show()

# <codecell>

plot_results(results)

# <markdowncell>

# ## Learning curve

# <markdowncell>

# Create a single fold

# <codecell>

data = dict()
data['training_error'] = []
data['test_error'] = []
data['n_range'] = []

N_range = [125, 250, 500,1000,2000,4000,8000,16000,32000,64000,128000]

clf = LogisticRegression()
print X.shape

for N in N_range:
    train_index, test_index = next(iter(StratifiedShuffleSplit(y, 1, test_size=0.2)))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print X_train.shape, X_test.shape
    XN = X_train[:N]
    yN = y_train[:N]    
    model = clf.fit(XN, yN)
    data['training_error'].append(f1_score(model.predict(XN), yN))
    data['test_error'].append(f1_score(model.predict(X_test), y_test))
    data['n_range'].append(N)
    print N, yN.sum()/float(len(yN))

# <codecell>

fig, ax = plt.subplots()

ax.plot(data['n_range'], data['test_error'], '-o', label='test')
ax.plot(data['n_range'], data['training_error'], '-o', label='training')
ax.set_ylim(0,1)
ax.set_xlabel('training size')
ax.set_ylabel('F1 score')
ax.set_xscale('log')
ax.set_xlim(100, 150000)
ax.set_ylim(0,1.1)
ax.hlines(y=np.max(data['test_error']), xmin=100, xmax=150000, color='lightgrey')
ax.hlines(0.5, xmin=100, xmax=150000, color='lightgrey')

clean(ax)
fig.show()

# <codecell>


