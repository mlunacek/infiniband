# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # The feature space

# <codecell>

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# <codecell>

filename = '/lustre/janus_scratch/molu8455/infiniband/data/combined/all_cables.csv'
df = pd.read_csv(filename)
df.dropna(inplace=True)
df['norm'] = df.res.values/np.min(df.res.values)
norm = df['norm'].values
corpus = df['upath'].values

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

# ## Image of feature space

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

corpus = df['upath'].values
print len(corpus)
X = vec.fit_transform(corpus)
print X.shape

# <markdowncell>

# Sort, and pick out the top and bottom `top`.

# <codecell>

top = 10000
ordering = np.argsort(df['norm'].values)
top_index = ordering[:top]
bot_index = ordering[-top:]
index = np.concatenate([top_index, bot_index])

# <markdowncell>

# Find the most intersting features.

# <codecell>

from sklearn.feature_selection import SelectKBest, chi2

def plot_features(vec):

    # Create pass or fail vector
    X = vec.fit_transform(corpus)
    y = np.zeros(X.shape[0])
    y[norm > 2] = 1
    
    ch2 = SelectKBest(chi2, k=2000)
    X_best = ch2.fit_transform(X, y)
    
    print X_best.shape
    Xd = X_best.toarray()
    Xd[index].shape
    
    fig, ax = plt.subplots()
    ax.imshow(Xd[index], cmap=plt.cm.gray_r, interpolation='nearest', aspect=0.05)

    fig.show()
    

# <codecell>

vec = CountVectorizer(min_df=0, 
                tokenizer=tokenize, 
                stop_words=None, 
                ngram_range=(1,1))

plot_features(vec)

# <codecell>


vec = TfidfVectorizer(min_df=0, 
                tokenizer=tokenize, 
                stop_words=None, 
                ngram_range=(1,1))

plot_features(vec)

# <codecell>

def lsa(vec, threshold=2, corpus=corpus, norm=norm, N=3):
    
    # Create the feature vector
    X = vec.fit_transform(corpus)
    
    # Create the response vector
    y = np.zeros(X.shape[0])
    y[norm > threshold] = 1.0

    
    print y.sum()/float(len(y))
    print(X.shape)
    
    lsa = TruncatedSVD(N, algorithm='arpack')
    Xl = lsa.fit_transform(X)
    Xn = Normalizer(copy=False).fit_transform(Xl)
 
    return Xn, y

def plot_lsa(Xn, y):
        
    passed = y == 0.0
    failed = y == 1.0
    
    fig, ax = plt.subplots(1,2, figsize=(12,8))
    
    tmp = ax[0]
    tmp.scatter(Xn[failed,0], Xn[failed,1], c='red', alpha=0.01, s=5, edgecolor='none')
    tmp.scatter(Xn[passed,0], Xn[passed,1], c='green', alpha=0.01, s=5, edgecolor='none')
    tmp.set_xlabel('x0')
    tmp.set_ylabel('x1')
    clean(tmp)
    
    tmp = ax[1]
    tmp .scatter(Xn[failed,2], Xn[failed,1], c='red', alpha=0.01, s=5, edgecolor='none')
    tmp .scatter(Xn[passed,2], Xn[passed,1], c='green', alpha=0.01, s=5, edgecolor='none')
    tmp .set_xlabel('x2')
    tmp .set_ylabel('x1')
    clean(tmp )

    fig.show()

# <codecell>

Xn, y = lsa(vec, threshold=2, corpus=corpus, norm=norm, N=25)

# <codecell>

ordering = np.argsort(Xn[:,0])
ordering = np.argsort(norm)
fig, ax = plt.subplots()
ax.imshow(Xn[ordering], cmap=plt.cm.gray_r, interpolation='nearest', aspect=0.00005)
fig.show()

# <markdowncell>

# ## Clustering

# <codecell>

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity

# <codecell>

vec = TfidfVectorizer(min_df=0, 
                tokenizer=tokenize, 
                stop_words=None, 
                ngram_range=(1,1))

vec = CountVectorizer(min_df=0, 
                tokenizer=tokenize, 
                stop_words=None, 
                ngram_range=(1,1))

norm = df['norm'].values
corpus = df['upath'].values

# <markdowncell>

# About 75% fails.

# <codecell>

Xn, y= lsa(vec, threshold=1.5, corpus=corpus, norm=norm)
plot_lsa(Xn, y)

# <markdowncell>

# With a threshold of `2.5`, about fails 40% pass

# <codecell>

Xn, y= lsa(vec, threshold=2.5, corpus=corpus, norm=norm)
plot_lsa(Xn, y)

# <markdowncell>

# With `threshold=2.0`, about 0.5 the tests pass.

# <codecell>

Xn, y= lsa(vec, threshold=2.0, corpus=corpus, norm=norm, N=10)
plot_lsa(Xn, y)

# <markdowncell>

# ## Cosine similarity...

# <codecell>

print Xn
# sim = cosine_similarity(Xn[0], Xn)
# plt.plot(sim)

# <codecell>


