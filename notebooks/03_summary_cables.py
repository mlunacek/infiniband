# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Cables summary

# <codecell>

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# <codecell>

filename = '/lustre/janus_scratch/molu8455/infiniband/data/combined/all_cables.csv'
df = pd.read_csv(filename)

# <codecell>

print df.shape
print df.columns
df.dropna(inplace=True)
print df.shape

df['norm'] = df.res.values/np.min(df.res.values)
norm = df['norm'].values

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

# ## How often is each cable used?

# <codecell>

import re
from sklearn.feature_extraction.text import CountVectorizer

REGEX = re.compile(r" ")
def tokenize(text):
    return [ str(tok.strip()) for tok in REGEX.split(text)]

vec = CountVectorizer(min_df=0, 
                tokenizer=tokenize, 
                stop_words=None, 
                ngram_range=(1,1))

corpus = df['upath'].values
print len(corpus)

# <codecell>

X = vec.fit_transform(corpus)
print X.shape

# <codecell>

x = np.ravel(X.sum(axis=0))

fig, ax = plt.subplots(1,2, figsize=(14,6))

_bins = np.linspace(0,10000,40)
ax[0].hist(x, bins=_bins, histtype='stepfilled', alpha=0.75)
ax[0].set_xlim(0,2000) #<- NOTICE
ax[0].set_xlabel('Number of times each cable is used')
clean(ax[0])

ax[1].hist(x, bins=_bins, histtype='stepfilled', alpha=0.75)
ax[1].set_ylim(0,10) # <- NOTICE
ax[1].set_xlim(2000,4000) #<- NOTICE
ax[1].set_xlabel('Number of times each cable is used')
fig.suptitle('Distribution of the number of times each cable is used')
clean(ax[1])

fig.show()

# <markdowncell>

# NOTES:
# 
# - While most of the cables are used less than a few hundred times, some are used very often.
# - There are about ~200K jobs in this test.
# - If a cable is used 2K times, then it is used in about once in every 100 jobs

# <markdowncell>

# ##What is the number of unique cables used in each job?

# <codecell>

tmp = X > 0
x = np.ravel(tmp.sum(axis=1))

fig, ax = plt.subplots()
_bins = np.linspace(0,12,20)
ax.hist(x, bins=_bins, alpha=0.75, histtype='stepfilled')
clean(ax)
fig.show()

# <markdowncell>

# ## Failures and links
# 
# Are there some cables that are always used in jobs that fail? Or that pass?
# 
# Using a threshold of **2X**

# <codecell>

fail_ = np.ravel(X[(df.norm > 2).values,].sum(axis=0).astype(np.float))
pass_ = np.ravel(X[(df.norm < 2).values,].sum(axis=0).astype(np.float))
diff = fail_ - pass_
percent = (fail_)/(fail_ + pass_)

# <codecell>

index = np.argsort(percent)
x = np.arange(len(index))

fig, ax = plt.subplots()
ax.plot(x, percent[index], color="red", alpha=0.5, 
            linewidth=2, label='Percent of jobs that fail')
ax.plot(x, 1-percent[index], color="green", alpha=0.5, 
            linewidth=2, label='Percent of jobs that pass')

ax.set_title('The percent of time a cable was used \nin a job that failed or passed')
ax.set_xlabel('cable')
ax.set_ylabel('percent of jobs that fail/pass')
ax.legend(loc='lower center')
clean(ax)
plt.show()

# <markdowncell>

# Results:
# 
# - 1000 cables that were exclusively used in jobs that failed.  
# - About 1500 exclusively used in jobs that passed.
# - The remaining 5500 were between were used in jobs that failed between ~20% and ~60% of the time.
# 
# Notes:
# 
# - This is sensitive to the threshold.
# - Increasing the threshold moves the inflection point to the right.
# - Decreasing it moves it left

# <markdowncell>

# ### How many jobs would the ~1000 cables impact?

# <codecell>

index = np.argsort(percent)

# This is a matrix with only the ~1000 fail columns
X_fail = X[:,index[np.argmax(percent[index]):]]
print X_fail.shape

# Which index values are zero vs non-zero
# e.g. exclusive vs non-exclusive
tmp = X_fail.sum(axis=1) > 0
print 'Accounts for about ', tmp.sum(), 'jobs'
print 'percent: {0}%'.format((tmp.sum()/float(len(tmp))*100))

# <markdowncell>

# They would impact about 1019 jobs, or less than 1%.

# <markdowncell>

# ### What is the importance of each cable?

# <markdowncell>

# How many times was each device used?
# 
# - We know that some are used more than 2K to 10K times.
# - See histogram above.

# <codecell>

tmp = X > 0
x = np.ravel(tmp.sum(axis=0))
print x.shape
print 'total usage count', x.sum()
y = list(reversed(sorted(x)))[:300]

fig, ax = plt.subplots()
ax.hist(y, histtype='stepfilled', alpha=0.75)
ax.set_xlabel('Number of times each cable is used')
ax.set_title('The top 300 \n most frequently used cables')
clean(ax)
fig.show()

# <markdowncell>

# ### Number of `cables` per percent group

# <codecell>

df = pd.DataFrame({'percent': percent, 'used': x})
print df.shape
print 'total number of cable hits in test', df['used'].sum()

# <codecell>

buckets = np.linspace(0,1,11)
per_buckets = pd.cut(df['percent'], buckets)
tmp = df.groupby(per_buckets)['percent'].count()

ax = tmp.plot(kind='barh')
ax.set_title('Number of cables grouped by \n the percentage of failure')
ax.set_xlabel('Number of cables')
clean(ax)
plt.show()

# <markdowncell>

# NOTES:
# 
# - There are ~1500 cables in that fail most of the time `(0.90,1]`.
#     - But only account for a small number of jobs, < 1%
# 
# - Most of the other cables fall in the `(0.4, 0.6]` bucket range.
#     - They are used in tests that pass as often as they fail.

# <markdowncell>

# ### Number of times each cable is used as a function of percent failure.

# <codecell>

ax = df.groupby(per_buckets)['used'].sum().plot(kind='barh')
ax.set_title('Number of total uses per cable\n grouped by the percent of \n failure')
ax.set_xlabel('Total number of times a cable is used')
clean(ax)
plt.show()

# <markdowncell>

# NOTES:
# 
# - This eliminates (puts in perspective) the tail of the failures shown previously.
# - This also implies that any single cable occurs in
#     - jobs that sometimes pass
#     - and sometimes fail.
# - Failure is not just because of a single cable
#     - **Congestion**?
#     - **Combination of cables**?

# <markdowncell>

# ## Summary

# <markdowncell>

# The problem is non-separable.
# 
# - The features (cables) do not act independently impact the failure of tests.
# - There is an interaction in the cables that may account for some of the failure.

# <codecell>


