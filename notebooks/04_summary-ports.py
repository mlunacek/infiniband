# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Ports summary

# <codecell>

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# <codecell>

filename = '/lustre/janus_scratch/molu8455/infiniband/data/combined/all_ports.csv'
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

# ## How often is each port used?

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
ax[0].set_xlabel('Number of times each port is used')
clean(ax[0])

ax[1].hist(x, bins=_bins, histtype='stepfilled', alpha=0.75)
ax[1].set_ylim(0,35) # <- NOTICE
ax[1].set_xlim(2000,4000) #<- NOTICE
ax[1].set_xlabel('Number of times each port is used')
fig.suptitle('Distribution of the number of times each port is used')
clean(ax[1])

fig.show()

# <markdowncell>

# NOTES:
# 
# - While most of the cables are used less than a few hundred times, some are used very often.
# - There are about ~200K jobs in this test.
# - If a cable is used 2K times, then it is used in about once in every 100 jobs

# <markdowncell>

# ##What is the number of ports used in each job?

# <codecell>

tmp = X > 0
x = np.ravel(tmp.sum(axis=1))

fig, ax = plt.subplots()
_bins = np.linspace(0,40,20)
ax.hist(x, bins=_bins, alpha=0.75, histtype='stepfilled')
clean(ax)
fig.show()

# <markdowncell>

# ## Failures and ports
# 
# Are there some ports that are always used in jobs that fail? Or that pass?

# <codecell>

fail_ = np.ravel(X[(df.norm > 2.).values,].sum(axis=0).astype(np.float))
pass_ = np.ravel(X[(df.norm < 2.).values,].sum(axis=0).astype(np.float))
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
ax.legend(loc='upper left')
clean(ax)
plt.show()

# <markdowncell>

# ### What is the importance of each port?

# <markdowncell>

# How many times was each port used?
# 
# - We know that some are used more than 2K times.
# - See histogram above.

# <codecell>

tmp = X > 0
x = np.ravel(tmp.sum(axis=0))
print x.shape
print 'total usage count', x.sum()
y = list(reversed(sorted(x)))[:300]

fig, ax = plt.subplots()
ax.hist(y, histtype='stepfilled', alpha=0.75)
ax.set_xlabel('Number of times each port is used')
ax.set_title('The top 300 \n most frequently used ports')
clean(ax)
fig.show()

# <markdowncell>

# ### Number of `ports` per percent group

# <codecell>

df = pd.DataFrame({'percent': percent, 'used': x})
print df.shape
print 'total number of cable hits in test', df['used'].sum()

# <codecell>

buckets = np.linspace(0,1,11)
per_buckets = pd.cut(df['percent'], buckets)
tmp = df.groupby(per_buckets)['percent'].count()

ax = tmp.plot(kind='barh')
ax.set_title('Number of ports grouped by \n the percentage of failure')
ax.set_xlabel('Number of ports')
clean(ax)
plt.show()

# <markdowncell>

# NOTES:
# 
# - Most of the other ports fall in the `(0.4, 0.7]` bucket range.
#     - They are used in tests that pass more often than they fail.

# <codecell>

print tmp

# <markdowncell>

# ### Number of times each port is used as a function of percent failure.

# <codecell>

ax = df.groupby(per_buckets)['used'].sum().plot(kind='barh')
ax.set_title('Number of total uses per port\n grouped by the percent of \n failure')
ax.set_xlabel('Total number of times a port is used')
clean(ax)
plt.show()

# <codecell>


