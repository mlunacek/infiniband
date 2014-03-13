# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Summary

# <codecell>

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# <markdowncell>

# Either should be fine.

# <codecell>

filename = '/lustre/janus_scratch/molu8455/infiniband/data/combined/all_ports.csv'
#filename = '/lustre/janus_scratch/molu8455/infiniband/data/combined/all_cables.csv'
df = pd.read_csv(filename)

# <codecell>

print df.shape
print df.columns
df.dropna(inplace=True)
print df.shape

# <markdowncell>

# Clean up figures...

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

# ## Distribution of performance

# <codecell>

from scipy import stats

_bins = np.linspace(0,20,50)

# Look at relative performance
df['norm'] = df.res.values/np.min(df.res.values)
norm = df['norm'].values

x = np.linspace(0, 12, 200) #NOTICE cutoff
density = stats.kde.gaussian_kde(norm)

threshold = 2
good = norm[norm<threshold]
bad = norm[norm>threshold]

fig, ax = plt.subplots()
ax.fill_between(x, density(x), alpha=0.75)
ax.set_xlim(0,10) #<-- NOTICE the limit
ax.set_xlabel('alltoall bandwidth decrease amount (e.g. 2X, 6X slower)')
ax.vlines(2, 1e-3, density(2.0), color='grey')
ax.set_title('Distribution of relative bandwith performance in alltoall')
clean(ax)
plt.show()

# <markdowncell>

# ## How many path changes occur?

# <codecell>

# Look at relative performance
same_path = df[df['same'] == True]['norm'].values
diff_path = df[df['same'] == False]['norm'].values

per = np.round(len(diff_path)/float(len(same_path)+len(diff_path))*100)

_bins = np.linspace(0,20,100)

x = np.linspace(0, 12, 200) #NOTICE cutoff
density_same = stats.kde.gaussian_kde(same_path)
density_diff = stats.kde.gaussian_kde(diff_path)

fig, ax = plt.subplots()
ax.hist(same_path, bins=_bins, histtype='stepfilled', alpha=0.75, label='No change')
ax.hist(diff_path, bins=_bins, histtype='stepfilled', alpha=0.75, label='Change in path')
ax.set_xlim(0,10) #<-- NOTICE the limit
ax.set_xlabel('alltoall bandwidth decrease amount (e.g. 2X, 6X slower)')
ax.set_title('~ {0} percent of jobs change paths'.format(per))
ax.vlines(2, 1e-3, density(2.0), color='grey')
ax.legend()
clean(ax)
plt.show()

# <markdowncell>

# NOTES:
# 
# - Similar distribution.

# <markdowncell>

# ## How many tests pass and fail?

# <codecell>

def percent_fail(thresh):
    return np.sum(norm>thresh)/float(len(norm))*100

threshold = np.linspace(1,10,20)
res = map(percent_fail, threshold)

# <codecell>

fig, ax = plt.subplots()
ax.plot(threshold, res, '-o')
clean(ax)
fig.show()

# <markdowncell>

# About half of the tests are more than 2x slower than the best.

# <codecell>


