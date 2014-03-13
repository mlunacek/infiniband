# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import glob
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt

# <codecell>

def combine_results(prefix):
    
    d_path = '/lustre/janus_scratch/molu8455/infiniband/data/parsed'
    o_path = '/lustre/janus_scratch/molu8455/infiniband/data/combined'
    
    files = glob.glob(os.path.join(d_path, prefix + '*'))
    frames = map(pd.read_csv, files)
    df = pd.concat(frames, ignore_index=True)
    print df.shape
    
    df.columns = ['unnamed','id','res','same','path']
    df.drop('unnamed', axis=1, inplace=True)
    print df.shape
    print 'number of null values', df.isnull().sum()
    #print 'number of null values', df.isna().sum()
    
    d_path = o_path
    filename = os.path.join(d_path,'all_' + prefix + '.csv')
    df.dropna(inplace=True)
    print df.shape
    
    # Unique elements too...
    df['upath'] = df['path'].map(lambda x: ' '.join(list(set(x.split()))))
    
    df.to_csv(filename, index=False)

# <codecell>

combine_results('cables')

# <codecell>

combine_results('ports')

# <codecell>


