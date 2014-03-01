#!/usr/bin/env python

import os
import re
import pandas as pd

ib_dir = '/lustre/janus_scratch/molu8455/infiniband/'
ib_file = 'ibnetdiscover-2-12-2014'

filename = os.path.join(ib_dir, ib_file)

with open(filename,'r') as infile:
	data = infile.read()

values = re.findall(r'(node[0-9]{4}).*lid ([0-9]+)', data)

tmp = pd.DataFrame(values)
tmp.columns = ['node','lid']

# Are the columns unique?
print tmp['node'].nunique()
print tmp['lid'].nunique()

# Get rid of duplicates
tmp.drop_duplicates('node', inplace=True)
tmp.drop_duplicates('lid', inplace=True)

print tmp['node'].nunique()
print tmp['lid'].nunique()

tmp.to_csv('node_lid.csv', index=False)


