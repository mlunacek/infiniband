#!/usr/bin/env python

import itertools
import pandas as pd

def expand_grid(*itrs):
   product = list(itertools.product(*itrs))
   return [ x for x in product if x[0] < x[1]]



node_list = []
for rack in range(1,18):
	for comp in range(1,81):
		node_list.append('node{r:02d}{c:02d}'.format(r=rack, c=comp))

print len(node_list)

all_pairs = expand_grid(node_list,node_list)

col1 = [ x[0] for x in all_pairs]
col2 = [ x[1] for x in all_pairs]

tmp = pd.DataFrame({'one': col1, 'two': col2 })
tmp['id'] = tmp['one'] + '-' + tmp['two']

tmp.to_csv('all_pairs_list.csv', index=False)


