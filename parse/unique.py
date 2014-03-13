#!/usr/bin/env python

from __future__ import print_function

import os
import glob

import config
import pandas as pd


def get_nodes(f):
	df = pd.read_csv(f)
	#print(df.head())
	tmp = df.groupby('id').mean()
	#tmp.reset_index(inplace=True)
	#print(tmp.head())
	# print(len(tmp),f)
	# print( tmp.index.values[:5])
	return list(tmp.index.values)

def unique_nodes(x,y):

	data = list(set(x).union(set(y)))
	return data

def total_tests(x,y):
	# print(x, y)
	# print(type(x), type(y))
	try:
		return x+y
	except TypeError:
		return 0

files = glob.glob(os.path.join(config.data_path,'results*.csv'))

#map(print, files)
nodes = map(get_nodes, files)

len_nodes = map(len, nodes)
map(print, len_nodes)

unique_list = reduce(unique_nodes, nodes)

print('-'*30)
print('unique tests', len(unique_list))
print('total tests', reduce(total_tests, len_nodes))






