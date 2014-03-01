#!/usr/bin/env python

import os
import pandas as pd

import config

def split_columns(df):

	order = df['order'].values
	dev = df['device'].values
	pin = df['pin'].values
	pout = df['pout'].values
	return order, dev, pin, pout

def remove_duplicate_neighbors(in_path):
	out_path = []
	out_path.append(in_path[0])
	for index, p in enumerate(in_path[1:]):
		if out_path[-1] != p:
			out_path.append(p)
	return out_path

def ports(df):
	"""
	This method creates a list (sentence) of ports.
	It duplicates the start node but not the end node.
	This may be an issue.
	"""
	path = []
	#idea create a dev:port for each link	
	order, dev, pin, pout = split_columns(df)
	
	for index in order:
		p1 = ':{num:02d}'.format(num=pin[index])
		p2 = ':{num:02d}'.format(num=pout[index])
		path.append(dev[index]+p1)
		if p1 != p2:
			path.append(dev[index]+p2)

	path = remove_duplicate_neighbors(path)
	return ' '.join(path)

def cables(df):


	print '-'*80
	print df
	order = df['order'].values
	dev = df['device'].values
	pin = df['pin'].values
	pout = df['pout'].values 

	links = []
	devlinks = []
	for index in order:
		p1 = ':{num:02d}'.format(num=pin[index])
		p2 = ':{num:02d}'.format(num=pout[index])
		if p1 == p2:
			links.append(dev[index]+p1)
			devlinks.append(dev[index])
		else:
			links.append(dev[index]+p1)
			links.append(dev[index]+p2)
			devlinks.append(dev[index])
			devlinks.append(dev[index])


	for index, l in enumerate(links[1:]):
		# if same device
		if devlinks[index-1] == devlinks[index]:
			print links[index], links[index+1]
		# else.. not same device
		# else:
		# 	print l, links[index-1], links[index], links[index+1]

	return None

def parse_ports(filename):
	job_id = os.path.basename(filename).split('.')[1]
	outfile = os.path.join(config.data_path, 'ports.'+job_id+'.csv')

	tmp = pd.read_csv(filename)
	res = tmp.groupby(['id','res','same']).apply(ports)

	# make sure it all makes sense
	r2 = res.reset_index()
	assert r2.shape[0] == r2['id'].groupby(r2['id']).count().shape[0]

	res.to_csv(outfile, index=True)



filename = os.path.join(config.data_path, 'results.2522001.csv')

#parse_ports(filename)

tmp = pd.read_csv(filename)
res = tmp.ix[:30].groupby(['id','res','same']).apply(cables)

print res






