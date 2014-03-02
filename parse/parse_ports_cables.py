#!/usr/bin/env python

import os
import sys
import pandas as pd
from functools import partial

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

def cables(df, switches=False):
	"""
	Creates a list of cables, which are device:port to device:port 
	links, sorted so that A:B is the same as B:A.
	"""
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

	cables = []
	for index, l in enumerate(links[1:]):
		# if same device
		#print devlinks[index-1], devlinks[index], devlinks[index+1]

		if devlinks[index] == devlinks[index+1]:
			if switches:
				cables.append(devlinks[index+1])
		else:
			if links[index] < links[index+1]:
				cables.append(links[index] + '-' + links[index+1])
			else:
				cables.append(links[index+1] + '-' + links[index])

	return ' '.join(cables)

def check_frame(res):
	# make sure it all makes sense
	r2 = res.reset_index()
	assert r2.shape[0] == r2['id'].groupby(r2['id']).count().shape[0]


def parse_ports(filename):
	job_id = os.path.basename(filename).split('.')[1]
	outfile = os.path.join(config.data_path, 'ports.'+job_id+'.csv')

	tmp = pd.read_csv(filename)
	res = tmp.groupby(['id','res','same']).apply(ports)

	# make sure it all makes sense
	check_frame(res)
	res = res.reset_index()
	res.to_csv(outfile, index=True)

def parse_cables(filename, switches=False):

	job_id = os.path.basename(filename).split('.')[1]
	outfile = os.path.join(config.data_path, 'cables.'+job_id+'.csv')
	if switches:
		outfile = os.path.join(config.data_path, 'switches.'+job_id+'.csv')

	tmp = pd.read_csv(filename)
	partial_cables = partial(cables, switches=switches)

	res = tmp.groupby(['id','res','same']).apply(partial_cables)
	check_frame(res)
	res = res.reset_index()
	res.to_csv(outfile, index=True)


if __name__ == '__main__':
	
	# open the file
	if len(sys.argv) < 2:
		print 'please specify the path of the file to parse\n'
		print 'e.g. $python '+ sys.argv[0]+' results.*.csv\n'
		sys.exit(1)

	filename = os.path.join(config.data_path, sys.argv[1])

	parse_ports(filename)
	parse_cables(filename)
	parse_cables(filename, True)






