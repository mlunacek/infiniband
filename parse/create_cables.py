#!/usr/bin/env python

import pandas as pd


filenames = 'raw_data/results.2522001.csv'
tmp = pd.read_csv(filenames)

def cables(df):


	print '-'*80
	print df
	order = df['order'].values
	dev = df['device'].values
	pin = df['pin'].values
	pout = df['pout'].values 

	# start_node = dev[0]+':{num:02d}'.format(num=pout[0])
	# start_node += ':{num:02d}'.format(num=pin[1])+ ':' +dev[1]
	# print start_node

	# for index, next in zip(order[:-1], order[1:]):
	# 	p1 = ':{num:02d}'.format(num=pin[index])
	# 	p2 = ':{num:02d}'.format(num=pout[index])
	# 	p3 = ':{num:02d}'.format(num=pin[next])
	# 	p4 = ':{num:02d}'.format(num=pout[next])
	# 	print p1, dev[index], p2, '->', p3, dev[next], p4

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

res = tmp.ix[:30].groupby('id').apply(cables)








