#!/usr/bin/env python

import os
import re
import sys
import numpy as np
import pandas as pd

from functools import partial
import config

def parse_results(tmp):
	values = re.findall(r'262144[ ]+([0-9.]+)', tmp)
	if len(values) > 0:
		res = np.array(values, dtype='float')
		ave = res.mean()
		return ave
	else:
		return None


def parse_parts(parts):
	path = []
	for i, p in enumerate(parts):
		port_in = None
		port_out = None
		device = None
		res = re.findall(r'{0x([0-9a-h]+)}.*(node[0-9]{4} HCA-1)', p)
		if len(res)>0:
			if p.find('ca port')<0:
				device = res[0][0]
				port_in = '1'
				port_out = '1'
		res = re.findall(r'{0x([0-9a-h]+)}.*\[([0-9]{,2})\].*\[([0-9]{,2})\]', p)
		if len(res)>0:
			device = res[0][0]
			port_in = res[0][1]
			port_out = res[0][2]
		path.append((i, port_in, device, port_out))

	return path


def parse_path(tmp):
	parts = []
	tmp = tmp.replace('\n','')
	for x in re.split('From ca', tmp):
		for y in re.split('->|To ca', x):
			parts.append(y)
	path = parse_parts(parts)
	return path


def parse_info(tmp):
	values = re.findall(r'(node[0-9]{4})', tmp)
	return '-'.join(values)


def parse_test(test):

	comp = re.split(r'BEFORE|RESULTS|AFTER', test)

	try:
		j_info = parse_info(comp[0])
		b_path = parse_path(comp[1])
		res = parse_results(comp[2])
		a_path = parse_path(comp[3])
		same = b_path == a_path

		return {'info': j_info, 
				'bp': b_path, 
				'ap': a_path, 
				'res': res, 
				'same': same}
	except IndexError:
		return None


def append_results(data, r):
	count = 0
	if r is not None:
		for b in r['bp']:
			if b[1] is not None:
				data['id'].append(r['info'])
				data['res'].append(r['res'])
				data['same'].append(r['same'])
				data['order'].append(count)
				data['pin'].append(b[1])
				data['device'].append(b[2])
				data['pout'].append(b[3])
				count += 1
	return count


def parse_file(filename):
	with open(filename, 'r') as infile:
		data = infile.read()

	# parse the tests
	tests = data.split('TEST_START')
	results = map(parse_test, tests)

	# Convert to dictionary
	data = {'id': [], 'res': [], 'same': [], 'order': [],
		    'pin': [], 'device': [], 'pout': []}

	convert_dataframe = partial(append_results, data)
	map(convert_dataframe, results)
	
	tmp = pd.DataFrame(data)
	tmp = tmp[['id','res','same','order','device','pin','pout']]

	base = os.path.basename(filename)+'.csv'
	csv_file = os.path.join(config.data_path,base)
	tmp.to_csv(csv_file, index=False)


if __name__ == '__main__':
	
	# open the file
	if len(sys.argv) < 2:
		print 'please specify the path of the file to parse\n'
		print 'e.g. $python '+ sys.argv[0]+' ../collect/test-*\n'
		sys.exit(1)

	filename = sys.argv[1]
	parse_file(filename)












