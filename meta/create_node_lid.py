#!/usr/bin/env python

import re
import sys
import pandas as pd

if len(sys.argv) < 2:
	print 'please specify the ibnetdiscover-* file.\n'
	print 'e.g. $python '+ sys.argv[0]+' ibnetdiscover-<date>\n'
	sys.exit(1)

filename = sys.argv[1]

with open(filename,'r') as infile:
	data = infile.read()

values = re.findall(r'(node[0-9]{4}).*lid ([0-9]+)', data)

tmp = pd.DataFrame(values)
tmp.columns = ['node','lid']

# Get rid of duplicates
tmp.drop_duplicates('node', inplace=True)
tmp.drop_duplicates('lid', inplace=True)

print 'There are {0} nodes with lids.'.format(len(tmp))

tmp.to_csv('node_lid_relationship.csv', index=False)
tmp.T.to_json('node_lid_relationship.json')
