#!/usr/bin/env python

import sys
import os
import json

import loadbalance
import learn
from random import shuffle

if __name__ == "__main__":

	job_id = os.environ['PBS_JOBID'].split('.')[0]
	filename = 'results.' + os.environ['PBS_JOBID'] + '.json'

	tests = learn.create_tests()
	shuffle(tests)
	
	lb = loadbalance.LoadBalance(ppn=4)
	ar = lb.lview.map(learn.evaluate, tests)
	
	best_so_far = {}
	best_so_far['roc'] = 0
	best_so_far['str'] = ''

	print 'number of tests', len(tests)
	with open(filename, 'w') as outfile:
		for i,r in enumerate(ar):
			print i, r['f1_train'],r['f1_test'],r['roc_area']
			if r['f1_test'] > 0.9:
				print '-----------------------------------'
			if best_so_far['roc'] < r['roc_area']:
				best_so_far['roc'] = r['roc_area']
				best_so_far['str'] = str(r['f1_train']) + ' ' + str(r['f1_test']) + ' ' + str(r['roc_area'])
			sys.stdout.flush()
			outfile.write(json.dumps(r)+'\n')
			outfile.flush()

			if i%100==0:
				print '============================'
				print i, best_so_far['str'] 
				print '============================'




