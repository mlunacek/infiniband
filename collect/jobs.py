#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import subprocess
import pandas as pd
from functools import partial
from time import time

import parallel_sets as ps

meta_dir = '/lustre/janus_scratch/molu8455/infiniband/meta'
meta_file = 'node_lid_relationship.csv' # may change to json
alltoall_cmd = '/home/molu8455/admin/benchmarks/software/mpi/osu_alltoall  -f'


# don't run on host
def get_node_list():
	nodes = open(os.environ['PBS_NODEFILE']).read().split('\n')
	node_list = [ x for x in nodes if len(x) == 8]
	hostname = os.environ['HOSTNAME']
	node_list = [ x for x in node_list if x != hostname]
	return list(set(node_list))


def get_node_lid():
	df = pd.read_csv(os.path.join(meta_dir,meta_file))
	return dict(zip(df['node'], df['lid']))


def remove_nolid_nodes(node_list, lid_list):
	return [ x for x in node_list if x in lid_list.keys()]


def execute_cmd(cmd):

	pid = subprocess.Popen(cmd, shell=True, 
							stdout=subprocess.PIPE,
							stderr=subprocess.PIPE)
	return (pid, time())

def wait_cmd(data):
	pid = data[0]
	# while pid.poll() and time()-start_time < 60:
	# 	sleep(5)
	pid.wait()
	output, stderr = pid.communicate()
	return (output, time()-data[1])


# This is a bit ugly at the moment.
def create_mpi_command(nodes):
	
	data = {'n1': nodes[0], 
			'n2': nodes[2],
			'l1': nodes[1],
			'l2': nodes[3] }

	data['timeout'] = 120

	cmds = []
	cmds.append('echo {n1} {l1} {n2} {l2}')
	cmds.append('echo "BEFORE"')
	cmds.append('ibtracert {l1} {l2}')
	# cmds.append('echo ""')
	cmds.append('ibtracert {l2} {l1}')
	cmds.append('echo "RESULTS"')
	cmds.append('timeout {timeout} mpirun --host {n1},{n2} ')
	cmds[-1] += alltoall_cmd
	cmds.append('echo "AFTER"')
	cmds.append('ibtracert {l1} {l2}')
	cmds.append('ibtracert {l2} {l1}')

	# format
	data_format = partial(str.format, **data)
	cmds = map(data_format, cmds)

	new_cmd = ';'.join(cmds)
	return new_cmd

def join_node_lid(lid_list, node):
	return (node[0], lid_list[node[0]], 
		    node[1], lid_list[node[1]])

def write_output(outfile, out):
	outfile.write('TEST_START\n')
	outfile.write(str(out[1])+'\n')
	outfile.write(out[0]+'\n')
	outfile.write('TEST_END\n')

def pbsnodes_down():
	cmd = 'timeout 20 pbsnodes -l | cut -d" " -f1'
	pid = subprocess.Popen(cmd, shell=True, 
						stdout=subprocess.PIPE,
						stderr=subprocess.PIPE)
	pid.wait()
	out, err = pid.communicate()
	return [ x for x in out.split('\n') if len(x) == 8]

if __name__ == '__main__':
	
	node_list = get_node_list()
	lid_list = get_node_lid()
	node_list = remove_nolid_nodes(node_list, lid_list)
	add_lids = partial(join_node_lid, lid_list)

	set_gen = ps.ParallelSets(node_list)

	filename = 'results.' + os.environ['PBS_JOBID'].split('.')[0]

	num_jobs = 10
	try:
		num_jobs = int(sys.argv[1])
	except:
		pass


	with open(filename,'w') as outfile:
		write_partial = partial(write_output, outfile)
		all_sets = set_gen.all_sets_generator(node_names=True)
		
		count = 1
		tic = time()
		for tests in all_sets:
			# keep track of how long each session takes
			toc = time()
			gen_time = toc-tic

			tests_lid = map(add_lids, tests)
			cmds = map(create_mpi_command, tests_lid)
			pids = map(execute_cmd, cmds)
			output = map(wait_cmd, pids)

			#map( lambda x: print(x[1]), output)
			print('status', count, len(output), time()-toc, gen_time)
			#map(print, cmds)
			sys.stdout.flush()

			# Write output to file after each iteration
			# ~ one minute
			map(write_partial, output)
			outfile.flush()

			# Exlude down nodes on next set
			down_nodes = pbsnodes_down()
			set_gen.exclude(down_nodes)

			tic = time()
			count += 1
			if count > num_jobs:
				break







