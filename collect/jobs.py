#!/usr/bin/env python

import os
import sys
import subprocess
import pandas as pd
from functools import partial
import time

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
							stderr=subprocess.PIPE, )
	return pid


def wait_cmd(pid):
	pid.wait()
	output, stderr = pid.communicate()
	return output


# This is a bit ugly at the moment.
def create_mpi_command(nodes):
	
	data = {'n1': nodes[0], 
			'n2': nodes[2],
			'l1': nodes[1],
			'l2': nodes[3] }

	#print **data
	cmds = []

	cmds.append('echo {n1} {l1} {n2} {l2}')
	cmds.append('echo "BEFORE"')
	cmds.append('ibtracert {l1} {l2}')
	cmds.append('ibtracert {l2} {l1}')
	cmds.append('echo "RESULTS"')
	cmds.append('mpirun --host {n1},{n2} '+ alltoall_cmd)
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


if __name__ == '__main__':
	
	node_list = get_node_list()
	lid_list = get_node_lid()
	node_list = remove_nolid_nodes(node_list, lid_list)

	set_gen = ps.SetGenerator(node_list)

	filename = 'results.' + os.environ['PBS_JOBID'].split('.')[0]

	with open(filename,'w') as outfile:
		while not set_gen.finished():
			# keep track of how long each session takes
			tic = time.time()

			tests = set_gen.single_set()
			add_lids = partial(join_node_lid, lid_list)
			tests_lid = map(add_lids, tests)
			cmds = map(create_mpi_command, tests_lid)

			pids = map(execute_cmd, cmds)
			output = map(wait_cmd, pids)

			print len(cmds), time.time()-tic
			sys.stdout.flush()

			# Write output to file after each iteration
			# ~ one minute
			for out in output:
				outfile.write('TEST_START\n')
				outfile.write(out+'\n')
				outfile.write('TEST_END\n')

			outfile.flush()











