#!/usr/bin/env python

import os
import sys
import subprocess
import pandas as pd
from functools import partial
import time

import parallel_sets as ps

# don't run on host
def get_node_list():
	nodes = open(os.environ['PBS_NODEFILE']).read().split('\n')
	node_list = [ x for x in nodes if len(x) == 8]
	hostname = os.environ['HOSTNAME']
	node_list = [ x for x in node_list if x != hostname]
	return list(set(node_list))

def get_node_lid():
	df = pd.read_csv('node_information/node_lid.csv')
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


def create_mpi_command(nodes):
	cmd0 = 'echo {0} {1} {2} {3};'
	cmd0 = cmd0.format(nodes[0],nodes[1], nodes[2], nodes[3])
	cmd1 = 'echo "BEFORE";ibtracert {0} {1};'
	cmd1 = cmd1.format(nodes[1], nodes[3])
	cmd2 = 'ibtracert {0} {1};'.format(nodes[3], nodes[1])
	cmd3 = 'echo "RESULTS"; mpirun --host {0},{1} /home/molu8455/admin/benchmarks/software/mpi/osu_alltoall  -f;'
	cmd3 = cmd3.format(nodes[0], nodes[2])
	cmd4 = 'echo "AFTER"; ibtracert {0} {1};'.format(nodes[1], nodes[3])
	cmd5 = 'ibtracert {0} {1};'.format(nodes[3], nodes[1])
	cmd = cmd0 + ' '+ cmd1+' '+cmd2+''+cmd3+' '+cmd4+''+cmd5
	return cmd

def join_node_lid(lid_list, node):
	return (node[0], lid_list[node[0]], node[1], lid_list[node[1]])


if __name__ == '__main__':
	
	node_list = get_node_list()
	lid_list = get_node_lid()
	node_list = remove_nolid_nodes(node_list, lid_list)

	set_gen = ps.SetGenerator(node_list)

	filename = 'results.' + os.environ['PBS_JOBID'].split('.')[0]

	with open(filename,'w') as outfile:
		while not set_gen.finished():
			tic = time.time()
			tests = set_gen.single_set()

			add_lids = partial(join_node_lid, lid_list)
			tests_lid = map(add_lids, tests)

			cmds = map(create_mpi_command, tests_lid)

			pids = map(execute_cmd, cmds)

			output = map(wait_cmd, pids)

			print len(cmds), time.time()-tic
			sys.stdout.flush()

			for out in output:
				outfile.write('TEST_START\n')
				outfile.write(out+'\n')
				outfile.write('TEST_END\n')

			outfile.flush()











