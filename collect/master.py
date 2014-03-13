#!/usr/bin/env python

import subprocess
import re
import jinja2 as jin
import random
import time
import os
import datetime
import dateutil.parser
import json

def date_handler(obj):
    return obj.isoformat() if hasattr(obj, 'isoformat') else obj
 
template2 = jin.Template('''
#!/bin/bash
#PBS -N job.IB
#PBS -q janus-admin
#PBS -l walltime=01:00:00
#PBS -l nodes={% for x in node_list %}{{x}}:ppn=12{%- if not loop.last -%}+{%- endif %}{% endfor %}
#PBS -j oe

cd $PBS_O_WORKDIR

module load python/anaconda-1.6.1 

mkdir -p 'test-'$PBS_JOBID
cd 'test-'$PBS_JOBID

cp ../jobs.py .
cp ../parallel_sets.py .

python jobs.py 20

rm jobs.py
rm parallel_sets.py 

''')

template = jin.Template('''
#!/bin/bash
#PBS -N job.IB
#PBS -q janus-admin
#PBS -l walltime=01:00:00
#PBS -l nodes={{nodes}}:ppn=12
#PBS -j oe

cd $PBS_O_WORKDIR

module load python/anaconda-1.6.1 

mkdir -p 'test-'$PBS_JOBID
cd 'test-'$PBS_JOBID

cp ../jobs.py .
cp ../parallel_sets.py .

python jobs.py 20

rm jobs.py
rm parallel_sets.py 

''')






# Which nodes are available?
# TODO deal with hanging
def command(cmd):
	cmd = 'timeout 10 ' + cmd
	#print cmd
	pid = subprocess.Popen(cmd, shell=True, 
								stdout=subprocess.PIPE,
								stderr=subprocess.PIPE)
	pid.wait()
	output, error = pid.communicate()
	return output.split('\n')

def reserved_nodes():
	if os.path.exists('reserved_nodes'):
		tmp = open('reserved_nodes').read().split('\n')
		tmp = re.findall(r'node[0-9]{4}', ''.join(tmp))
		print 'reserved:       ', len(tmp)
		return tmp
	else:
		cmd = "showres | cut -d' ' -f1"
		out = command(cmd)
		res = [ x for x in out if not x.startswith('PM') and x.find('.') > 0]

		tmp = []
		for r in res:
			if not r.startswith('wide') and not r.startswith('small'):
				cmd = "showres -n -g " + r + " | cut -d' ' -f1"
				out = command(cmd)
				nodes = re.findall(r'node[0-9]{4}', ''.join(out))
				for n in nodes:
					tmp.append(n)

		print 'reserved:       ', len(tmp)
		with open('reserved_nodes','w') as outfile:
			for n in tmp:
				outfile.write(n+'\n')

		return tmp

def parse_node_status(output):
	down = 0
	job_exclusive = 0
	free = 0
	free_list = []
	for lines in output:
		try:
			node, status = lines.split()
			if status == 'free' and node.startswith('node'):
				if node[4:6] != '11':
					free_list.append(node)
					free += 1
			elif status == 'job-exclusive':
				job_exclusive+=1
			elif status == 'down':
				down+=1
		except ValueError:
			pass

	print 'jobs_exclusive: ', job_exclusive
	print 'free:           ', free
	print 'down:           ', down
	return free_list

def free_nodes():
	cmd = 'pbsnodes -l all'
	output = command(cmd)
	free_list = parse_node_status(output)
	res_list = reserved_nodes()
	return list(set(free_list).difference(set(res_list)))

# Create jobs scripts
def create_jobs_scripts(node_list):
	random.shuffle(node_list)

	one = node_list[::2]
	two = node_list[1::2]

	print 'node list one:', len(one)
	print 'node list two:', len(two)

	l1 = len(one)-20
	l2 = len(two)-20

	with open('submit_one.pbs','w') as output:
		output.write(template.render({'nodes': l1 }))
	with open('submit_two.pbs','w') as output:
		output.write(template.render({'nodes': l2 }))

# Submit jobs
def submit_jobs(myjobs):
	
	print 'submit_jobs'
	cmd = 'qsub submit_one.pbs'
	x = command(cmd)
	print x
	jobid = re.findall(r'([0-9]{7})', x[0])[0]
	myjobs[jobid] = datetime.datetime.now()
	
	cmd = 'qsub submit_two.pbs'
	x = command(cmd)
	print x
	jobid = re.findall(r'([0-9]{7})', x[0])[0]
	myjobs[jobid] = datetime.datetime.now()

	# write jobs to file
	tmp = json.dumps(myjobs, default=date_handler)
	with open('jobs.json','w') as outfile:
		outfile.write(tmp)

	return myjobs

# Wait for jobs to finish
def jobs_running(myjobs):
	cmd ="showq | grep molu"
	out = command(cmd)
	tmp = re.findall(r'([0-9]{7}).*molu8455[ ]+([R|I|D])','\n'.join(out))
	for t in tmp:
		try:
			print t[0], t[1]
			# print myjobs[t]
			c = datetime.datetime.now() - myjobs[t[0]]
			print c.seconds
			if c.seconds > 300 and t[1] != 'R':
				# kill the job
				cmd = 'qdel '+ t[0]
				print cmd
				_ = command(cmd)
				print _

		except KeyError:
			pass
	return len(tmp)

myjobs = {}
if os.path.exists('jobs.json'):
	tmp = open('jobs.json','r').read()
	tmp = json.loads(tmp)
	for k in tmp:
		tmp[k] = dateutil.parser.parse(tmp[k])
	myjobs = tmp

# tic = time.time()
# while time.time() - tic < 600:
	
jobs = jobs_running(myjobs)
print 'jobs running:   ', jobs

if jobs == 1:

	node_list = free_nodes()
	print 'available:      ', len(node_list)

	create_jobs_scripts(node_list)
	myjobs = submit_jobs(myjobs)
	# print myjobs
	
# time.sleep(10) #adjust this




