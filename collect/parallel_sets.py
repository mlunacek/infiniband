#!/usr/bin/env python
"""
Generates a list of sets that can be run in parallel.
Does not scale:

In [63]: run parallel_sets.py 150
number of nodes 150
number of tests 11175
single 0.0729038715363

In [58]: run parallel_sets.py 300
number of nodes 300
number of tests 44850
single 0.533872842789

In [60]: run parallel_sets.py 600
number of nodes 600
number of tests 179700
single 4.20527100563

In [61]: run parallel_sets.py 1200
number of nodes 1200
number of tests 719400
single 34.6860539913

"""


from __future__ import print_function

from random import shuffle
import itertools
import sys
import numpy as np
from time import time

def random_node_list(df, size=None):
	node_list = list(set(df['one'].values).union(set(df['two'])))
	shuffle(node_list)
	if size:
		return node_list[:size]
	return node_list

def expand_grid(*itrs):
   product = list(itertools.product(*itrs))
   return [ x for x in product if x[0] < x[1]]

class ParallelSets:

	def __init__(self, node_list):
		self.node_list = node_list
		self.N = len(node_list)
		self.A = np.tril(np.ones((self.N, self.N)))
		self.max_set = (self.N/2)+1

	def __del__(self):
		pass

	def exclude_index(self, x):
		try:
			self.A[x,] = 1
			self.A[:,x] = 1
		except IndexError:
			pass

	def exclude_node(self, x):
		try:
			y = self.node_list.index(x)
			self.A[y,] = 1
			self.A[:,y] = 1
		except ValueError:
			pass

	def exclude(self, x):
		if isinstance(x, int):
			self.exclude_index(x)
		else: #hope it's a node
			self.exclude_node(x)

	def finished(self):
		if self.A.sum() < self.A.size:
			return False
		return True	

	def all_sets_generator(self, node_names=False):
		self.A = np.tril(np.ones((self.N,self.N)))
		while not self.finished():
			yield self.single_index_set(node_names)

	def single_set(self):

		data = []
		tmp = self.single_index_set()
		for d in tmp:
			data.append((self.node_list[d[0]], self.node_list[d[1]]))
		return data


	def single_index_set(self, node_names=False):
		data = []
		B = np.zeros((self.N,self.N))
		#print(self.A)
		for i in range(self.max_set):

			C = self.A+B
			available = np.argwhere(C == 0) 

			if available.shape[0] > 0:

				index = np.random.random_integers(0,len(available)-1, size=1)[0]
				row = available[index][0]
				col = available[index][1]
				
				#This test is added to the list
				self.A[row, col] = 1 

				# But don't pick the node again
				B[row,] = 1
				B[:,col] = 1
				B[col,] = 1
				B[:,row] = 1

				assert available[index][0] < available[index][1]
				if node_names == True:
					data.append((self.node_list[row],self.node_list[col]))
				else:
					data.append((row,col))
			# we are done
			else: 
				return data

		return data

if __name__ == '__main__':
	
	N = int(sys.argv[1])

	# df = pd.read_csv('all_pairs_list.csv')
	# node_list = random_node_list(df,N)
	node_list = ['n{0}'.format(x) for x in range(N)]

	print('number of nodes', len(node_list))
	print('number of tests', (N*N-N)/2)

	# Example using the all_sets_generator()
	tic = time()
	set_gen = ParallelSets(node_list)
	data = set_gen.all_sets_generator()
	first = data.next()
	print('single', time()-tic)
	total = map(len, data) #consumes the generator
	print('rest...', time()-tic)
	
	# But we might need to exclude nodes
	data = set_gen.all_sets_generator(node_names=True)
	first = data.next()
	print(first)
	map(set_gen.exclude, [2,3,100])
	map(set_gen.exclude, ['n0','n1','xnode'])
	print('------')
	for d in data:
		print(d)
	
	# #map(print, total)
	# print('total', reduce(lambda x,y: x+y, total)+len(first))






