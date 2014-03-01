#!/usr/bin/env python

from random import shuffle
import itertools
import sys
import numpy as np

def random_node_list(df, size=None):
	node_list = list(set(df['one'].values).union(set(df['two'])))
	shuffle(node_list)
	if size:
		return node_list[:size]
	return node_list

def expand_grid(*itrs):
   product = list(itertools.product(*itrs))
   return [ x for x in product if x[0] < x[1]]

class SetGenerator:

	def __init__(self, node_list):
		self.node_list = node_list
		self.N = len(node_list)
		self.A = np.tril(np.ones((self.N, self.N)))
		self.max_set = (self.N/2)+1

	def __del__(self):
		pass

	def finished(self):
		if np.sum(np.argwhere(self.A == 0)) > 0:
			return False
		return True	

	def all_sets(self):
		data = []
		self.A = np.tril(np.ones((self.N,self.N)))
		while not self.finished():
			tmp = self.single_index_set()
			data.append(tmp)

		return data

	def single_set(self):

		data = []
		tmp = self.single_index_set()
		for d in tmp:
			data.append((self.node_list[d[0]], self.node_list[d[1]]))
		return data

	def single_index_set(self):
		data = []
		B = np.zeros((self.N,self.N))

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
				data.append((row,col))
			else:
				return data

		return data


if __name__ == '__main__':
	

	N = int(sys.argv[1])

	# df = pd.read_csv('all_pairs_list.csv')
	# node_list = random_node_list(df,N)
	node_list = ['n{0}'.format(x) for x in range(N)]

	print 'number of nodes', len(node_list)
	print 'number of tests', (N*N-N)/2

	set_gen = SetGenerator(node_list)
	data = set_gen.all_sets()

	total = 0
	for tmp in data:
		print len(tmp)
		total+=len(tmp)
		
	print len(data), total






