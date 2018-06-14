from __future__ import division
import numpy as np
import sys

coreset_size = 500
feature_size = 250
final_size = 200
iterations = 120
MIN_IMAGE = 0
MAX_IMAGE = 10000


def extract(value) :
	# coreset = np.zeros((coreset_size, feature_size))
	v = np.random.randint(MIN_IMAGE, MAX_IMAGE, coreset_size)
	# for i in range(coreset_size) :		
	# 	coreset[i] = value[v[i]]
	coreset = value[v] # NumPy is the shiz
	print(coreset.shape)
	return coreset


def mapper(key, value):
    coreset = extract(value)
    yield 1, coreset

def kMeans(values, size) :
	# 10*500 x 250 
	r = np.random.randint(0, len(values), size)
	# centroids = np.zeros((size, feature_size))
	# for i in range(size) :
	# 	centroids[i] = values[r[i]]
	centroids = values[r]
	
	for t in range(iterations):
		summ = np.zeros((size, feature_size))
		count = np.zeros(size)
		print("Iteration " + str(t))
		for i in range(len(values)) :
			m = sys.maxint
			j = -1
			for k in range(len(centroids)) :
				dist = np.linalg.norm(centroids[k] - values[i])
				if dist < m :
					m = dist
					j = k

			summ[j] += values[i]
			count[j] += 1

		for c in range(len(centroids)) :
			if count[c] == 0 : 
				count[c] = 1
			centroids[c] = np.divide(summ[c], count[c])

	return centroids


def reducer(key, values):
	print(values.shape)
	cents = np.zeros((final_size, feature_size))
	for i in range(4) :
		centroids = kMeans(values, final_size)
		cents += centroids
	# np.divide(cents, 4)
	cents /= 4.0 # careful of types, just in case 
	yield cents

