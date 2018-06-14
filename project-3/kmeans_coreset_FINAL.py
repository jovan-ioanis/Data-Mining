import numpy as np
import scipy.spatial.distance as spd
import sys
import time

np.random.seed(123456789*5)
# Constants
coreset_size = 4000 # variable
iterations = 150   # variable, number of iterations of k-means
feature_size = 250
final_size = 200 # total number of output cluster centers
MIN_IMAGE = 0
MAX_IMAGE = 10000

def mapper(key, value):
    c = [value[np.random.choice(value.shape[0])]]	# Randomly choose a starting element of coreset
    mindist = spd.cdist(np.reshape(c[0], (1, -1)), value, metric='sqeuclidean').flatten()

    for i in xrange(coreset_size - 1): # Coreset creation using D2-Sampling
        idx = np.random.choice(value.shape[0], p=mindist/np.sum(mindist)) # Choose a random data point based on distance
        c.append(value[idx])
        dist = spd.cdist(np.reshape(value[idx], (1, -1)), value, metric='sqeuclidean').flatten()
        mindist = np.minimum(mindist, dist)

    yield 1, np.asarray(c)

def kMeans(values, size):
        centroids = [values[np.random.choice(values.shape[0])]] # choose a random point as the first cluster center
        mindist = spd.cdist(np.reshape(centroids[0], (1, -1)), values, metric='sqeuclidean').flatten()
        for i in xrange(size - 1): # Sample the rest of the cluster centers using D2-Sampling as well
          idx = np.random.choice(values.shape[0], p=mindist/np.sum(mindist))
          centroids.append(values[idx])
          dist = spd.cdist(np.reshape(values[idx], (1, -1)), values, metric='sqeuclidean').flatten()
          mindist = np.minimum(mindist, dist)
        centroids = np.asarray(centroids)

	for t in xrange(iterations): # Standard k-means clustering algorithm (LLoyd's Algorithm)
		summ = np.zeros((size, feature_size))
		count = np.zeros(size)
                dist = spd.cdist(centroids, values, metric='sqeuclidean')
                z = np.argmin(dist, axis=0)
                for i in xrange(len(values)):
                    j = z[i]
		    summ[j] += values[i]
		    count[j] += 1

		for c in xrange(size) :
			if count[c] != 0 :
			    centroids[c] = np.divide(summ[c], count[c])

	return centroids

def reducer(key, values):
    cents = kMeans(values, final_size) # Do the clustering in reducer
    yield cents
