import numpy as np
import scipy.spatial.distance as spd
import sys
import time

np.random.seed(123456789*5)
coreset_size = 4000 # vary
iterations = 150   # vary
feature_size = 250
final_size = 200
MIN_IMAGE = 0
MAX_IMAGE = 10000

def mapper(key, value):
    #print value.shape
    c = [value[np.random.choice(value.shape[0])]]
    mindist = spd.cdist(np.reshape(c[0], (1, -1)), value, metric='sqeuclidean').flatten()
    for i in xrange(coreset_size - 1):
        idx = np.random.choice(value.shape[0], p=mindist/np.sum(mindist))
        c.append(value[idx])
        dist = spd.cdist(np.reshape(value[idx], (1, -1)), value, metric='sqeuclidean').flatten()
        mindist = np.minimum(mindist, dist)
    yield 1, np.asarray(c)

def kMeans(values, size) :
        begin = time.time()
        centroids = [values[np.random.choice(values.shape[0])]]
        mindist = spd.cdist(np.reshape(centroids[0], (1, -1)), values, metric='sqeuclidean').flatten()
        for i in xrange(size - 1):
          idx = np.random.choice(values.shape[0], p=mindist/np.sum(mindist))
          centroids.append(values[idx])
          dist = spd.cdist(np.reshape(values[idx], (1, -1)), values, metric='sqeuclidean').flatten()
          mindist = np.minimum(mindist, dist)
        centroids = np.asarray(centroids)
        #r = np.random.randint(0, len(values), size)
        #centroids = values[r]
	
	for t in xrange(iterations):
		summ = np.zeros((size, feature_size))
		count = np.zeros(size)
		#print "Iteration " + str(t) + " ", (time.time() - begin)
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
    print values.shape
    cents = kMeans(values, final_size)
    yield cents
