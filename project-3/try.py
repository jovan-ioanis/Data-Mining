import numpy as np
import scipy.spatial.distance as spd

np.random.seed(123456789*5)

def printList(c) :
	print("----------------------")
	for i in range(len(c)) :
		print(c[i])
	print("======================")

def mapper(key, value):
	print(value.shape)
	c = np.array(value[np.random.choice(value.shape[0]), :], ndmin=2)
	for i in range(6):
		printList(c)
		if i == 0:
			ds = spd.cdist(c, value, 'sqeuclidean')
		else :
			c_ = np.array(c[len(c)-1, :], ndmin=2)
			ds_ = spd.cdist(c, value, 'sqeuclidean')
			ds = np.vstack((ds, ds_))
		mindist = np.min(ds, axis=0).flatten()
		print(mindist)
		print(np.sum(mindist))
		print(mindist/np.sum(mindist))
		idx = np.random.choice(value.shape[0], p=mindist/np.sum(mindist))
		c = np.vstack((c, value[idx, :]))
		rest = np.setdiff1d(value, c)
		randchosen = np.random.randint(0, 17 - 6, 2)
		chosen = np.zeros((2, 3))
		for i in range(2):
			chosen[i] = value[randchosen[i]]
		print(c.shape)
		print(chosen.shape)
		c = np.vstack((c, chosen))

def main():
	a = np.array([[0, 0, 1], [2, 0, 9], [3, 4, 5], [6, 5, 2], [8, 9, 6], [3, 5, 4], [9, 0, 1], [5, 6, 7],
	[4, 7, 6], [4, 2, 0], [7, 6, 1], [0, 0, 5], [7, 3, 5], [6, 2, 3], [7, 7, 7], [9, 9, 9], [2, 3, 2]])

	mapper(1, a)

if __name__ == "__main__":
    main()



# def mapper(key, value):
# 	print(value.shape)
# 	c = np.array(value[np.random.choice(value.shape[0]), :], ndmin=2)
# 	summation = 0.0
# 	minimal = 0.0
# 	for i in range(coreset_size):
# 		if i == 0:
# 			ds = spd.cdist(c, value, 'sqeuclidean')
# 			mindist = np.min(ds, axis=0).flatten()
# 			minimal = mindist
# 			summation += np.sum(mindist)
# 		else :
# 			c_ = np.array(c[len(c)-1, :], ndmin=2)
# 			ds_ = spd.cdist(c, value, 'sqeuclidean')
# 			mindist1 = np.min(ds, axis=0).flatten()
# 			minimal = min(mindist1, minimal)
# 			summation += np.sum(mindist1)
# 		idx = np.random.choice(value.shape[0], p=minimal/summation)
# 		c = np.vstack((c, value[idx, :]))
# 	yield 1, c