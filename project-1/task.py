import random

P = 1000000007
H = 100
M = 8192
Hlist = []

def getHashFn()
    a = random.randrange(0, P)
    b = random.randrange(0, P)
    return (a, b)

def hash((a, b), val, M):
    return ((a * val + b) % P) % M

def makeHashList():
    for i in xrange(H):
        Hlist.append(getHashFn())

def getVideoID(line):
	name = s.split()[0]
	return name[-4:-1]

def mapper(key, value):
    # key: None
    # value: one line of input file
    if (len(Hlist) == 0):
        makeHList()
    col = generateShingle(value)
    
    if False:
        yield "key", "value"  # this is how you yield a key, value pair


def generateShingleMatrix(value):
	s = value;
	thelist = s.split()

	X = []

	for i in xrange(H):
		X.append(9999);

	for i in xrange(1, len(thelist)) :
		x = int(thelist[i])
		for j in xrange(H) :
			hf = Hlist[j]
			h = hash(hf, x, M)
			X[j] = min(H[j], h)

	return X



def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    if False:
        yield "key", "value"  # this is how you yield a key, value pair
