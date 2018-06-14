from __future__ import division
import random


P = 1000000007
Hlist = []
M = 8193
r = 7
b = 40
H = r*b


def getHashCoefficients(k):
    a = k
    b = k+1
    return (a, b)

def hash((a, b), val):
    return ((a * val + b) % P) % M

def makeHashList():
    for i in xrange(r*b):
        Hlist.append(getHashCoefficients(i))

def getVideoID(line):
	name = line.split()[0]
	return int(name[-4:])

def generateShinglesMatrix(value):
    s = value
    thelist = s.split()
    del thelist[0]
    thelist = list(map(int, thelist))    

    X = [9999] * H
    for i in xrange(len(thelist)) :
        x = thelist[i]                                  #shingle order number
        for j in xrange(r*b) :
            hashCoefficients = Hlist[j]               #coefficients for the hash function
            h = hash(hashCoefficients, x)
            X[j] = min(X[j], h)
    return X

def hashBand(signature, band):
    sum = 0;
    for row in xrange(r):
        i = band*r + row
        hashCoefficient = Hlist[i]
        outcome = hash(hashCoefficient, signature[i])
        sum  = (sum + outcome) % M
    return sum

def boostHashing(ID, signature):
    for band in xrange(b):
        result = hashBand(signature, band)
        keystring = str(band) + "--" + str(result)
        yield keystring, ID

def mapper(key, value):
    if (len(Hlist) == 0):
        makeHashList()

    signatureMatrix = generateShinglesMatrix(value)
    ID = getVideoID(value)

    for band in xrange(b):
        result = hashBand(signatureMatrix, band)
        keystring = str(band) + "--" + str(result)
        yield keystring, value

def calculateJaccardSimilarity(minline, maxline):
    list1 = minline.split()
    del list1[0]
    list1 = [float(x) for x in list1]

    list2 = maxline.split()
    del list2[0]
    list2 = [float(x) for x in list2]

    unique1 = set(list1)
    unique2 = set(list2)
    intersection = unique1 & unique2    
    union = unique1 | unique2
    jaccard = float(len(intersection))/float(len(union))    
    return jaccard

def reducer(key, values):
    if len(values) > 1:
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                jaccard = calculateJaccardSimilarity(values[i], values[j])
                if jaccard >= 0.85:
                    minvideo = getVideoID(min(values[i], values[j]))
                    maxvideo = getVideoID(max(values[i], values[j]))
                    
                    yield minvideo, maxvideo

        
