from __future__ import division
import random

# Prime number bigger than the number of M (=the possible values of a shingle): 
prime = 1000000007

# Creation of the list containing the hashfunctions:
hashList = []

# Number of possible values for a shingle
M = 8193

# Number of bands in the locality sensitive hashing implementation:
nBands = 40

# Number of rows per band in the locality sensitive hashing implementation:
nRows = 7

# Number of hashfunctions required to partion the signature matrix into b bands
# of r rows:
nHashFunctions = nRows*nBands


def getHashCoefficients(k):
    """
    To determine the coefficients a and b used for computing the hashfunctions
    """
    a = k
    b = k+1
    return (a, b)

def hash((a, b), val):
    """
    To compute the hashfunctions given a and b
    """
    return ((a * val + b) % prime) % M

def makeHashList():
    """
    To create the list containing all the hashfunctions
    """
    for i in xrange(nHashFunctions):
        hashList.append(getHashCoefficients(i))

def getVideoID(line):
    """
    To get the ID (XXXX) of the video from its name VIDEO_00000XXXX
    """
    name = line.split()[0]
    return int(name[-4:])

def getSignatureMatrix(value):
    """
    To create the signature matrix
    """
    # Creation of the list of shingles for a given videos:
    shingleList = value.split()
    
    # We delete the first elements of the list since it corresponds to the 
    # video ID and not to a shingle: 
    del shingleList[0]
    shingleList = list(map(int, shingleList))

    # Number of shingles for the given video:
    nShingles = len(shingleList)
    
    # Initialisation of the signature matrix
    signatureMatrix = [9999] * nHashFunctions
    
    # In this two loops we compute the signature matrix for the given video
    for s in xrange(nShingles) :
        shingleValue = shingleList[s]                                  
        for j in xrange(nRows*nBands) :
            hashCoefficients = hashList[j]               
            h = hash(hashCoefficients, shingleValue)
            
            # We compute the signature matrix using the min-hash
            signatureMatrix[j] = min(signatureMatrix[j], h)
            
    return signatureMatrix

def hashBand(signatureMatrix, band):
    """
    To create the hash function for the band
    """
    sumRowHash = 0
    for row in xrange(nRows):
        # Row number (=index in the signature matrix)
        indexRow = band*nRows + row
        
        hashCoefficient = hashList[indexRow]

        # We hash each row of the band:
        rowHash = hash(hashCoefficient, signatureMatrix[indexRow])
        
        # The sum of each row-hash function is the hash function of the band
        sumRowHash  = (sumRowHash + rowHash) % M

    return sumRowHash
    
def jaccardSimilarity(vide01, video2):
    """
    To compute the real Jaccard similarity between two videos, and so check 
    that the 2 videos are not false positive
    """
    # List of shingles of the first video
    listSinglesV1 = vide01.split()
    del listSinglesV1[0]
    listSinglesV1 = [float(shingle) for shingle in listSinglesV1]
    
    # List of shingles of the first video
    listSinglesV2 = video2.split()
    del listSinglesV2[0]
    listSinglesV2 = [float(shingle) for shingle in listSinglesV2]
    
    # Set containing only unique shingle values:
    uniqueV1 = set(listSinglesV1)
    uniqueV2 = set(listSinglesV2)
    
    # Intersection (elements in common) between the shingles of the first and 
    # the second videos:
    intersection = uniqueV1 & uniqueV2

    # Union (total number of shingle) between the shingles of the first and 
    # the second videos:    
    union = uniqueV1 | uniqueV2
    
    # Jaccard similarity formula:
    jaccard = float(len(intersection))/float(len(union))    
    
    return jaccard
    
###############################################################################
###############################  MAP FUNCTION  ################################
###############################################################################
def mapper(key, value):
    """
    INPUT:
        key: None
        value: A string containin the video ID and the list of the shingles of
        this video
    
    OUTPUT:
        key: the hash function of a given band of the signature matrix.
        value: A string containin the video ID and the list of the shingles of
        this video. So this is the same value as the input value
        
    The mapper emits consequently b (= number of bands in the signature matrix)
    (key, value) pairs, one for each band
    """
    
    # If the list containing all the hash functions has not been created, we 
    # create it using the function makeHashList
    if (len(hashList) == 0):
        makeHashList()
    
    # Computation of the signature matrix:
    signatureMatrix = getSignatureMatrix(value)
    
    # ID of the video
    ID = getVideoID(value)
    
    # For each band of the signature matrix we emit a (key, value) pair:
    for band in xrange(nBands):
        result = hashBand(signatureMatrix, band)
        keystring = str(band) + "--" + str(result)
        
        yield keystring, value


#################################  SHUFFLING  #################################
      
        
###############################################################################
##############################  REDUCE FUNCTION  ##############################
###############################################################################
def reducer(key, values):
    """
    INPUT:
        key: the hash function of a given band of the signature matrix
        value: a list of videos (string with the video's ID and all its 
        shingles) having the same key
    
    OUTPUT:
        key: the smallest video'a ID of the pair of the videos having indeed a 
        Jaccard similarity greater or equal to 0.85
        value: the largest video'a ID of the pair of the videos having indeed a 
        Jaccard similarity greater or equal to 0.85
    """
    
    # Number of similar videos:
    nVideoSimilar = len(values)
    
    if nVideoSimilar > 1: # We treat only the case where there are at least 2 videos
        for i in range(nVideoSimilar):
            for j in range(i+1, nVideoSimilar):
                # We cross check each videos by computing their Jaccard 
                # similarities to be sure that we do not take into account 
                # false positive and false negative:
                jaccard = jaccardSimilarity(values[i], values[j])
                
                # If the Jaccard similarity between two videos is indeed 
                # greater or equal to 0.85, we emit a (key, value) pair:
                if jaccard >= 0.85:
                    minvideo = getVideoID(min(values[i], values[j]))
                    maxvideo = getVideoID(max(values[i], values[j]))
                    
                    yield minvideo, maxvideo

        
