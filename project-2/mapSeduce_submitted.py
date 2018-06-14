import numpy as np

# Number of random Fourier features:
nRFF=2000


def transform(X):

    return kernel(X)

def kernel(X):

    # X is a 2d np.array (size: (Number of images, number of random Fourier features)
    nImages, nFeatures = X.shape
    
    # We define the multivariate gaussian from which the weights of the random fourier
    # features will be computed:
    sigma = 0.11
    np.random.seed(123)
    weightsRFF = np.random.standard_normal(size=(nRFF/2, nFeatures))
    #offset = np.random.uniform(0, 2 * np.pi, size=nRFF)

    # Phase:
    phase = np.dot(X, weightsRFF.T)/sigma

    # Offset
    #phase += offset

    # Creation of the random Fourier features:
    cosX = np.cos(phase)
    #np.cos(phase, phase)
    sinX = np.sin(phase)
    RFF = np.concatenate((cosX, sinX),axis=1)
    
    return RFF

def miniBatches(batchLength, nBatch, nImages):

    # We create the mini-batch matrix which contains all the mini-batches:
    miniBatch = np.zeros((batchLength, nBatch),dtype=int)

    # We generate random integers which will corespond to the start index of 
    # the mini-batch:
    miniBatch[0,:] = np.random.randint(nImages-batchLength, size=nBatch)
    
    # We create the mini-batch matrix which contains all the mini-batches:
    for b in range(1,batchLength):
        miniBatch[b,:] = miniBatch[b-1,:]+1
    
    return miniBatch

def SGD(label, featureMatrix):

    # Number of images:
    nImages,nFeatures = featureMatrix.shape
    
    # step size:
    stepSize = 1
    # Exponential decay rates for the first moment estimate:
    expDecay1 = 0.95
    # Exponential decay rates for the second moment estimate:
    expDecay2 = 0.95
    # Wished precision of the algorithm:
    precision = 1e-7
    
    # Pegasos Regularizer:
    regularizer = 2.5e-10

    # Number of elements in each mini-batch:
    batchLength = 10
    # Number of mini-batches:
    nBatch = 15000
    # We compute the mini-batches:
    #miniBatch = miniBatches(batchLength, nBatch, nImages)

    # First moment vector:
    moment1 = np.zeros((nFeatures,1))
    # Second moment vector:
    moment2 = np.zeros((nFeatures,1))

    # weight vector:
    weights=np.zeros((nRFF,1))
    
    # Matrix resulting of the product between the labels and the features:
    preGradient = np.array(label, ndmin=2).T * featureMatrix
     
    for t in range(nBatch):
        #localGradient = preGradient[miniBatch[:,t],:]
        miniBatchStart = np.random.randint(nImages - batchLength)
        localGradient = preGradient[range(miniBatchStart, miniBatchStart + batchLength),:]
       
        # The gradient is zero if label*weight*features>=1:
        nonZeroGradient = (localGradient.dot(weights) < 1)
        
        # Gradient of the mini-batch:
        gradientMiniBatch = np.reshape(np.sum(-(nonZeroGradient * localGradient),axis=0),(nRFF,1))
        
        # Pegasos gradient:
        gradPegasos = regularizer*weights + gradientMiniBatch
       
        # Update biaised first moment estimate:
        moment1 = expDecay1*moment1 + (1-expDecay1) * gradPegasos

        # Update biaised second moment estimate:
        moment2 = expDecay2*moment2 + (1-expDecay2) * gradPegasos**2

        # Bias-corrected first moment estimate:
        moment1Corrected = moment1 / (1-expDecay1**(t+1))

         # Bias-corrected second moment estimate:
        moment2Corrected = moment2 / (1-expDecay2**(t+1))
       
        # Update of the weight vector:
        weights = weights - stepSize*moment1Corrected / (precision + \
            np.sqrt(moment2Corrected))
    
    return np.squeeze(weights)

def mapper(key, value):
    #key: None
    #value: one line of input file

    # Number of images sent to the mapper:
    nImages = len(value)
    print nImages
    
    # Number of features:
    nFeatures = 400

    # Original features of the images:
    originalFeatures = np.empty((nImages, nFeatures))

    # Labels of the images of the training dataset:
    labels = np.empty(nImages)

    for i in range(nImages):
        lineImage = value[i].split()
    	labels[i] = lineImage[0]
        
        originalFeatures[i,:] = lineImage[1:]

    # We use the function transform to transform our initial features
    # into random Fourier ones:
    randomFourierFeatures = transform(originalFeatures)

    weights= SGD(labels, randomFourierFeatures)

    #print(np.inner(weights,weights))

    yield 1,weights

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    
    # Number of features:
    nFeatures = len(values[0])
    
    # Initialization of the averaged weight vector which will 
    # be used for predictions:
    avgWeights = np.zeros(nFeatures)

    # Averaging of the weight vectors:
    for i in range(len(values)):
        avgWeights += values[i]

    avgWeights /= len(values)
    
    yield avgWeights
   



