from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
import math
from scipy.special import logsumexp


dataDir = '/u/cs401/A3/data/'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
        
        Parameters:
        m = integer from 0 to M-1
        x = numpy row vector with dimensions ( d x 1 )
        myTheta = object of class 'theta'.
        preComputedForM =  list of the precomputable terms independent of x
    '''

    #mu
    mu = myTheta.mu[m]

    #variance
    v = myTheta.Sigma[m]

    f1 = 0
    a = 0
    b = 1

    #myTheta.d = 13
    for n in range(myTheta.d):
        f1 += ( (x[n]**2) / (2*v[n]) ) - ( (mu[n]*x[n]) / v[n] )
        
        a += ( (mu[n]**2) / (2*v[n]) )

        b *= v[n]

    f2 = a + ((myTheta.d/2)*math.log(2*math.pi, 2)) + ((0.5)*math.log(b, 2))

    logprob = (-f1) - f2

    return logprob
    
def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    
    numerator = myTheta.omega[m] * log_b_m_x( m, x, myTheta, [])
    denominaor = 0

    #myTheta.M = 8
    for k in range(myTheta.M):
        o = myTheta.omega[k]
        denominaor += o *  log_b_m_x( k, x, myTheta, [])

    logprob = (math.log(numerator, 2) - math.log(denominaor, 2))

    return logprob

    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    r = []

    for m in range(myTheta.M):
        eq4 = math.log(myTheta.omega[m], 2) * log_Bs[m]
        r.append(eq4)

    eq3 = logsumexp(r)

    return eq3

    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''

    #Initialize theta


    
    prev_L :=float("-Inf")
    improvement = float("Inf")

    i = 0
    while i <= maxIter and improvement >= epsilon:
        #ComputeIntermediateResults

        #L = ComputeLikelihood (X, theta)
            #create  M × T numPy (equation 1)
                #comput log_b_m_x
            #create  M × T numPy (equation 2)
                #compute log_p_m_x

        #θ = UpdateParameters (theta, X, L) ;
        

        improvement = L − prev L
        prev_L = L
        
        i += 1 
    
    myTheta = theta( speaker, M, X.shape[1] )

    
    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    print ('TODO')
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)

