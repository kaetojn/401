from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary
	
	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
	
	#TODO: Implement by student.

	words = sentence.split()
	corpsize = sum(LM["uni"].values())
	prob = LM["uni"][words[0]]/corpsize

	for i in range(1, len(words)):
		numerator = LM["bi"][words[i-1]][words[i]]
		denominator = LM["uni"][words[i-1]]

		if denominator == 0:
			return float('-inf')
			prob *= float('-inf')

		elif smoothing == False:
			#maximum likelihood estimate of the sentence
			prob *= numerator/denominator
		else:
			#a delta-smoothed estimate of the sentence
			prob *= (numerator + delta)/(denominator + (delta*vocabSize))

    log_prob = log(prob)        
    return log_prob