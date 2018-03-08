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

	words = sentence.split()
	corpsize = sum(LM["uni"].values())
	#corpsize = len(LM["uni"])

	try:
		prob = LM["uni"][words[0]]/corpsize
	except KeyError:
		prob = 0

	for i in range(1, len(words)):
		try:
			numerator = LM["bi"][words[i-1]][words[i]]
		except KeyError:
			numerator = 0
			continue
		
		try:
			denominator = LM["uni"][words[i-1]]
		except KeyError:
			denominator = 0
			continue
		
		if denominator == 0:
			return float('-inf')
			#prob *= float('-inf')

		elif smoothing == False:
			#maximum likelihood estimate of the sentence
			prob *= numerator/denominator
		else:
			#a delta-smoothed estimate of the sentence
			prob *= (numerator + delta)/(denominator + (delta*vocabSize))

	if prob > 0:
		log_prob = log(prob, 2)
	else:
		log_prob = 0

	return log_prob