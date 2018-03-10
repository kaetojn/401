import math

def BLEU_score(candidate, references, n):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.

	
	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""
	C = 0
	words = candidate.split()
	N = len(words) - (n-1)
	for i in range(N):
		ngram = " ".join(words[i:i+n])
		if any(ngram in word for word in references):
			C += 1
	precision = C/N

	reflens = []
	for refs in references:
		reflens.append(abs(len(words) - len(refs.split())))

	index = reflens.index(min(reflens))
	ri = len(references[index].split())
	ci = len(words)

	brevity = ri/ci

	if brevity < 1:
		BP = 1
	else:
		BP = math.exp(1-brevity)

    bleu_score = (BP * (precision ** (1/n)))       
    return bleu_score