import re

def preprocess(in_sentence, language):
	""" 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:#commas
	out_sentence: (string) the modified sentence
	"""
	separate = in_sentence

	#separate end-of-sentence punctuation
	separate = re.sub(r"([\w/'+$\s-])+([!\.?\)\",:;\'\-])+\s*", r"\1 \2 ", separate) #colons and semicolons, parentheses
	separate = re.sub(r"([A-Za-z0-9])([\+\-\<\>\=]+)([A-Za-z0-9])\s*", r"\1 \2 \3 ", separate) #separate mathematical operations 
	separate = re.sub(r"\(([A-Za-z0-9])+([-]+\s*[-]*)([A-Za-z0-9])+\)", r"\1 \2 \3 ", separate) #separate hypen in bracket
	separate = re.sub(r"([\"])([\w/'+$\s-])+([\"])", r"\1 \2 \3 ", separate) #separate hypen leading quoatation 

	#separate contractions according to the source language
	if language == 'f':
		separate = re.sub(r"(l\')([A-Za-z]+)", r"\1 \2 ", separate) #leading l’
		separate = re.sub(r"([A-Za-z]{0,1}l\')([A-Za-z]+)", r"\1 \2 ", separate) #leading consonant
		separate = re.sub(r"(qu\')([A-Za-z]+)", r"\1 \2 ", separate) # leading qu’
		separate = re.sub(r"([A-Za-z]+\')(on|il)", r"\1 \2 ", separate) # on or il



	#convert all tokens to lower-case
	out_sentence = separate.lower()

	#convert all tokens to lower-case
	out_sentence = "SENTSTART " + out_sentence.strip('\n') + "SENTEND \n"

	return out_sentence