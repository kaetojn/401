from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os
import pickle
from collections import defaultdict

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
	
	"""
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""

	AM = {}	

    # Read training data
	d = read_hansard(train_dir, num_sentences)

    # Initialize AM uniformly
	AM = initialize(d["eng"], d["fre"])

	total = {}
	tcount = {}
	for key in AM.keys():
		total[key] = 0
		for key2 in AM[key].keys():
			if key not in tcount.keys():
				tcount[key] = {key2: 0}
			else:
				tcount[key][key2] = 0

    # Iterate between E and M steps
	i = 0
	while i < max_iter:
		print(i)
		em_step(AM, tcount, total, d["eng"], d["fre"])
		i+=1

    
    #Save Model
	with open(fn_AM+'.pickle', 'wb') as handle:
		pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return AM

# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
	"""
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	"""
	
	sentence = {}

	for subdir, dirs, files in os.walk(train_dir):
		english = []
		french = []
		for file in files:
			filename, file_extension = os.path.splitext(file)

			if file_extension == ".e":
				englishFile = os.path.join(train_dir, file)
				ffilename = filename + ".f"
				frenchFile = os.path.join(train_dir, ffilename)

          
				with open(englishFile, "r") as e:
					i = 0
					for line in e:
						if i == num_sentences:
							break
						i += 1
						line = preprocess(line, "e")
						english.append(line.strip("SENTSTART").strip("SENTEND").split())

				with open(frenchFile, "r") as f:
					i = 0
					for line in f:
						if i == num_sentences:
							break
						i += 1
						line = preprocess(line, "f")
						french.append(line.strip("SENTSTART").strip("SENTEND").split())

		sentence["eng"] = english
		sentence["fre"] = french

	return sentence

                   

def initialize(eng, fre):
	"""
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
	AM = {}

	for i in range(len(eng)):
		for word in range(len(eng[i])):

			
			AM[eng[i][word]] = {}

			#Get total number of french
			total = 0 
			for x in range(len(eng)):
				if eng[i][word] in eng[x]:
					total += len(fre[x])

			#Get all french occurences
			temp = {}
			for z in range(len(eng)):
				if eng[i][word] in eng[z]:
					for q in fre[z]:
						temp[q] = 1/total
			
			AM[eng[i][word]] = temp
	AM['SENTSTART']	= {'SENTSTART': 1}	
	AM['SENTEND']	= {'SENTEND': 1}	
	return AM

def em_step(AM, tcount, total, eng, fre):
	"""
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""

	for i in range(len(eng)):
		uniqueEnglish = set(eng[i])
		uniqueFrench = set(fre[i])

		for fword in uniqueFrench:
			denom = 0
			for eword in uniqueEnglish:
				try:
					denom += AM[eword][fword] * fre[i].count(fword)
				except KeyError:
					denom += 0
			for eword in uniqueEnglish:
				tcount[eword][fword] += (AM[eword][fword] * fre[i].count(fword) * eng[i].count(eword))/denom
				total[eword] += (AM[eword][fword] * fre[i].count(fword) * eng[i].count(eword))/denom

	for e in total:
		for f in tcount[e]:
			AM[e][f] = tcount[e][f]/total[e]



if __name__ == "__main__":


	print(align_ibm1("/h/u9/g6/00/ndukaeto/CSC401/401/A2/Hansard/Training/", 1000, 5, "/h/u9/g6/00/ndukaeto/CSC401/401/A2/New Folder/AM3"))

