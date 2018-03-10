from decode import *
import os
import pickle
from preprocess import *

def evalAlign(file, L, A):
	LM = pickle.load(open(L, "rb"))
	AM = pickle.load(open(A, "rb"))

	with open(file, "r") as f:
		for line in f:
			#preprocess(line, "f")
			english = decode(line, LM, AM)
			
			print(english)
			print(line)

if __name__ == "__main__":

	evalAlign("/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f", "/h/u9/g6/00/ndukaeto/CSC401/401/A2/train_english.pickle", "/h/u9/g6/00/ndukaeto/CSC401/401/A2/AM.pickle")
	