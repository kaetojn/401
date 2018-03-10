from decode import *
import pickle
from preprocess import *
from align_ibm1 import *
from BLEU_score import *

def evalAlign(file, references, LM, AM):

	buff = []

	for i in range(len(references)):	
		buff.append(open(references[i], "r"))

	with open(file, "r") as f:
		for line in f:
			procFrench = preprocess(line, "f")
			english = decode(procFrench, LM, AM)

			blueRef = []
			for j in range(len(buff)):
				newline = buff[j].readline()
				blueRef.append(newline)

			blue1 = BLEU_score(english, blueRef, 1)
			blue2 = BLEU_score(english, blueRef, 2)
			blue3 = BLEU_score(english, blueRef, 3)

			print(blue1, blue2, blue3)
			
	for i in buff:
		i.close()

if __name__ == "__main__":

	references = ["/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e", "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e"]
	LM = pickle.load(open("/h/u9/g6/00/ndukaeto/CSC401/401/A2/New Folder/train_english.pickle", "rb"))

	train = "/h/u9/g6/00/ndukaeto/CSC401/401/A2/Hansard/Training/"
	test = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f"


	1K = align_ibm1(train, 1000, 5, "/h/u9/g6/00/ndukaeto/CSC401/401/A2/New Folder/AM1K")
	evalAlign(test, references, LM, 1K)

	10K = align_ibm1(train, 10000, 5, "/h/u9/g6/00/ndukaeto/CSC401/401/A2/New Folder/AM10K")
	evalAlign(test, references, LM, 10K)
	
	15K = align_ibm1(train, 15000, 5, "/h/u9/g6/00/ndukaeto/CSC401/401/A2/New Folder/AM15K")
	evalAlign(test, references, LM, 15K)
	
	30K = align_ibm1(train, 30000, 5, "/h/u9/g6/00/ndukaeto/CSC401/401/A2/New Folder/AM30K")
	evalAlign(test, references, LM, 30K)