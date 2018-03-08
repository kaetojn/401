from log_prob import *
from preprocess import *
import os

def preplexity(LM, test_dir, language, smoothing = False, delta = 0):
    """
	Computes the preplexity of language model given a test corpus
	
	INPUT:
	
	LM : 		(dictionary) the language model trained by lm_train
	test_dir : 	(string) The top-level directory name containing data
				e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	language : `(string) either 'e' (English) or 'f' (French)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	"""
	
    files = os.listdir(test_dir)
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])
    
    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue
        
        opened_file = open(test_dir+ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language)
            tpp = log_prob(processed_line, LM, smoothing, delta, vocab_size)
            
            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())
        opened_file.close()
    if N > 0:
        pp = 2**(-pp/N)
    return pp
'''
if __name__ == "__main__":
    #test
    test_LM = lm_train("2", 'e', "/h/u9/g6/00/ndukaeto/CSC401/401/A2/train_english")
    x = preplexity(test_LM, "/u/cs401/A2_SMT/data/Hansard/Testing/", "e")
    print("English MLE Perplexity: ", x)

    print("\n")

    test_LM = lm_train("2", 'f', "/h/u9/g6/00/ndukaeto/CSC401/401/A2/train_french")
    x = preplexity(test_LM, "/u/cs401/A2_SMT/data/Hansard/Testing/", "f")
    print("French MLE Perplexity: ", x)

    print("\n")

    test_LM = lm_train("2", 'e', "/h/u9/g6/00/ndukaeto/CSC401/401/A2/train_french")
    x = preplexity(test_LM, "/u/cs401/A2_SMT/data/Hansard/Testing/", "e", True, 0.1)
    print("English add-1 Perplexity: ", x)

    test_LM = lm_train("2", 'e', "/h/u9/g6/00/ndukaeto/CSC401/401/A2/train_french")
    x = preplexity(test_LM, "/u/cs401/A2_SMT/data/Hansard/Testing/", "e", True, 0.2)
    print("English add-2 Perplexity: ", x)

    test_LM = lm_train("2", 'e', "/h/u9/g6/00/ndukaeto/CSC401/401/A2/train_french")
    x = preplexity(test_LM, "/u/cs401/A2_SMT/data/Hansard/Testing/", "e", True, 0.3)
    print("English add-3 Perplexity: ", x)

    test_LM = lm_train("2", 'e', "/h/u9/g6/00/ndukaeto/CSC401/401/A2/train_french")
    x = preplexity(test_LM, "/u/cs401/A2_SMT/data/Hansard/Testing/", "e", True, 0.5)
    print("English add-5 Perplexity: ", x)

    print("\n")

    test_LM = lm_train("2", 'f', "/h/u9/g6/00/ndukaeto/CSC401/401/A2/train_french")
    x = preplexity(test_LM, "/u/cs401/A2_SMT/data/Hansard/Testing/", "f", True, 0.1)
    print("French add-1 Perplexity: ", x)

    test_LM = lm_train("2", 'f', "/h/u9/g6/00/ndukaeto/CSC401/401/A2/train_french")
    x = preplexity(test_LM, "/u/cs401/A2_SMT/data/Hansard/Testing/", "f", True, 0.2)
    print("French add-2 Perplexity: ", x)

    test_LM = lm_train("2", 'f', "/h/u9/g6/00/ndukaeto/CSC401/401/A2/train_french")
    x = preplexity(test_LM, "/u/cs401/A2_SMT/data/Hansard/Testing/", "f", True, 0.3)
    print("French add-3 Perplexity: ", x)

    test_LM = lm_train("2", 'f', "/h/u9/g6/00/ndukaeto/CSC401/401/A2/train_french")
    x = preplexity(test_LM, "/u/cs401/A2_SMT/data/Hansard/Testing/", "f", True, 0.5)
    print("French add-5 Perplexity: ", x)
'''


