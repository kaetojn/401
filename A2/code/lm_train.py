from preprocess import *
import pickle
import os

def lm_train(data_dir, language, fn_LM):
    """
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM
	
	INPUTS:
	
    data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT
	
	LM			: (dictionary) a specialized language model
	
	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts
	
	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """
    language_model = {"uni": 0, "bi": 0}

    u = {}
    b = {}

    path = os.getcwd()
    dirPath = os.path.join(path, data_dir)
    for subdir, dirs, files in os.walk(dirPath):
        for file in files:
            fullFile = os.path.join(dirPath, file)
            if file.endswith(language):
                with open(fullFile, "r") as f:
                    for line in f:
                        line = preprocess(line, language)
                        words = line.split()

                        
                        for i in range(1, len(words)):
                            if words[i-1] not in b.keys():
                                b[words[i-1]] = {}
                                b[words[i-1]][words[i]] = 1
                                
                            elif words[i] not in b[words[i-1]].keys():
                                b[words[i-1]][words[i]] = 1
                                
                            else:
                                b[words[i-1]][words[i]] += 1


                        for i in range(len(words)):
                            if words[i] not in u.keys():
                                u[words[i]] = 1
                            else:
                                u[words[i]] += 1
                        

    language_model = {"uni": u, "bi": b}

    #Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return language_model

if __name__ == "__main__":

    #"/u/cs401/A2_SMT/data/"

    lm_train("2", 'e', "/h/u9/g6/00/ndukaeto/CSC401/401/A2/train_english")
    lm_train("2", 'f', "/h/u9/g6/00/ndukaeto/CSC401/401/A2/train_french")
