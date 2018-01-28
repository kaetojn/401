import numpy as np
import numpy
import sys
import argparse
import os
import json
from collections import Counter
import re
import string
import csv

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    #Split comment
    z  = comment.split()

    #Coordinating conjunctions
    cc = ["for", "and", "nor", "but", "or", "yet", "so"]

    #Future
    future = ["will", "gonna"]

    #Dictionary to track amount of Features 1-14
    worddict = {"fP": 0, "sP": 0, "tP": 0, "cc": 0, "past": 0, "fut": 0, "coma": 0, "mult": 0, "cNoun": 0, "pNoun": 0, "advb": 0, "wh": 0, "slang": 0, "upper": 0}
    
    #Initializiing Variables
    tokenlen = 0
    totalaoa = 0
    totalimg = 0
    totalfam = 0
    aoa = np.array([])
    img = np.array([])
    fam = np.array([])
    totalV = 0
    totalA = 0
    totalD = 0
    VMS = np.array([])
    AMS = np.array([])
    DMS = np.array([])

    for i in z:
        delimiter = i.index('/')
        word = i[:delimiter]
        tag = i[delimiter+1:]

        #Populating worddict

        #Feature 16 (Average length of tokens, excluding punctuation-only tokens)
        if (tag not in word) and (tag != ".") and (word not in string.punctuation):
            tokenlen += len(word)
        #Feature 1 (Number of first-person pronouns)
        if word in open('../Wordlists/First-person').read().splitlines():
            worddict["fP"] += 1
        #Feature 2 (Number of second-person pronouns)    
        elif word in open('../Wordlists/Second-person').read().splitlines():
            worddict["sP"] += 1
        #Feature 3 (Number of third-person pronouns)  
        elif word in open('../Wordlists/Third-person').read().splitlines():
            worddict["tP"] += 1
        #Feature 4 (Number of coordinating conjunctions)
        elif (word in open('../Wordlists/Conjunct').read().splitlines()) or (word in cc):
            worddict["cc"] += 1
        #Feature 5 (Number of past-tense verbs)
        elif tag == "vbd":
            worddict["past"] += 1
        #Feature 6 (Number of future-tense verbs)
        elif word == "will" or word == "gonna" or "'ll" in word:
            worddict["fut"] += 1
        elif word == "going":
            index = z.index(word+i[delimiter:])
            if(z[index+1][:2]) == "to":
                if(z[index+2][-2:]) == "vb":
                    worddict["fut"] += 1
        #Feature 7 (Number of commas)
        elif word == "," and tag == ",":
            worddict["coma"] += 1
        #Feature 8 (Number of multi-character punctuation tokens)
        elif len(word) > 1 and tag == ".":
            worddict["mult"] += 1
        #Feature 9 (Number of common nouns)
        elif tag == "nn" or  tag == "nns":
            worddict["cNoun"] += 1
        #Feature 10 (Number of proper nouns)    
        elif tag == "nnp" or tag == "nnps":
            worddict["pNoun"] += 1
        #Feature 11 (Number of adverbs) 
        elif tag == "rb" or tag == "rbr" or  tag == "rbs":
            worddict["advb"] += 1
        #Feature 12 (Number of wh- words)
        elif tag == "wdt" or tag == "wp" or  tag == "wp$" or tag == "wrb":
            worddict["wh"] += 1
        #Feature 13 (Number of slang words)
        elif word in open('../Wordlists/Slang').read().splitlines():
            worddict["slang"] += 1
        #Feature 14 (Number of words in uppercase (≥ 3 letters long))
        elif word.isupper() and length(word) > 3:
            worddict["upper"] += 1
        #For Features 18-23 (Bristol, Gilhooly, and Logie norms)
        with open('../Wordlists/BristolNorms+GilhoolyLogie.csv') as f:
            rows = csv.reader(f)
            for row in rows:
                if word == row[1]:                 
                    totalaoa += int(row[3])
                    totalimg += int(row[4])
                    totalfam += int(row[5])
                
                    aoa = numpy.append(aoa, [int(row[3])])
                    img = numpy.append(img, [int(row[4])])
                    fam = numpy.append(fam, [int(row[5])])
        #For Features 24 - 29 ( Warringer norms)
        with open('../Wordlists/Ratings_Warriner_et_al.csv') as f:
            rows = csv.reader(f)
            for row in rows:
                if word == row[1]:                 
                    totalV += float(row[2])
                    totalA += float(row[5])
                    totalD += float(row[8])
                
                    VMS = numpy.append(VMS, [float(row[2])])
                    AMS = numpy.append(AMS, [float(row[5])])
                    DMS = numpy.append(DMS, [float(row[8])])
                   
    '''
    tagcount = Counter()
    words = re.compile(r'\w+')

    for o in tags:
        tagcount.update(words.findall(o.lower()))
    tagdict = dict(tagcount)

    (word in open('../Wordlists/femaleFirstNames.txt').read().splitlines()) or (word in open('../Wordlists/maleFirstNames.txt').read().splitlines()) or (word in open('../Wordlists/lastNames.txt').read().splitlines())
    '''
    #For Features 15, 16 & 17    
    sentences = comment.splitlines()
    totalLen = 0
    
    for i in sentences:
        split = i.split()
        totalLen += len(split)

    aoa = numpy.append(aoa, [np.zeros((len(z) - len(aoa)))])
    img = numpy.append(img, [np.zeros((len(z) - len(img)))])
    fam = numpy.append(fam, [np.zeros((len(z) - len(fam)))])

    VMS = numpy.append(VMS, [np.zeros((len(z) - len(VMS)))])
    AMS = numpy.append(AMS, [np.zeros((len(z) - len(AMS)))])
    DMS = numpy.append(DMS, [np.zeros((len(z) - len(DMS)))])

    numpyarry = np.zeros((174))
    numpyarry[0] = worddict['fP']
    numpyarry[1] = worddict['sP']
    numpyarry[2] = worddict['tP']
    numpyarry[4] = worddict['past']
    numpyarry[5] = worddict['fut']
    numpyarry[6] = worddict['coma']
    numpyarry[7] = worddict['mult']
    numpyarry[8] = worddict['cNoun']
    numpyarry[9] = worddict['pNoun']
    numpyarry[10] = worddict['advb']
    numpyarry[11] = worddict['wh']
    numpyarry[12] = worddict['slang']
    numpyarry[13] = worddict['upper'] 
    numpyarry[14] = totalLen//len(sentences)
    numpyarry[15] = tokenlen//len(z)
    numpyarry[16] = len(sentences)
    numpyarry[17] = totalaoa//len(z)
    numpyarry[18] = totalimg//len(z)
    numpyarry[19] = totalfam//len(z)
    numpyarry[20] = np.std(aoa)
    numpyarry[21] = np.std(img)
    numpyarry[22] = np.std(fam)
    numpyarry[23] = totalV//len(z)
    numpyarry[24] = totalA//len(z)
    numpyarry[25] = totalD//len(z)
    numpyarry[26] = np.std(VMS)
    numpyarry[27] = np.std(AMS)
    numpyarry[28] = np.std(DMS)
    
    return numpyarry



def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 174), dtype=float)
    

    

    # TODO: your code here
    
    for i in range(len(data)):
        '''print("feats:")
        print(feats)
        print("\n")
        print("\n")'''
        #result = extract1(data[i]['body'])
        
        IDs = open('../feats/' + data[i]['cat'] +'_IDS.txt').read().splitlines()
        IDindex = IDs.index(data[i]['id'])
        #test = numpy.append(test, [result.reshape(1,29)])

        with np.load('../feats/' + data[i]['cat'] +'_feats.dat.npy') as data:
            a = data[IDindex]
        print(a)
        #print(IDs)
        print("\n")
        #feats[i] = result
           
    np.savez_compressed( args.output, feats)
    

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)