import numpy as np
import numpy
import sys
import argparse
import os
import json
import re
import string
import csv
import warnings

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
        if word in open('First-person').read().splitlines():
            worddict["fP"] += 1
        #Feature 2 (Number of second-person pronouns)    
        elif word in open('Second-person').read().splitlines():
            worddict["sP"] += 1
        #Feature 3 (Number of third-person pronouns)  
        elif word in open('Third-person').read().splitlines():
            worddict["tP"] += 1
        #Feature 4 (Number of coordinating conjunctions)
        elif (word in open('Conjunct').read().splitlines()) or (word in cc):
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
        elif word in open('/u/cs401/Wordlists/Slang').read().splitlines():
            worddict["slang"] += 1
        #Feature 14 (Number of words in uppercase (≥ 3 letters long))
        elif word.isupper() and length(word) > 3:
            worddict["upper"] += 1
        #For Features 18-23 (Bristol, Gilhooly, and Logie norms)
        with open('BristolNorms+GilhoolyLogie.csv') as f:
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
        with open('Ratings_Warriner_et_al.csv') as f:
            rows = csv.reader(f)
            for row in rows:
                if word == row[1]:                 
                    totalV += float(row[2])
                    totalA += float(row[5])
                    totalD += float(row[8])
                
                    VMS = numpy.append(VMS, [float(row[2])])
                    AMS = numpy.append(AMS, [float(row[5])])
                    DMS = numpy.append(DMS, [float(row[8])])
                   
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
    
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore", category=RuntimeWarning)
    numpyarry = np.zeros((174))
    #np.seterr(divide='ignore', invalid='ignore')
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

    try:
        np.seterr(divide='ignore', invalid='ignore')
        numpyarry[14] = totalLen//len(sentences)
    except ZeroDivisionError:
        numpyarry[14] = 0
    try:
        np.seterr(divide='ignore', invalid='ignore')
        numpyarry[15] = tokenlen//len(z)
    except ZeroDivisionError:
        numpyarry[15] = 0

    numpyarry[16] = len(sentences)

    try:
        np.seterr(divide='ignore', invalid='ignore')
        numpyarry[17] = totalaoa//len(z)
    except ZeroDivisionError:
        numpyarry[17] = 0
    try: 
        np.seterr(divide='ignore', invalid='ignore')
        numpyarry[18] = totalimg//len(z)
    except ZeroDivisionError:
        numpyarry[18] = 0
    try: 
        np.seterr(divide='ignore', invalid='ignore')
        numpyarry[19] = totalfam//len(z)
    except ZeroDivisionError:
        numpyarry[19] = 0

    
    try:
        np.seterr(divide='ignore', invalid='ignore')
        numpyarry[23] = totalV//len(z)
    except ZeroDivisionError:
        numpyarry[23] = 0
    try:
        np.seterr(divide='ignore', invalid='ignore')
        numpyarry[24] = totalA//len(z)
    except ZeroDivisionError:
        numpyarry[24] = 0
    try:
        np.seterr(divide='ignore', invalid='ignore') 
        numpyarry[25] = totalD//len(z)
    except ZeroDivisionError:
        numpyarry[25] = 0

    numpyarry[26] = np.std(VMS)
    numpyarry[27] = np.std(AMS)
    numpyarry[28] = np.std(DMS)
    
    return numpyarry



def main( args ):

    data = json.load(open(args.input))
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore", category=RuntimeWarning)
    feats = np.zeros( (len(data), 174))
    #feats = np.seterr(divide='ignore', invalid='ignore')
    
    
    for i in range(len(data)):
        result = extract1(data[i]['body'])
        
        IDs = open('/u/cs401/A1/feats/' + data[i]['cat'] +'_IDs.txt').read().splitlines()
        IDindex = IDs.index(data[i]['id'])
        #test = numpy.append(test, [result.reshape(1,29)])

        a = np.load('/u/cs401/A1/feats/' + data[i]['cat'] +'_feats.dat.npy')
        
        for x in range(a[IDindex].size):
            result[x+29] = a[IDindex][x]

        if data[i]['cat'] == "Left":
            result[173] = 0
        elif data[i]['cat'] == "Center":
            result[173] = 1
        elif data[i]['cat'] == "Right":
            result[173] = 2
        elif data[i]['cat'] == "Alt":
            result[173] = 3
	
        feats[i] = result
        result = numpy.around(result, 2)

    np.savez_compressed( args.output, feats)

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)
