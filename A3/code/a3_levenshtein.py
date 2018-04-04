import os
import numpy as np
import string
import statistics

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """

    n = len(r)
    m = len(h)

    R = np.zeros((n+1, m+1))
    B = np.empty([n, m], dtype="S8")

    ((n+1,m+1))

    #Initialize 
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i,j] = float("Inf")

    R[0,0] = 0
    nS, nI, nD, x = 0,0,0,0

    if m > n:
        nI = m-n
        for x in r:
            if x not in h:
                nS += 1

    elif n > m:
        nD = n-m
        for x in h:
            if x not in r:
                nS += 1
    
    #Pseudocode
    for i in range(1, n+1):
        for j in range(1, m+1):
            delete = R[i-1, j] + 1
            insert = R[i,j-1] + 1

            try:
                if r[i-1] == h[i-1]:
                    substitute = R[i-1,j-1] + 0
                else:
                    substitute = R[i-1,j-1] + 1
            except IndexError:
                substitute = float("Inf")

            R[i,j] = min(delete, substitute, insert)
            
            #Backtracking
            if R[i,j] == delete:
                B[i-1,j-1]= 'up'
            elif R[i,j] == insert:
                B[i-1,j-1]= 'left'
            else:
                B[i-1,j-1]= 'up-left'
    
    #Case Transcript is empty
    if h == []:
        try:
            WER = (nS + nI + nD)/n
        except ZeroDivisionError:
            WER = 100.0
    #Case Transcript is empty
    elif r == []:
        try:
            WER = (nS + nI + nD)/n
        except ZeroDivisionError:
            WER = ((100*R[n,m])/n)
    else:
        WER = (nS + nI + nD)/n

    return [WER,  nS, nI, nD]


if __name__ == "__main__":

    exclude = set(string.punctuation)
    exclude.remove('[')
    exclude.remove(']')


    WK = []
    WG = []
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            path = os.path.join( dataDir, speaker )
            transcripts = os.path.join( path, "transcripts.txt")
            Kaldi = os.path.join( path, "transcripts.Kaldi.txt")
            Google = os.path.join( path, "transcripts.Google.txt")
            
            with open(transcripts) as t:
                t_lines = t.read().splitlines()

            with open(Kaldi) as k:
                k_lines = k.read().splitlines()

            with open(Google) as g:
                g_lines = g.read().splitlines()

            for i in range(len(t_lines)):
                t_lines[i] = ''.join(ch for ch in t_lines[i] if ch not in exclude)
                k_lines[i] = ''.join(ch for ch in k_lines[i] if ch not in exclude)
                g_lines[i] = ''.join(ch for ch in k_lines[i] if ch not in exclude)

                t_lines[i] = t_lines[i].lower()
                k_lines[i] = k_lines[i].lower()
                g_lines[i] = g_lines[i].lower()


                if i == 0 and speaker == "S-14B":
                    print(k_lines[i])
                    (print("\n"))
                    print(g_lines[i])
                    (print("\n"))
                    print(t_lines[i])
                k_levn = Levenshtein(t_lines[i].split(), k_lines[i].split())
                g_levn = Levenshtein(t_lines[i].split(), g_lines[i].split())


                #WK.append(k_levn[0])
                #WG.append(g_levn[0])
                #gtotal += g_levn[0]
                #print(speaker, " Kaldi ", i, k_levn[0], "S: ", k_levn[1], "I: ", k_levn[2], "D: ", k_levn[3])
                #print(speaker, " Google ", i, k_levn[0], "S: ", k_levn[1], "I: ", k_levn[2], "D: ", k_levn[3])


    #print("Kalidi Mean", statistics.mean(WK))
    #print("Google Mean", statistics.mean(WG))
    #print("Kalidi STDV", statistics.stdev(WK))
    #print("Kalidi STDV", statistics.stdev(WG))

                
