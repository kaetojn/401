from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import argparse
import sys
import os
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import operator
import csv


def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    numer = 0
    denom = 0
    for i in range(4):
        for j in range(4):
            if i == j:
                numer += C[i, j]
            denom += C[i, j]

    if(denom != 0):
        return numer/denom
    return 0



def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recallList = []
    denom = 0
    for k in range(4):
        numer = C[k, k]
        for j in range(4):
            denom += C[k, j]

        if(denom != 0):
            recallList.append(numer/denom)
        else:
            recallList.append(float(0))
    return recallList



def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precisionList = []
    denom = 0
    for k in range(4):
        numer = C[k, k]
        for i in range(4):
            denom += C[i, k]

        if(denom != 0):
            precisionList.append(numer/denom)
        else:
            precisionList.append(float(0))
    return precisionList
    

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''

    #implement dictionary to story accuracy for classifiers
    accuracydict = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}

    #data to write to the csv file
    data = [[1, accuracydict['1']],[2, accuracydict['2']],[3, accuracydict['3']],[4, accuracydict['4']],[5, accuracydict['5']]]

    #load and store features
    features = np.load(filename)
    X = features.f.arr_0[:,range(0,174)]
    y = features.f.arr_0[:,173]
    
    #train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

    # 1. SVC (Radial Basis Function Kernel)
    clf = SVC()
    clf.fit(X_train, y_train)
    svc1 = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
    accuracydict['1'] = accuracy(svc1)
    [data[0].append(i) for i in recall(svc1)]
    [data[0].append(i) for i in precision(svc1)]
    [data[0].append(j) for i in svc1 for j in i]

    # 2. SVC (Linear Kernel)
    clf = LinearSVC(random_state=0)
    clf.fit(X_train, y_train)
    svc2 = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
    accuracydict['2'] = accuracy(svc2)
    [data[1].append(i) for i in recall(svc2)]
    [data[1].append(i) for i in precision(svc2)]
    [data[1].append(j) for i in svc2 for j in i]

    # 3. RandomForestClassifier
    clf = RandomForestClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    rfc = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
    accuracydict['3'] = accuracy(rfc)
    [data[2].append(i) for i in recall(rfc)]
    [data[2].append(i) for i in precision(rfc)]
    [data[2].append(j) for i in rfc for j in i]


    # 4. MLPClassifier:
    clf = MLPClassifier(alpha=0.05)
    clf.fit(X_train, y_train)
    mlp = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
    accuracydict['4'] = accuracy(mlp)
    [data[3].append(i) for i in recall(mlp)]
    [data[3].append(i) for i in precision(mlp)]
    [data[3].append(j) for i in mlp for j in i]

    # 5. AdaBoostClassifier
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    abc = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
    accuracydict['5'] = accuracy(abc)
    [data[4].append(i) for i in recall(abc)]
    [data[4].append(i) for i in precision(abc)]
    [data[4].append(j) for i in abc for j in i]
    
    
    iBest = max(accuracydict, key=lambda key: accuracydict[key])

    with open('a1_3.1.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(data)

    return (X_train, X_test, y_train, y_test,iBest)    


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    if iBest == 1:
        clf = SVC()
        clf.fit(X_train, y_train)
        
    elif iBest == 2:
        clf = LinearSVC(random_state=0)
        clf.fit(X_train, y_train)
        matrix = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
    elif iBest == 3:
        clf = RandomForestClassifier(max_depth=5)
        clf.fit(X_train, y_train)
        matrix = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])   
    elif iBest == 4:
        clf = MLPClassifier(alpha=0.05)
        clf.fit(X_train, y_train)
        matrix = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
    elif iBest == 5:
        clf = AdaBoostClassifier()
        clf.fit(X_train, y_train)
        matrix = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])

    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
#def main( args ):
if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Process each .')
    #parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    #args = parser.parse_args()


    #class31(args.input)
    x = class31("feats.npz")
    best = x[4]

    '''
    #features = np.load(args.input)
    features = np.load(feats.npz)
    X = features.f.arr_0[:,range(0,174)]
    y = features.f.arr_0[:,173]

    #1K
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000, random_state=42)
    class32(X_train, X_test, y_train, y_test, best)
    
    #5K
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, random_state=42)
    class32(X_train, X_test, y_train, y_test, best)

    #10k
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, random_state=42)
    class32(X_train, X_test, y_train, y_test, best)
    
    #15k
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=15000, random_state=42)
    class32(X_train, X_test, y_train, y_test, best)

    #20k
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=20000, random_state=42)
    class32(X_train, X_test, y_train, y_test, best)
    '''
    
    #class33("feats.npz"):
    #class34("feats.npz"):

    #main(args)
