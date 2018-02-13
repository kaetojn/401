from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import argparse
import sys
import os
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import operator
import csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold
from scipy import stats


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

    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    
    #train_test_split (32K)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

    
    # 1. SVC (Linear Kernel)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    svc1 = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
    accuracydict['1'] = accuracy(svc1)
    [data[0].append(i) for i in recall(svc1)]
    [data[0].append(i) for i in precision(svc1)]
    [data[0].append(j) for i in svc1 for j in i]

    # 2. SVC (Radial Basis Function Kernel)
    clf = SVC(kernel='rbf', gamma = 2)
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

List32 = [[]]
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
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        matrix = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
    elif iBest == 2:
        clf = SVC(kernel='rbf', gamma = 2)
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

    y = accuracy(matrix)
    List32[0].append(y)

    X_1k = X_train[:1000, :]
    y_1k = y_train[:1000]

    return (X_1k, y_1k)

'''
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
     This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    
    List33[[5],[10],[20],[30],[40],[50],[],[],[],[]] 

    #k=5
    selector = SelectKBest(f_classif, 5)
    X_new = selector.fit_transform(X_train, y_train)
    pp = selector.pvalues_
    List33[0].append(pp)

    #X_new = selector.fit_transform(X_train, y_train)

    #k=10
    selector = SelectKBest(f_classif, 10)
    X_new = selector.fit_transform(X_train, y_train)
    pp = selector.pvalues_
    List33[1].append(pp)

    #k=20
    selector = SelectKBest(f_classif, 20)
    X_new = selector.fit_transform(X_train, y_train)
    pp = selector.pvalues_
    List33[2].append(pp)

    #k=30
    selector = SelectKBest(f_classif, 30)
    X_new = selector.fit_transform(X_train, y_train)
    pp = selector.pvalues_
    List33[3].append(pp)

    #k=40
    selector = SelectKBest(f_classif, 40)
    X_new = selector.fit_transform(X_train, y_train)
    pp = selector.pvalues_
    List33[4].append(pp)

    #k=50
    selector = SelectKBest(f_classif, 50)
    X_new = selector.fit_transform(X_train, y_train)
    pp = selector.pvalues_
    List33[5].append(pp)


    if i == 1:
        clf = SVC(kernel='linear')
        #Original 32K Training 
        clf.fit(5X_new)

        #1K Traing 
        #clf.fit(5X_new_1k)
        matrix = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
        List33[6].append(accuracy(matrix))
    elif i == 2:
        clf = SVC(kernel='rbf')
        #Original 32K Training 
        clf.fit(5X_new)

        #1K Traing 
        #clf.fit(5X_new_1k)
        matrix = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
        List33[6].append(accuracy(matrix))
    elif i == 3:
        clf = RandomForestClassifier(max_depth=5)
        #Original 32K Training 
        clf.fit(5X_new)

        #1K Traing 
        #clf.fit(5X_new_1k)
        matrix = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])   
        List33[6].append(accuracy(matrix))
    elif i == 4:
        clf = MLPClassifier(alpha=0.05)
        #Original 32K Training 
        clf.fit(5X_new)

        #1K Traing 
        #clf.fit(5X_new_1k)
        matrix = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
        List33[6].append(accuracy(matrix))
    elif i == 5:
        clf = AdaBoostClassifier()
        #Original 32K Training 
        clf.fit(5X_new)

        #1K Traing 
        #clf.fit(5X_new_1k)
        matrix = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
        List33[6].append(accuracy(matrix))

    

    with open('a1_3.3.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(List33)
    

'''
def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    kf = KFold(n_splits=5, shuffle=True)

    features = np.load(filename)
    
    X = features.f.arr_0[:,range(0,174)]
    y = features.f.arr_0[:,173]

    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    List34 = [[],[],[],[],[]]

    index = 0
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

         # 1. SVC (Linear Kernel)
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        svc1 = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
        List34[index].append(accuracy(svc1))


        # 2. SVC (Radial Basis Function Kernel)
        clf = SVC(kernel='rbf', gamma = 2)
        clf.fit(X_train, y_train)
        svc2 = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
        List34[index].append(accuracy(svc2))
        

        # 3. RandomForestClassifier
        clf = RandomForestClassifier(max_depth=5)
        clf.fit(X_train, y_train)
        rfc = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
        List34[index].append(accuracy(rfc))
        


        # 4. MLPClassifier:
        clf = MLPClassifier(alpha=0.05)
        clf.fit(X_train, y_train)
        mlp = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
        List34[index].append(accuracy(mlp))
        

        # 5. AdaBoostClassifier
        clf = AdaBoostClassifier()
        clf.fit(X_train, y_train)
        abc = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1,2,3])
        List34[index].append(accuracy(abc))

        index += 1
    
    y = np.array(List34)
    a = y[:, i-1]
    list1 = []
    for j in range(5):
        
        if j != i-1:
            b = y[:, j]
            S = stats.ttest_rel(a, b)
            list1.append(S[1])
    List34.append(list1)   
    List34.append(["The liner SVC kernel produces the most accurate and consistent results with the smallest p-value. Next is the AdaBoostClassifier then RandomForestClassifier then MLPClassifier the Radial Basis SVC. In this case the 5fold cross verification is in line with my results."])  

    with open('a1_3.4.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(List34)

def main( args ):

    x = class31(args.input)

    best = int(x[4])

    
    features = np.load(args.input)
    X = features.f.arr_0[:,range(0,174)]
    y = features.f.arr_0[:,173]
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    
    #1K
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000)
    c32_1k = class32(X_train, X_test, y_train, y_test, best)
    #class33(X_train, X_test, y_train, y_test, best, c32_1k[0], c32_1k[1])
    
    #5K
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000)
    c32_5k = class32(X_train, X_test, y_train, y_test, best)
    #class33(X_train, X_test, y_train, y_test, best, c32_5k[0], c32_5k[1])

    #10k
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000)
    c32_10k = class32(X_train, X_test, y_train, y_test, best)
    #class33(X_train, X_test, y_train, y_test, best, c32_10k[0], c32_10k[1])

    #15k
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=15000)
    c32_15k = class32(X_train, X_test, y_train, y_test, best)
    #lass33(X_train, X_test, y_train, y_test, best, c32_15k[0], c32_15k[1])

    #20k
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=20000)
    c32_20k = class32(X_train, X_test, y_train, y_test, best)
    #class33(X_train, X_test, y_train, y_test, best, c32_20k[0], c32_20k[1])

    List32.append(["The accuracy of the classifer increases as the train increases. This is because the is more data for the for the classifier to learn from, hence improving its understanding"])
    with open('a1_3.2.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(List32)
    '''
    #Originial 32k
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
    c32_32k = class32(X_train, X_test, y_train, y_test, best)
    class33(X_train, X_test, y_train, y_test, best, c32_32k[0], c32_32k[1])
    
    '''
    class34(args.input, best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    main(args)
