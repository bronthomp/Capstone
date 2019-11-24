# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:04:01 2019

@author: BRTHOMPSON
"""

import pandas as pd
import numpy as np
import datetime as dt
import sklearn as sk
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import wilcoxon
import statistics 

readmit_model1 = pd.read_csv(r'C:\Users\brthompson\bt_school\data\interim\readmit_model1.csv')
readmit_model2 = pd.read_csv(r'C:\Users\brthompson\bt_school\data\interim\secondmodel_readmit.csv')

readmit_model1.shape, readmit_model2.shape

#Split dependent and independent variables
Xad = readmit_model1.drop(['readmit'], 1) #Features
Yad = readmit_model1['readmit'] #Target variable
Xad.shape, Yad.shape

#Decision tree
#Use cross-validation for hyperparamter tuning - choose the best max_depth
depth = []
for i in range(3, 50):

    clf = DecisionTreeClassifier(max_depth=i) 
    scores = cross_val_score(estimator=clf, X = Xad, y = Yad, cv = 5, n_jobs = 4)
    depth.append((i, scores.mean()))
print(depth)
pd.DataFrame(depth, columns = ['max_depth', 'score']).set_index('max_depth').plot()
#Stabilizes at approximately 8
#Closer to 15 for larger set of variables
clf_tree = DecisionTreeClassifier(max_depth=8)

#Now determine best number of estimators for Boosted Decision Tree
#Use only the training set for the cross-validation
estimators = []
for i in range(25, 250, 25):

    clf = AdaBoostClassifier(base_estimator = clf_tree, random_state=0, n_estimators = i) 
    scores = cross_val_score(estimator=clf, X = Xad, y = Yad, cv = 5, n_jobs = 4)
    estimators.append((i, scores.mean()))
print(estimators)
pd.DataFrame(estimators, columns = ['n_estimators', 'score']).set_index('n_estimators').plot()
#max score achieved with 25 estimators
booster = AdaBoostClassifier(base_estimator = clf_tree, random_state=0, n_estimators = 25) 

#now perform parameter tuning for Random Forest
#Begin with number of estimators
estimators = []
for i in range(50, 250, 10):

    clf = RandomForestClassifier(n_estimators=i) 
    scores = cross_val_score(estimator=clf, X = Xad, y = Yad, cv = 5, n_jobs = 4)
    estimators.append((i, scores.mean()))
print(estimators)
pd.DataFrame(estimators, columns = ['n_estimators', 'score']).set_index('n_estimators').plot()
#Not much difference in number of estimators. Use 50

#Repeat for number of min samples
samples=[]
for i in range(80, 200, 10):

    clf = RandomForestClassifier(n_estimators=50, min_samples_split = i) 
    scores = cross_val_score(estimator=clf, X = Xad, y = Yad, cv = 5, n_jobs = 4)
    samples.append((i, scores.mean()))
print(samples)
pd.DataFrame(samples, columns = ['n_samples', 'score']).set_index('n_samples').plot()
#Best around 150
forest = RandomForestClassifier(n_estimators=50, min_samples_split = 150)

#Define logistic regression model (no tuning required)
logreg = LogisticRegression()

#Define a function that will run the selected model and output the measures
#of interest
def modelrun(model, xtrain, ytrain, xtest, ytest):
    start = time.time()
    model.fit(xtrain, ytrain)
    y_pred=model.predict(xtest)
    print(' ')
    
    print("Confusion Matrix:",metrics.confusion_matrix(ytest, y_pred))
    acc = metrics.accuracy_score(ytest, y_pred)
    prec = metrics.precision_score(ytest, y_pred)
    rec = metrics.recall_score(ytest, y_pred)
    print("Accuracy:",acc)
    print("Precision:",prec)
    print("Recall:",rec)
    y_pred_proba = model.predict_proba(xtest)[::,1]
    fpr, tpr, _ = metrics.roc_curve(ytest,  y_pred_proba)
    auc = metrics.roc_auc_score(ytest, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
    print('MCC:')
    print(matthews_corrcoef(ytest, y_pred))
    mcc = matthews_corrcoef(ytest, y_pred)
    elapsed = time.time() - start
    print("Time to run",elapsed)
    return acc, prec, rec, auc, mcc

#Define functions that append outcome variables to lists so that analyses can
#be performed on the results

def logreg_append(modelrun):
    acc_logreg.append(modelrun[0])
    prec_logreg.append(modelrun[1])
    rec_logreg.append(modelrun[2])
    aur_logreg.append(modelrun[3])
    mcc_logreg.append(modelrun[4])
    
def tree_append(modelrun):
    acc_tree.append(modelrun[0])
    prec_tree.append(modelrun[1])
    rec_tree.append(modelrun[2])
    aur_tree.append(modelrun[3])
    mcc_tree.append(modelrun[4])

def boost_append(modelrun):
    acc_booster.append(modelrun[0])
    prec_booster.append(modelrun[1])
    rec_booster.append(modelrun[2])
    aur_booster.append(modelrun[3])
    mcc_booster.append(modelrun[4])
    
def forest_append(modelrun):
    acc_forest.append(modelrun[0])
    prec_forest.append(modelrun[1])
    rec_forest.append(modelrun[2])
    aur_forest.append(modelrun[3])
    mcc_forest.append(modelrun[4])
    

##Run through cross validation for initial models
acc_logreg = []
prec_logreg = []
rec_logreg = []
aur_logreg = []
mcc_logreg = []

acc_tree = []
prec_tree = []
rec_tree = []
aur_tree = []
mcc_tree = []

acc_booster = []
prec_booster = []
rec_booster = []
aur_booster = []
mcc_booster = []

acc_forest = []
prec_forest = []
rec_forest = []
aur_forest = []
mcc_forest = []


for i in range (0,10):
    X_train,X_test,y_train,y_test=train_test_split(Xad,Yad,test_size=0.25,random_state=i)
    print('Logistic Regression', i)
    logreg1 = modelrun(model =logreg, xtrain=X_train, ytrain=y_train, xtest = X_test, ytest=y_test)
    logreg_append(logreg1)
    print('')
    print('Decision Tree', i)
    tree1 = modelrun(model =clf_tree, xtrain=X_train, ytrain=y_train, xtest = X_test, ytest=y_test)
    tree_append(tree1)
    print('')
    print('Boosted Trees', i)
    boost1 = modelrun(model = booster, xtrain=X_train, ytrain=y_train, xtest = X_test, ytest=y_test)
    boost_append(boost1)
    print('')
    print('Random Forest', i)
    forest1 = modelrun(model =forest, xtrain=X_train, ytrain=y_train, xtest = X_test, ytest=y_test)
    forest_append(forest1)

print("Logistic Regression accuracy:",acc_logreg, plt.plot(acc_logreg, scaley=False))
print("Logistic Regression precision:",prec_logreg, plt.plot(prec_logreg, scaley=False))
print("Logistic Regression recall:",rec_logreg, plt.plot(rec_logreg, scaley=False))
print("Logistic Regression AUROC:",aur_logreg, plt.plot(aur_logreg, scaley=False))
print("Logistic Regression MCC:",mcc_logreg, plt.plot(mcc_logreg, scaley=False)) 

print("Tree Accuracy:",acc_tree, plt.plot(acc_tree, scaley=False))
print("Tree Precision:",prec_tree, plt.plot(prec_tree, scaley=False))
print("Tree Recall:",rec_tree, plt.plot(rec_tree, scaley=False))
print("Tree AUROC:",aur_tree, plt.plot(aur_tree, scaley=False))
print("Tree MCC:",mcc_tree, plt.plot(mcc_tree, scaley=False))

print("Boosted Tree Accuracy:",acc_booster, plt.plot(acc_booster, scaley=False))
print("Boosted Tree Precision:",prec_booster, plt.plot(prec_booster, scaley=False))
print("Boosted Tree Recall:",rec_booster, plt.plot(rec_booster, scaley=False))
print("Boosted Tree AUROC:",aur_booster, plt.plot(aur_booster, scaley=False))
print("Boosted Tree MCC:",mcc_booster, plt.plot(mcc_booster, scaley=False))

print("Forest Accuracy:",acc_forest, plt.plot(acc_forest, scaley=False))
print("Forect Precision:",prec_forest, plt.plot(prec_forest, scaley=False))
print("Forest Recall:",rec_forest, plt.plot(rec_forest, scaley=False))
print("Forest AUROC:",aur_forest, plt.plot(aur_forest, scaley=False))
print("Forest MCC:",mcc_forest, plt.plot(mcc_forest, scaley=False)) 


#Copy results to new lists so that code can be reused

lr_acc_orig = acc_logreg.copy()
lr_prec_orig = prec_logreg.copy()
lr_rec_orig = rec_logreg.copy()
lr_auroc_orig = aur_logreg.copy()
lr_mcc_orig = mcc_logreg.copy()

tree_acc_orig = acc_tree.copy()
tree_prec_orig = prec_tree.copy()
tree_rec_orig = rec_tree.copy()
tree_auroc_orig = aur_tree.copy()
tree_mcc_orig = mcc_tree.copy()

boost_acc_orig = acc_booster.copy()
boost_prec_orig = prec_booster.copy()
boost_rec_orig = rec_booster.copy()
boost_auroc_orig = aur_booster.copy()
pboost_mcc_orig = mcc_booster.copy()

forest_acc_orig = acc_forest.copy()
forest_prec_orig = prec_forest.copy()
forest_rec_orig = rec_forest.copy()
forest_auroc_orig = aur_forest.copy()
forest_mcc_orig = mcc_forest.copy()

##Now repeat entire process for second set of models (includes social data)
Xad2 = readmit_model2.drop(['readmit'], 1) #Features
Yad2 = readmit_model2['readmit'] #Target variable
Xad2.shape, Yad2.shape

acc_logreg = []
prec_logreg = []
rec_logreg = []
aur_logreg = []
mcc_logreg = []

acc_tree = []
prec_tree = []
rec_tree = []
aur_tree = []
mcc_tree = []

acc_booster = []
prec_booster = []
rec_booster = []
aur_booster = []
mcc_booster = []

acc_forest = []
prec_forest = []
rec_forest = []
aur_forest = []
mcc_forest = []


for i in range (0,10):
    X_train2,X_test2,y_train2,y_test2=train_test_split(Xad2,Yad2,test_size=0.25,random_state=i)
    print('Logistic Regression', i)
    logreg1 = modelrun(model =logreg, xtrain=X_train2, ytrain=y_train2, xtest = X_test2, ytest=y_test2)
    logreg_append(logreg1)
    print('')
    print('Decision Tree', i)
    tree1 = modelrun(model =clf_tree, xtrain=X_train2, ytrain=y_train2, xtest = X_test2, ytest=y_test2)
    tree_append(tree1)
    print('')
    print('Boosted Trees', i)
    boost1 = modelrun(model = booster, xtrain=X_train2, ytrain=y_train2, xtest = X_test2, ytest=y_test2)
    boost_append(boost1)
    print('')
    print('Random Forest', i)
    forest1 = modelrun(model =forest, xtrain=X_train2, ytrain=y_train2, xtest = X_test2, ytest=y_test2)
    forest_append(forest1)
    
print("Logistic Regression accuracy:",acc_logreg, plt.plot(acc_logreg, scaley=False))
print("Logistic Regression precision:",prec_logreg, plt.plot(prec_logreg, scaley=False))
print("Logistic Regression recall:",rec_logreg, plt.plot(rec_logreg, scaley=False))
print("Logistic Regression AUROC:",aur_logreg, plt.plot(aur_logreg, scaley=False))
print("Logistic Regression MCC:",mcc_logreg, plt.plot(mcc_logreg, scaley=False)) 

print("Tree Accuracy:",acc_tree, plt.plot(acc_tree, scaley=False))
print("Tree Precision:",prec_tree, plt.plot(prec_tree, scaley=False))
print("Tree Recall:",rec_tree, plt.plot(rec_tree, scaley=False))
print("Tree AUROC:",aur_tree, plt.plot(aur_tree, scaley=False))
print("Tree MCC:",mcc_tree, plt.plot(mcc_tree, scaley=False))

print("Boosted Tree Accuracy:",acc_booster, plt.plot(acc_booster, scaley=False))
print("Boosted Tree Precision:",prec_booster, plt.plot(prec_booster, scaley=False))
print("Boosted Tree Recall:",rec_booster, plt.plot(rec_booster, scaley=False))
print("Boosted Tree AUROC:",aur_booster, plt.plot(aur_booster, scaley=False))
print("Boosted Tree MCC:",mcc_booster, plt.plot(mcc_booster, scaley=False))

print("Forest Accuracy:",acc_forest, plt.plot(acc_forest, scaley=False))
print("Forect Precision:",prec_forest, plt.plot(prec_forest, scaley=False))
print("Forest Recall:",rec_forest, plt.plot(rec_forest, scaley=False))
print("Forest AUROC:",aur_forest, plt.plot(aur_forest, scaley=False))
print("Forest MCC:",mcc_forest, plt.plot(mcc_forest, scaley=False)) 

lr_accuracy_social = acc_logreg.copy()
lr_prec_social = prec_logreg.copy()
lr_rec_social = rec_logreg.copy()
lr_auroc_social = aur_logreg.copy()
lr_mcc_social = mcc_logreg.copy()

tree_accuracy_social = acc_tree.copy()
tree_prec_social = prec_tree.copy()
tree_rec_social = rec_tree.copy()
tree_auroc_social = aur_tree.copy()
tree_mcc_social = mcc_tree.copy()

boost_acc_social = acc_booster.copy()
boost_prec_social = prec_booster.copy()
boost_rec_social = rec_booster.copy()
boost_auroc_social = aur_booster.copy()
pboost_mcc_social = mcc_booster.copy()

forest_acc_social = acc_forest.copy()
forest_prec_social = prec_forest.copy()
forest_rec_social = rec_forest.copy()
forest_auroc_social = aur_forest.copy()
forest_mcc_social = mcc_forest.copy()

##Test models with only one or the other of the social variables to determine importance
##Now repeat entire process for second set of models (includes social data)
vbd = readmit_model2.drop(columns = 'VALUE')
val = readmit_model2.drop(columns = 'visitbyday')

Xad3 = vbd.drop(['readmit'], 1) #Features
Yad3 = vbd['readmit'] #Target variable
Xad3.shape, Yad3.shape

acc_logreg = []
prec_logreg = []
rec_logreg = []
aur_logreg = []
mcc_logreg = []

acc_tree = []
prec_tree = []
rec_tree = []
aur_tree = []
mcc_tree = []

acc_booster = []
prec_booster = []
rec_booster = []
aur_booster = []
mcc_booster = []

acc_forest = []
prec_forest = []
rec_forest = []
aur_forest = []
mcc_forest = []

for i in range (0,10):
    X_train2,X_test2,y_train2,y_test2=train_test_split(Xad3,Yad3,test_size=0.25,random_state=i)
    print('Logistic Regression', i)
    logreg1 = modelrun(model =logreg, xtrain=X_train2, ytrain=y_train2, xtest = X_test2, ytest=y_test2)
    logreg_append(logreg1)
    print('')
    print('Decision Tree', i)
    tree1 = modelrun(model =clf_tree, xtrain=X_train2, ytrain=y_train2, xtest = X_test2, ytest=y_test2)
    tree_append(tree1)
    print('')
    print('Boosted Trees', i)
    boost1 = modelrun(model = booster, xtrain=X_train2, ytrain=y_train2, xtest = X_test2, ytest=y_test2)
    boost_append(boost1)
    print('')
    print('Random Forest', i)
    forest1 = modelrun(model =forest, xtrain=X_train2, ytrain=y_train2, xtest = X_test2, ytest=y_test2)
    forest_append(forest1)
    

print("Logistic Regression accuracy:",acc_logreg, plt.plot(acc_logreg, scaley=False))
print("Logistic Regression precision:",prec_logreg, plt.plot(prec_logreg, scaley=False))
print("Logistic Regression recall:",rec_logreg, plt.plot(rec_logreg, scaley=False))
print("Logistic Regression AUROC:",aur_logreg, plt.plot(aur_logreg, scaley=False))
print("Logistic Regression MCC:",mcc_logreg, plt.plot(mcc_logreg, scaley=False)) 

print("Tree Accuracy:",acc_tree, plt.plot(acc_tree, scaley=False))
print("Tree Precision:",prec_tree, plt.plot(prec_tree, scaley=False))
print("Tree Recall:",rec_tree, plt.plot(rec_tree, scaley=False))
print("Tree AUROC:",aur_tree, plt.plot(aur_tree, scaley=False))
print("Tree MCC:",mcc_tree, plt.plot(mcc_tree, scaley=False))

print("Boosted Tree Accuracy:",acc_booster, plt.plot(acc_booster, scaley=False))
print("Boosted Tree Precision:",prec_booster, plt.plot(prec_booster, scaley=False))
print("Boosted Tree Recall:",rec_booster, plt.plot(rec_booster, scaley=False))
print("Boosted Tree AUROC:",aur_booster, plt.plot(aur_booster, scaley=False))
print("Boosted Tree MCC:",mcc_booster, plt.plot(mcc_booster, scaley=False))

print("Forest Accuracy:",acc_forest, plt.plot(acc_forest, scaley=False))
print("Forect Precision:",prec_forest, plt.plot(prec_forest, scaley=False))
print("Forest Recall:",rec_forest, plt.plot(rec_forest, scaley=False))
print("Forest AUROC:",aur_forest, plt.plot(aur_forest, scaley=False))
print("Forest MCC:",mcc_forest, plt.plot(mcc_forest, scaley=False)) 

lr_accuracy_vbd = acc_logreg.copy()
lr_prec_vbd = prec_logreg.copy()
lr_rec_vbd = rec_logreg.copy()
lr_auroc_vbd = aur_logreg.copy()
lr_mcc_vbd = mcc_logreg.copy()

tree_accuracy_vbd = acc_tree.copy()
tree_prec_vbd = prec_tree.copy()
tree_rec_vbd = rec_tree.copy()
tree_auroc_vbd = aur_tree.copy()
tree_mcc_vbd = mcc_tree.copy()

boost_acc_vbd = acc_booster.copy()
boost_prec_vbd = prec_booster.copy()
boost_rec_vbd = rec_booster.copy()
boost_auroc_vbd = aur_booster.copy()
pboost_mcc_vbd = mcc_booster.copy()

forest_acc_vbd = acc_forest.copy()
forest_prec_vbd = prec_forest.copy()
forest_rec_vbd = rec_forest.copy()
forest_auroc_vbd = aur_forest.copy()
forest_mcc_vbd = mcc_forest.copy()

Xad4 = val.drop(['readmit'], 1) #Features
Yad4 = val['readmit'] #Target variable
Xad4.shape, Yad4.shape

acc_logreg = []
prec_logreg = []
rec_logreg = []
aur_logreg = []
mcc_logreg = []

acc_tree = []
prec_tree = []
rec_tree = []
aur_tree = []
mcc_tree = []

acc_booster = []
prec_booster = []
rec_booster = []
aur_booster = []
mcc_booster = []

acc_forest = []
prec_forest = []
rec_forest = []
aur_forest = []
mcc_forest = []

for i in range (0,10):
    X_train2,X_test2,y_train2,y_test2=train_test_split(Xad4,Yad4,test_size=0.25,random_state=i)
    print('Logistic Regression', i)
    logreg1 = modelrun(model =logreg, xtrain=X_train2, ytrain=y_train2, xtest = X_test2, ytest=y_test2)
    logreg_append(logreg1)
    print('')
    print('Decision Tree', i)
    tree1 = modelrun(model =clf_tree, xtrain=X_train2, ytrain=y_train2, xtest = X_test2, ytest=y_test2)
    tree_append(tree1)
    print('')
    print('Boosted Trees', i)
    boost1 = modelrun(model = booster, xtrain=X_train2, ytrain=y_train2, xtest = X_test2, ytest=y_test2)
    boost_append(boost1)
    print('')
    print('Random Forest', i)
    forest1 = modelrun(model =forest, xtrain=X_train2, ytrain=y_train2, xtest = X_test2, ytest=y_test2)
    forest_append(forest1)
    

print("Logistic Regression accuracy:",acc_logreg, plt.plot(acc_logreg, scaley=False))
print("Logistic Regression precision:",prec_logreg, plt.plot(prec_logreg, scaley=False))
print("Logistic Regression recall:",rec_logreg, plt.plot(rec_logreg, scaley=False))
print("Logistic Regression AUROC:",aur_logreg, plt.plot(aur_logreg, scaley=False))
print("Logistic Regression MCC:",mcc_logreg, plt.plot(mcc_logreg, scaley=False)) 

print("Tree Accuracy:",acc_tree, plt.plot(acc_tree, scaley=False))
print("Tree Precision:",prec_tree, plt.plot(prec_tree, scaley=False))
print("Tree Recall:",rec_tree, plt.plot(rec_tree, scaley=False))
print("Tree AUROC:",aur_tree, plt.plot(aur_tree, scaley=False))
print("Tree MCC:",mcc_tree, plt.plot(mcc_tree, scaley=False))

print("Boosted Tree Accuracy:",acc_booster, plt.plot(acc_booster, scaley=False))
print("Boosted Tree Precision:",prec_booster, plt.plot(prec_booster, scaley=False))
print("Boosted Tree Recall:",rec_booster, plt.plot(rec_booster, scaley=False))
print("Boosted Tree AUROC:",aur_booster, plt.plot(aur_booster, scaley=False))
print("Boosted Tree MCC:",mcc_booster, plt.plot(mcc_booster, scaley=False))

print("Forest Accuracy:",acc_forest, plt.plot(acc_forest, scaley=False))
print("Forect Precision:",prec_forest, plt.plot(prec_forest, scaley=False))
print("Forest Recall:",rec_forest, plt.plot(rec_forest, scaley=False))
print("Forest AUROC:",aur_forest, plt.plot(aur_forest, scaley=False))
print("Forest MCC:",mcc_forest, plt.plot(mcc_forest, scaley=False)) 

#Check distribution to determine if parametric or non-parametric tests should be used
plt.hist(lr_acc_orig)
#distributions are non-normal so should use a non-parametric test

#Define a function that runs the required parametric test
def wilcox(d1, d2):
    stat, p = wilcoxon(d1, d2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    print(statistics.mean(d1), statistics.mean(d2))

##Compare results of the models to see if they perform differently
wilcox(lr_acc_orig, tree_acc_orig)
wilcox(lr_acc_orig, boost_acc_orig)
wilcox(lr_acc_orig, forest_acc_orig)
wilcox(tree_acc_orig, boost_acc_orig)
wilcox(tree_acc_orig, forest_acc_orig)
wilcox(boost_acc_orig, forest_acc_orig)
#All models behave differently with the exception of logistic regression and
#random forest - for these models, check the other metric
print("Precision")
wilcox(lr_prec_orig, forest_prec_orig)
print("Recall")
wilcox(lr_rec_orig, forest_rec_orig)
print("AUROC")
wilcox(lr_auroc_orig, forest_auroc_orig)
print("MCC")
wilcox(lr_mcc_orig, forest_mcc_orig)
##Models perform differently in terms of precision, recall, and AUROC

##Compare original models with models including social variables
#Accuracy
print("LR")
wilcox(lr_acc_orig, lr_accuracy_social)

print("Tree")
wilcox(tree_acc_orig, tree_accuracy_social)

print("Boosted Trees")
wilcox(boost_acc_orig, boost_acc_social)

print("Random Forest")
wilcox(forest_acc_orig, forest_acc_social)

##Compare original models with models including social variables
#Precision
print("LR")
wilcox(lr_prec_orig, lr_prec_social)

print("Tree")
wilcox(tree_prec_orig, tree_prec_social)

print("Boosted Trees")
wilcox(boost_prec_orig, boost_prec_social)

print("Random Forest")
wilcox(forest_prec_orig, forest_prec_social)

##Compare original models with models including social variables
#Recall
print("LR")
wilcox(lr_rec_orig, lr_rec_social)

print("Tree")
wilcox(tree_rec_orig, tree_rec_social)

print("Boosted Trees")
wilcox(boost_rec_orig, boost_rec_social)

print("Random Forest")
wilcox(forest_rec_orig, forest_rec_social)

##Compare original models with models including social variables
#AUROC
print("LR")
wilcox(lr_auroc_orig, lr_auroc_social)

print("Tree")
wilcox(tree_auroc_orig, tree_auroc_social)

print("Boosted Trees")
wilcox(boost_auroc_orig, boost_auroc_social)

print("Random Forest")
wilcox(forest_auroc_orig, forest_auroc_social)

##Compare original models with models including social variables
#MCC
print("LR")
wilcox(lr_mcc_orig, lr_mcc_social)

print("Tree")
wilcox(tree_mcc_orig, tree_mcc_social)

print("Boosted Trees")
wilcox(pboost_mcc_orig, pboost_mcc_social)

print("Random Forest")
wilcox(forest_mcc_orig, forest_mcc_social)

##With the exception of logistic regression, all models
#perform better with social variables included, consitent
#across all metrics

##Compare models with VALUE and visitbyday individually
#Accuracy
print("LR")
wilcox(lr_acc_orig, lr_accuracy_vbd)

print("Tree")
wilcox(tree_acc_orig, tree_accuracy_vbd)

print("Boosted Trees")
wilcox(boost_acc_orig, boost_acc_vbd)

print("Random Forest")
wilcox(forest_acc_orig, forest_acc_vbd)

##Compare models with VALUE and visitbyday individually
#Accuracy
print("LR")
wilcox(lr_accuracy_social, lr_accuracy_vbd)

print("Tree")
wilcox(tree_accuracy_social, tree_accuracy_vbd)

print("Boosted Trees")
wilcox(boost_acc_social, boost_acc_vbd)

print("Random Forest")
wilcox(forest_acc_social, forest_acc_vbd)

##Compare models with VALUE and visitbyday individually
#Accuracy
print("LR")
wilcox(lr_accuracy_social, acc_logreg)

print("Tree")
wilcox(tree_accuracy_social, acc_tree)

print("Boosted Trees")
wilcox(boost_acc_social, acc_booster)

print("Random Forest")
wilcox(forest_acc_social, acc_forest)

#Fairly simlar across all models, with some slight superior performance with
#both variables
#Next steps could be to test the rest of the measures and choose one of the variables