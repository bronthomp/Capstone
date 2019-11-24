# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:40:38 2019

@author: BRTHOMPSON
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

readmit = pd.read_csv(r'C:\Users\brthompson\bt_school\data\interim\firstmodel_cleaned.csv')
readmit.head()

#Check balance - do not need to balance this dataset
readmit['readmit'].value_counts()

#Separate out variable names from values
array = readmit.values
Xad = array[:, 1:]
Yad = array[:,0]

#split the data into training and test 
X_train,X_test,y_train,y_test=train_test_split(Xad,Yad,test_size=0.25,random_state=0)

###Feature Selection
#Begin with chi-square
tester = SelectKBest(score_func=chi2, k=4)
fit = tester.fit(X_train, y_train)

np.set_printoptions(precision=3)
print(fit.scores_)

#In order to interpret scores, append them to a dataframe with the column names
readmit_chisq= pd.DataFrame()
readmit_chisq['scores'] = fit.scores_
readmit_chisq['names'] = readmit.columns[1:]

readmit_chisq.sort_values(by = 'scores', ascending = False)
readmit_chisq['scores'].mean()
readmit_chisq['scores'].median()
keep_readmit_chisq = readmit_chisq[readmit_chisq['scores']>readmit_chisq['scores'].median()]

keep = list(keep_readmit_chisq['names'])
readmit_chi = readmit[keep]

#Now perform Recursive Feature Elimination on the variables selected 
Xch = readmit_chi.values
Ych = Yad

#Kept half of the remaining columns - accuracies for LR, DT and RF were ~70.5%
##Try again with more variables -keep 125
model = LogisticRegression()
rfe = RFE(model, 125)
fit = rfe.fit(Xch, Ych)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

readmit_rfe = pd.DataFrame()
readmit_rfe['include'] = fit.support_
readmit_rfe['names'] = readmit_chi.columns
readmit_rfe

keep_readmit_rfe = readmit_rfe[readmit_rfe['include']==True]
keep_readmit_rfe['names']

readmit_cols = list(keep_readmit_rfe['names'])
readmit_cols
readmit_keep1 = readmit_cols.copy()
readmit_keep1
readmit_keep1.append('readmit')
readmit_keep1
readmit_model1 = readmit[readmit_keep1]

readmit_model1.to_csv(r'C:\Users\brthompson\bt_school\data\interim\readmit_model1.csv', index=False)

newvals = ['VALUE', 'visitbyday', 'readmit']

readmit_cols2 = readmit_cols + newvals
readmit_cols2

readmit_2 = pd.read_csv(r'C:\Users\brthompson\bt_school\data\interim\secondmodel_cleaned.csv')
read_model2 = readmit_2[readmit_cols2]
read_model2.head()

read_model2.to_csv(r'C:\Users\brthompson\bt_school\data\interim\secondmodel_readmit.csv')
