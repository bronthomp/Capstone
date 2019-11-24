# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:08:50 2019

@author: BRTHOMPSON
"""
import pandas as pd



##This code imports the required datasets, transforms and merges them

#Import csv's for 3 required datasets
social = pd.read_csv(r'C:\Users\brthompson\bt_school\data\raw\social_support.csv', usecols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'CGID', 'VALUE'])
pat = pd.read_csv(r'C:\Users\brthompson\bt_school\data\raw\PATIENTS.csv')
admit = pd.read_csv(r'C:\Users\brthompson\bt_school\data\raw\ADMISSIONS.csv')

#Examine the datasets
social.head()
pat.head()
admit.head()

#Drop ROW_ID column from pat and admit tables so that they do not interfere with merging - these are unique to the table rows
pat = pat.drop('ROW_ID', 1)
admit = admit.drop('ROW_ID', 1)

#For social dataset, interested in the number of visitors they have, indicated by the number of rows in the social dataset. 
#Reshape this dataset to count number of visitors per patient stay (SUBJECT_ID and HADM_ID)
#Need only keep ID variables and Value count
soc = social.groupby(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']).count()
soc = soc.drop(['ITEMID','CHARTTIME','CGID'],1)

#Merge the 3 datasets together by SUBJECT_ID
#Outer joins required since not all patients have social information
socpat = pd.merge(pat, soc, how = 'outer', on = ['SUBJECT_ID'])
merged = pd.merge(socpat, admit, how = 'outer', on = ['SUBJECT_ID'])
merged.head()

#Send to csv to be used in next code
merged.to_csv(r'C:\Users\brthompson\bt_school\data\interim\merged.csv', index=False)