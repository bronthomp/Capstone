# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:42:07 2019

@author: BRTHOMPSON
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

##This code imports the merged dataset created in Capstone-MergeData and prepares it for analysis

#import merged dataset and examine it
merged = pd.read_csv(r'C:\Users\brthompson\bt_school\data\interim\merged.csv')
merged.shape
#Investigate data types
merged.dtypes

#Dates have been imported as objects. Convert necessary date fields to the appropriate datatype
merged['admitdate'] = pd.to_datetime(merged['ADMITTIME'], format='%Y-%m-%d')
merged['dischdate'] = pd.to_datetime(merged['DISCHTIME'], format='%Y-%m-%d')
merged['dobdate'] = pd.to_datetime(merged['DOB'], format='%Y-%m-%d')
merged.dtypes

#Create 2 new variables - age in years and length of stay in days - by subtracting dates
merged['age']=(merged['admitdate'].dt.date - merged['dobdate'].dt.date)
merged['age'] = merged['age'].astype('timedelta64[D]').dt.days
merged['age'] = merged['age']/365

merged['los'] = (merged['dischdate'].dt.date - merged['admitdate'].dt.date)
merged['los'] = merged['los'].dt.days

#Create another variable that divides the VALUE field (total number of visitors a patient received) by the total number of days
merged['visitbyday'] = merged['VALUE']/merged['los']

#Create an additional variable that flags whether there was a readmission - if the patient has another subsequent record. 
#This is the classifier of interest
merged = merged.sort_values(['SUBJECT_ID', 'admitdate'])
merged['readmit'] = np.where(merged['SUBJECT_ID'].shift(-1)==merged['SUBJECT_ID'], 1, 0)
merged['readmit'].value_counts()

#check data types to ensure all were created properly
merged.dtypes

#Remove anyone with a LOS < 1 day since that will result in infinite visits per day
#Change name of dataframe in case need to go back to version with all records and columns
#Keep track of changes using the shape function
merged.shape
refined = merged[merged['los']!=0]
refined.shape

#Remove any records that do not have chart events data since they will not have information on visitors
refined = refined[refined['HAS_CHARTEVENTS_DATA']==1]
refined.shape

#Remove any records with age<0
refined = refined[refined['age']>=0]

#Examine the VALUE and visitperday attributes to look for outliers
refined['VALUE'].max()
plt.subplot(211)
hist, bins, _ = plt.hist(np.log((refined['VALUE']+0.000000001)), bins=100)
refined['VALUE'].mean()
refined['VALUE'].std()
plt.subplot(211)
hist, bins, _ = plt.hist(np.log((refined['visitbyday']+0.000000001)), bins=100)

refined['visitbyday'].mean()
refined['visitbyday'].median()
refined['visitbyday'].std()

#Outliers those with more than 50 visits per day. Remove outliers after
#filling NA with 0 (since VALUE and visitby day are created variables, absence
#of these variables means there were no vists/social support presence). 
refined['VALUE'] = np.where(refined['VALUE'].isna(), '0', refined['VALUE'])
refined['visitbyday'] = np.where(refined['visitbyday'].isna(), '0', refined['visitbyday'])

refined['VALUE'] = refined['VALUE'].astype(float)
refined['visitbyday'] = refined['visitbyday'].astype(float)

#Now remove outliers, indicated by a visitbyday of over 50
print(refined.shape)
refined = refined[refined['visitbyday']<=50]
refined.shape

#Remove 'HAS_CHARTEVENTS_DATA' since it no longer captures any variability
#Remove ID variable since will not contribute to the model
##REMOVE ALL DATE VARIABLES, have captured all necessary information under age and LOS
#Can also remove EXPIRE_FLAG and HOSPITAL_EXPIRE_FLAG
refined = refined.drop(['admitdate', 'dischdate', 'dobdate', 'SUBJECT_ID', 'HADM_ID', 'EDREGTIME', 'EDOUTTIME', 'DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN', 'EXPIRE_FLAG', 'DEATHTIME','HAS_CHARTEVENTS_DATA', 'ADMITTIME', 'DISCHTIME', 'HOSPITAL_EXPIRE_FLAG'], 1)

##Examine other variables for missing
refined.isna().sum()

refined['LANGUAGE'].value_counts()
#Group missing Language under "UNKNOWN"
refined['LANGUAGE'] = np.where(refined['LANGUAGE'].isna(), 'UNKNOWN', refined['LANGUAGE'])
refined['LANGUAGE'].value_counts()

refined['RELIGION'].value_counts()
#Include missing Religions under "NOT SPECIFIED"
refined['RELIGION'] = np.where(refined['RELIGION'].isna(), 'NOT SPECIFIED', refined['RELIGION'])
refined['RELIGION'].value_counts()

refined['MARITAL_STATUS'].value_counts()
#Include missing marital status under "UNKNOWN (DEFAULT)"
refined['MARITAL_STATUS'] = np.where(refined['MARITAL_STATUS'].isna(), 'UNKNOWN (DEFAULT)', refined['MARITAL_STATUS'])
refined['MARITAL_STATUS'].value_counts()

refined['DIAGNOSIS'].value_counts()
#Remove rows that are missing the diagnosis variable
refined.shape
refined = refined[pd.notnull(refined['DIAGNOSIS'])]
refined.shape

##EXAMINE ALL CATEGORICAL VARIABLES
#For all categorical variables, any category with n<25 will be grouped under "Other" to reduce dimensionality
#create a function that will compile categories with less than x entries into "Other"
def reducedim(var, df=refined, x=25):
    v=df[var].value_counts()
    print(v)
    vf = v[v<x]
    vi = vf.index.to_list()
    print(len(df[df[var].isin(vi)]))
    df[var] = np.where(df[var].isin(vi), 'OTHER', df[var])
    print(df[var].value_counts())

refined['ADMISSION_TYPE'].value_counts()
refined['ADMISSION_LOCATION'].value_counts()
refined['GENDER'].value_counts()
reducedim(var='ADMISSION_LOCATION', df=refined, x=25)
refined['DISCHARGE_LOCATION'].value_counts()
reducedim(var='DISCHARGE_LOCATION')
refined['INSURANCE'].value_counts()
refined['LANGUAGE'].value_counts()
reducedim(var='LANGUAGE')
refined['MARITAL_STATUS'].value_counts()
reducedim(var='MARITAL_STATUS')
refined['RELIGION'].value_counts()
reducedim(var='RELIGION')
refined['DIAGNOSIS'].value_counts()
reducedim(var='DIAGNOSIS')

#reorder variables and send to csv
refined = refined[['readmit', 'GENDER', 'VALUE', 'ADMISSION_TYPE', 'ADMISSION_LOCATION',
       'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION',
       'MARITAL_STATUS', 'ETHNICITY', 'DIAGNOSIS', 'age', 'los', 'visitbyday'
       ]]
refined.to_csv(r'C:\Users\brthompson\bt_school\data\interim\refined.csv', index=False)

#Now create dataframe of dummy variables ready to perform feature selection
#Since the VALUE and visitbyday are currently included, this will be the dataset for the second model
secondmodel_readmit = pd.get_dummies(refined, columns = ['GENDER', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE',
       'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'DIAGNOSIS'])
#Create second version of dummies dataset to use for initial models. This will not include the data on visits -VALUE and visitbyday variables
firstmodel_readmit = secondmodel_readmit.drop(['VALUE', 'visitbyday'], 1)

#Send all to csv for ease of tracking
secondmodel_readmit.to_csv(r'C:\Users\brthompson\bt_school\data\interim\secondmodel_cleaned.csv', index=False)
firstmodel_readmit.to_csv(r'C:\Users\brthompson\bt_school\data\interim\firstmodel_cleaned.csv', index=False)





