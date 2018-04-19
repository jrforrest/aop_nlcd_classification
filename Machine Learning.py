# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 09:34:37 2018

@author: kmurphy
"""

import pandas as pd
import numpy as np
import gdal, osr
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# Import training data as dataframes

inPath = "A:/TOS/TOS_Workspace/kmurph/Spectro Data/Clean_Spectral/"

DF_training = pd.read_csv(inPath + "DF_training.csv")
EF_training = pd.read_csv(inPath + "EF_training.csv")
DHI_training = pd.read_csv(inPath + "DHI_training.csv")
PH_training = pd.read_csv(inPath + "PH_training.csv")
OW_training = pd.read_csv(inPath + "OW_training.csv")
SS_training = pd.read_csv(inPath + "SS_training.csv")

DF_training = DF_training.drop(columns = ('Unnamed: 0'))
EF_training = EF_training.drop(columns = ('Unnamed: 0'))
DHI_training = DHI_training.drop(columns = ('Unnamed: 0'))
PH_training = PH_training.drop(columns = ('Unnamed: 0'))
OW_training = OW_training.drop(columns = ('Unnamed: 0'))
SS_training = SS_training.drop(columns = ('Unnamed: 0'))

# Randomly separate ~ 60% of the total points to be training data
# This leaves ~20% to be testing data and ~20% for cross-validation
DF_training['is_train'] = np.random.uniform(0,1,len(DF_training)) <= 0.6
EF_training['is_train'] = np.random.uniform(0,1,len(EF_training)) <= 0.6
DHI_training['is_train'] = np.random.uniform(0,1,len(DHI_training)) <= 0.6
PH_training['is_train'] = np.random.uniform(0,1,len(PH_training)) <= 0.6
OW_training['is_train'] = np.random.uniform(0,1,len(OW_training)) <= 0.6
SS_training['is_train'] = np.random.uniform(0,1,len(SS_training)) <= 0.6

# Separate into training rows and test/cv rows
DF_train = DF_training[DF_training['is_train'] == True]
DF_check = DF_training[DF_training['is_train'] == False]

EF_train = EF_training[EF_training['is_train'] == True]
EF_check = EF_training[EF_training['is_train'] == False]

DHI_train = DHI_training[DHI_training['is_train'] == True]
DHI_check = DHI_training[DHI_training['is_train'] == False]

PH_train = PH_training[PH_training['is_train'] == True]
PH_check = PH_training[PH_training['is_train'] == False]

OW_train = OW_training[OW_training['is_train'] == True]
OW_check = OW_training[OW_training['is_train'] == False]

SS_train = SS_training[SS_training['is_train'] == True]
SS_check = SS_training[SS_training['is_train'] == False]

# Break up check into testing and cross-validation
DF_test = DF_check.sample(frac = 0.5)
EF_test = EF_check.sample(frac = 0.5)
DHI_test = DHI_check.sample(frac = 0.5)
PH_test = PH_check.sample(frac = 0.5)
OW_test = OW_check.sample(frac = 0.5)
SS_test = SS_check.sample(frac = 0.5)

DF_cv = DF_check.loc[~DF_check.index.isin(DF_test.index)]
EF_cv = EF_check.loc[~EF_check.index.isin(EF_test.index)]
DHI_cv = DHI_check.loc[~DHI_check.index.isin(DHI_test.index)]
PH_cv = PH_check.loc[~PH_check.index.isin(PH_test.index)]
OW_cv = OW_check.loc[~OW_check.index.isin(OW_test.index)]
SS_cv = SS_check.loc[~SS_check.index.isin(SS_test.index)]

# Create a list of feature names
nlcd_features = DF_training.columns[2:34]
print(nlcd_features)

# Combin all training, check, and cross validation data into a single file for each
nlcd_training = pd.concat([DF_training, EF_training, DHI_training, PH_training, OW_training, SS_training])
nlcd_test = pd.concat([DF_test, EF_test, DHI_test, PH_test, OW_test, SS_test])
nlcd_cv = pd.concat([DF_cv, EF_cv, DHI_cv, PH_cv, OW_cv, SS_cv])


# Drop 'is_train' column from nlcd_training, test, and cv
nlcd_training = nlcd_training.drop(columns = ('is_train'))
nlcd_test = nlcd_test.drop(columns = ('is_train'))
nlcd_cv = nlcd_cv.drop(columns = ('is_train'))

# Drop rows if there are nan values
nlcd_training = nlcd_training.dropna()
nlcd_test = nlcd_test.dropna()
nlcd_cv = nlcd_cv.dropna()

# nlcd_training['NLCD'] contains the actual land cover names. Before we can use it,
# we need to convert each land cover name into a digit. So, in this case there
# are 6 species, which have been coded as 0, 1, 2, 3, 4, or 5.
nlcd_digit = pd.factorize(nlcd_training['NLCD'])[0]

# Create a random forest Classifier. By convention, clf means 'Classifier'
#clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators = 50)
clf = RandomForestClassifier(n_jobs=2, n_estimators = 50)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(nlcd_training[nlcd_features], nlcd_digit)

# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
pred = clf.predict(nlcd_test[nlcd_features])

# View the predicted probabilities of the first 10 observations
clf.predict_proba(nlcd_test[nlcd_features])[0:10]

# Create actual english names for the plants for each predicted plant class
#nlcd_training['NLCD'] = pd.Categorical.from_codes(nlcd_training.target, nlcd_training.target_names)

nlcd_names = nlcd_training['NLCD']

preds = nlcd_names[clf.predict(nlcd_test[nlcd_features])]
preds = 

#if preds[[0]] == 0:
#    preds[[1]] = 'DF'

# View the PREDICTED species for the first five observations
preds[0:5]

pd.crosstab(nlcd_test['NLCD'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])

# View a list of the features and their importance scores
list(zip(nlcd_train[nlcd_features], clf.feature_importances_))
