# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 20:43:36 2016

@author: can
"""

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import scipy.sparse as sp
import math
from matplotlib.pyplot import hist

products = pd.read_csv('input/products_features.csv')
sample_train = pd.read_csv('input/train_sample_1000000.csv')
sample_test = pd.read_csv('input/test.csv')

result = pd.concat([products, sample_train], axis=1, join='inner')
#print result.head()
result.fillna(0,inplace=True)
testdata = pd.concat([products, sample_test], axis=1, join='outer')
testdata.fillna(0,inplace=True)


print(list(result.columns.values))

result = result.drop(result.columns[[0,27,28,34,35,36,37]], axis=1)

print(list(result.columns.values))

#train_df = pd.read_csv("train.csv")
 
et = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)
 
columns = ['grams', 'ml', 'inches', 'pct', 'pieces', 'bim', 'mla', 'mta', 'tab', 'pan', 'mtb', 'lar', 'gbi', 'won', 'duo', 'tubo', 'tr', 'cu', 'sp', 'prom', 'fresa', 'dh', 'vainilla', 'deliciosas', 'blanco', 'chocolate']
 
labels = result["Demanda_uni_equil"].values
features = result[list(columns)].values
test_features = testdata[list(columns)].values

#print("features obtained")
#
#n_feat = [xs for xs in features if not any(math.isnan(x) for x in xs)]
#
#
#X = sp.csc_matrix(n_feat)
#imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
#imp.fit(X)
#
#print("sparse created")
#
#imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
#imp.fit(cleanedList)
 
#test_df = replace_non_numeric(pd.read_csv("test.csv"))
#et.fit(features, labels)
#print et.predict(imp.transform(test_df[columns].values))

 
#et_score = cross_val_score(et, features, labels, n_jobs=-1).mean()

# Create linear regression object
regr = linear_model.LinearRegression()
print("so far ok")
# Train the model using the training sets
our_wonderful_model = regr.fit(features, labels) 
 
prediction = regr.predict(test_features)
pred_int = prediction.astype(int)
pred_df = pd.DataFrame({'Demanda_uni_equil': pred_int})

pred_df = pred_df.fillna(value=6)
pred_df_nozeros = pred_df.replace(to_replace=0,value=2) 
pred_df_nozeros = pred_df.replace(to_replace=-1,value=2) 


#pred_df['Demanda_uni_equil'] = 2
#result = pd.concat([sample_test['id'], pred_df], axis=1)
#result.to_csv("input/submit_2222.csv", index=False)

result = pd.concat([sample_test['id'], pred_df_nozeros], axis=1)
result.to_csv("input/submit4.csv", index=False)

