"""
Regression Baseline
Leon Zheng
"""

import pandas as pd
from sklearn import linear_model, model_selection, preprocessing
from sklearn.metrics import mean_squared_error
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
import numpy as np
from metrics import cindex


radiomics = pd.read_csv("data/train/features/radiomics.csv", index_col=0, usecols=[0, 5, 32, 39, 6])
# print(radiomics)
clinical = pd.read_csv("data/train/features/clinical_data.csv", index_col=0, usecols=[0, 3])
# print(clinical.head())
input_train = pd.concat([radiomics, clinical], axis=1)
# print(input_train.head())
output_train = pd.read_csv("data/train/y_train.csv", index_col=0, header=0)

# """ Linear Regression """
# linear_reg = linear_model.LinearRegression()
# linear_reg.fit(X_train, Y_train[:, 0])
# Y_pred = linear_reg.predict(X_test)
# print(f'Mean squared error: {mean_squared_error(Y_test[:, 0], Y_pred)}')
# print(concordance_index_censored(Y_test[:, 1].astype('bool'), Y_test[:, 0], Y_pred))

""" CoxPH Regression """
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(input_train, output_train, test_size=0.21)
print(X_train)
print(Y_train)

# Training
strct_Y_train = Surv.from_dataframe('Event', 'SurvivalTime', Y_train)
print(strct_Y_train)
coxph_reg = CoxPHSurvivalAnalysis(alpha=1e3)
coxph_reg.fit(X_train, strct_Y_train)

# Prediction
prediction = coxph_reg.predict(X_test)
prediction = 100 * (prediction - np.min(prediction) + 1e-3)
Y_pred = pd.DataFrame(np.array([[y, np.nan] for y in prediction]), index=X_test.index, columns=['SurvivalTime', 'Event'])
print(Y_pred)

# Score
print(cindex(Y_test, Y_pred))

""" Prediction """
strct_Y_data = Surv.from_arrays(Y_data[:, 1].astype('bool'), Y_data[:, 0])
print(strct_Y_data)
# coxph_reg.fit(X_data, strct_Y_data)
# input_test = pd.read_csv("data/test/features/radiomics.csv", index_col=0, header=[0,1])
# X_data_test = input_test.to_numpy()
# X_data_test = preprocessing.scale(X_data_test)
# Y_pred = coxph_reg.predict(X_data_test)
# print(Y_pred)
#
# # Write submission
# submission = pd.DataFrame([[y, 'nan'] for y in Y_pred], columns=['SurvivalTime', 'Event'], index=input_test.index)
# print(submission.head())
# submission.to_csv('submission.csv')