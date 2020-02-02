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


radiomics_train = pd.read_csv("data/train/features/radiomics.csv", index_col=0, usecols=[0, 5, 32, 39, 6])
clinical_train = pd.read_csv("data/train/features/clinical_data.csv", index_col=0, usecols=[0, 3])
input_train = pd.concat([radiomics_train, clinical_train], axis=1)
output_train = pd.read_csv("data/train/y_train.csv", index_col=0, header=0)

# """ Linear Regression """
# linear_reg = linear_model.LinearRegression()
# linear_reg.fit(X_train, Y_train[:, 0])
# Y_pred = linear_reg.predict(X_test)
# print(f'Mean squared error: {mean_squared_error(Y_test[:, 0], Y_pred)}')
# print(concordance_index_censored(Y_test[:, 1].astype('bool'), Y_test[:, 0], Y_pred))

# """ CoxPH Regression """
# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(input_train, output_train, test_size=0.21)
# print(X_train)
# print(Y_train)
#
# # Training
# strct_Y_train = Surv.from_dataframe('Event', 'SurvivalTime', Y_train)
# print(strct_Y_train)
# coxph_reg = CoxPHSurvivalAnalysis(alpha=10)
# coxph_reg.fit(X_train, strct_Y_train)
#
# # Prediction
# prediction = coxph_reg.predict_survival_function(X_test)
# threshold = 0.5
# y_pred = []
# for pred in prediction:
#     time = pred.x
#     survival_prob = pred.y
#     i_pred = 0
#     while i_pred < len(survival_prob) - 1 and survival_prob[i_pred] > threshold:
#         i_pred += 1
#     print(time[i_pred])
#     y_pred.append(time[i_pred])
# Y_pred = pd.DataFrame(np.array([[y, np.nan] for y in y_pred]), index=X_test.index, columns=['SurvivalTime', 'Event'])
# print(Y_pred)
#
# # Score
# print(cindex(Y_test, Y_pred))

""" Prediction """
# Test set
radiomics_test = pd.read_csv("data/test/features/radiomics.csv", index_col=0, usecols=[0, 3, 5, 32, 39, 6])
clinical_test = pd.read_csv("data/test/features/clinical_data.csv", index_col=0, usecols=[0, 3])
input_test = pd.concat([radiomics_test, clinical_test], axis=1)
print(input_test)

# Training on the training set
strct_Y_train = Surv.from_dataframe('Event', 'SurvivalTime', output_train)
# print(strct_Y_train)
coxph_reg = CoxPHSurvivalAnalysis(alpha=10)
coxph_reg.fit(input_train, strct_Y_train)

# Prediction
prediction = coxph_reg.predict_survival_function(input_test)
threshold = 0.55
y_pred = []
for pred in prediction:
    time = pred.x
    survival_prob = pred.y
    i_pred = 0
    while i_pred < len(survival_prob) - 1 and survival_prob[i_pred] > threshold:
        i_pred += 1
    print(time[i_pred])
    y_pred.append(time[i_pred])
submission = pd.DataFrame(np.array([[y, np.nan] for y in y_pred]), index=input_test.index, columns=['SurvivalTime', 'Event'])
print(submission)

# Write submission
submission.to_csv('submission.csv')