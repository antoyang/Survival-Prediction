"""
Cox-Ph regression. Baseline.
Leon Zheng
"""

import pandas as pd
from Coxph_regression import CoxPhRegression

"""
Reading data
"""
# Read training set
radiomics_train = pd.read_csv("data/train/features/radiomics.csv", index_col=0, usecols=[0, 5, 32, 39, 6, 3])
clinical_train = pd.read_csv("data/train/features/clinical_data.csv", index_col=0, usecols=[0, 3])
# added_features = pd.read_csv("data/train/features/train_extracted_features.csv", index_col='patient_number', usecols=[])
# print(added_features)
input_train = pd.concat([radiomics_train, clinical_train], axis=1)
output_train = pd.read_csv("data/train/y_train.csv", index_col=0, header=0)

# Read testing set
radiomics_test = pd.read_csv("data/test/features/radiomics.csv", index_col=0, usecols=[0, 3, 5, 32, 39, 6])
clinical_test = pd.read_csv("data/test/features/clinical_data.csv", index_col=0, usecols=[0, 3])
# dum = pd.get_dummies(clinical_test['SourceDataset'], prefix='SourceDataset')
# print(dum)
input_test = pd.concat([radiomics_test, clinical_test['Nstage']], axis=1)

""" 
Grid search
"""
from sklearn.model_selection import GridSearchCV

tuned_params = {"alpha": [0.01, 0.1, 1, 10],
                "threshold": [0.2, 0.4, 0.6, 0.8]}
grid_search = GridSearchCV(CoxPhRegression(), tuned_params, cv=5, n_jobs=4)
grid_search.fit(input_train, output_train)
print(grid_search.best_score_)
best_params = grid_search.best_params_
print(best_params)

"""
Create submission
"""
# Create submission
coxph = CoxPhRegression()
coxph.set_params(**best_params)
coxph.fit(input_train, output_train)
y_pred = coxph.predict(input_test)
y_pred.to_csv('submission.csv')