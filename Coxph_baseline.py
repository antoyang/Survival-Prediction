"""
Cox-Ph regression. Baseline.
Leon Zheng
"""

import pandas as pd
from Coxph_regression import CoxPhRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

"""
Feature selection
"""
radiomics_features = ['PatientID', 'original_shape_Sphericity', 'original_shape_SurfaceVolumeRatio',
                      'original_shape_Maximum3DDiameter', 'original_glcm_JointEntropy', 'original_glcm_Id',
                      'original_glcm_Idm']
clinical_features = ['PatientID', 'SourceDataset', 'Nstage']

"""
Reading data
"""
# Read training set
radiomics_train = pd.read_csv("data/train/features/radiomics.csv", index_col=0, usecols = radiomics_features)
clinical_train = pd.read_csv("data/train/features/clinical_data.csv", index_col=0, usecols = clinical_features)
clinical_train.SourceDataset = pd.factorize(clinical_train.SourceDataset)[0]
input_train = pd.concat([radiomics_train, clinical_train], axis=1)
print(input_train)
output_train = pd.read_csv("data/train/y_train.csv", index_col=0, header=0)
print(output_train)

# Read testing set
radiomics_test = pd.read_csv("data/test/features/radiomics.csv", index_col=0, usecols= radiomics_features)
clinical_test = pd.read_csv("data/test/features/clinical_data.csv", index_col=0, usecols= clinical_features)
clinical_test.SourceDataset = pd.factorize(clinical_test.SourceDataset)[0]
input_test = pd.concat([radiomics_test, clinical_test], axis=1)

""" 
Grid search
"""
tuned_params = {"alpha": np.logspace(-8, -3, 10),
                "threshold": np.linspace(0.85, 0.95, 10)}
grid_search = GridSearchCV(CoxPhRegression(), tuned_params, cv=5, n_jobs=4)
grid_search.fit(input_train, output_train)
print(grid_search.best_score_)
best_params = grid_search.best_params_
print(best_params)

"""
Create submission
"""
# Create submission
coxph = CoxPhRegression(**best_params)
coxph.fit(input_train, output_train)
y_pred = coxph.predict(input_test)
y_pred.to_csv('submission.csv')
