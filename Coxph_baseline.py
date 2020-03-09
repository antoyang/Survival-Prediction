"""
Cox-Ph regression. Baseline.
Leon Zheng
"""

import pandas as pd
from Coxph_regression import CoxPhRegression
from sklearn.model_selection import GridSearchCV
import read_data
import numpy as np

"""
Feature selection
"""
radiomics_features = ['original_shape_Sphericity', 'original_shape_SurfaceVolumeRatio',
                      'original_shape_Maximum3DDiameter', 'original_glcm_JointEntropy', 'original_glcm_Id',
                      'original_glcm_Idm']
clinical_features = ['SourceDataset_l1', 'Nstage']
features = radiomics_features + clinical_features

"""
Reading data
"""
# Read training set
radiomics_path_train = 'data/train/features/radiomics.csv'
clinical_path_train = 'data/train/features/clinical_data_cleaned.csv'
label_path_train = 'data/train/y_train.csv'
input_train, output_train = read_data.load_owkin_data(radiomics_path_train, clinical_path_train, label_path_train)
input_train = input_train[features]
print(input_train)
print(output_train)

# Read testing set
radiomics_path_test = 'data/test/features/radiomics.csv'
clinical_path_test = 'data/test/features/clinical_data_cleaned.csv'
input_test = read_data.read_input(radiomics_path_test, clinical_path_test)
input_test = input_test[features]

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
print(input_test)
coxph = CoxPhRegression(**best_params)
coxph.fit(input_train, output_train)
y_pred = coxph.predict(input_test)
y_pred.to_csv('submission.csv')
