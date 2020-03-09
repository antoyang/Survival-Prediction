"""
Cox-Ph regression. Baseline.
Leon Zheng
"""

from Coxph_regression import CoxPhRegression
from sklearn.model_selection import GridSearchCV
import preprocessing
import numpy as np

"""
Feature selection
"""
radiomics_features = ['original_shape_Sphericity', 'original_shape_SurfaceVolumeRatio',
                      'original_shape_Maximum3DDiameter', 'original_glcm_JointEntropy', 'original_glcm_Id',
                      'original_glcm_Idm']
clinical_features = ['SourceDataset', 'Nstage']
features = radiomics_features + clinical_features

"""
Reading data
"""
# Read clean data
input_train, output_train, input_test = preprocessing.load_owkin_data()
input_train = input_train[features]
input_test = input_test[features]

# Normalization
input_train, input_test = preprocessing.normalizing_input(input_train, input_test)
print(input_train)
print(output_train)

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
