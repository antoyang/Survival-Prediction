"""
Coxnet: CoxPH with Lasso penalty. On Owkin dataset.
Leon Zheng
"""

import preprocessing
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sksurv.util import Surv
import numpy as np
import pandas as pd

# Features
# features = ['original_shape_Compactness2',
#  'original_shape_SphericalDisproportion',
#  'original_shape_SurfaceVolumeRatio',
#  'original_firstorder_Kurtosis',
#  'original_firstorder_MeanAbsoluteDeviation',
#  'original_firstorder_Minimum',
#  'original_glcm_ClusterProminence',
#  'original_glcm_Contrast',
#  'original_glcm_DifferenceEntropy',
#  'original_glcm_DifferenceAverage',
#  'original_glcm_JointEnergy',
#  'original_glcm_Id',
#  'original_glcm_Idm',
#  'original_glcm_Imc1',
#  'original_glcm_Imc2',
#  'original_glcm_Idmn',
#  'original_glcm_Idn',
#  'original_glrlm_ShortRunEmphasis',
#  'original_glrlm_LongRunEmphasis',
#  'original_glrlm_GrayLevelNonUniformity',
#  'original_glrlm_RunPercentage',
#  'original_glrlm_ShortRunLowGrayLevelEmphasis',
#  'original_glrlm_LongRunLowGrayLevelEmphasis',
#  'original_glrlm_LongRunHighGrayLevelEmphasis',
#  'Nstage',
#  'age',
#  'SourceDataset']

# radiomics_features = ['original_shape_Sphericity', 'original_shape_SurfaceVolumeRatio',
#                       'original_shape_Maximum3DDiameter', 'original_glcm_JointEntropy', 'original_glcm_Id',
#                       'original_glcm_Idm']
# clinical_features = ['SourceDataset', 'Nstage']
# features = radiomics_features + clinical_features

features = ['Mstage',
            'Nstage',
            'SourceDataset',
            'age',
            'original_shape_VoxelVolume',
            'original_firstorder_Maximum',
            'original_firstorder_Mean',
            'original_glcm_ClusterProminence',
            'original_glcm_Idm',
            'original_glcm_Idn',
            'original_glrlm_RunPercentage']

# Read data
input_train, output_train, input_test = preprocessing.load_owkin_data()
input_train = input_train[features]
input_test = input_test[features]
input_train, input_test = preprocessing.normalizing_input(input_train, input_test)
structured_y = Surv.from_dataframe('Event', 'SurvivalTime', output_train)

# Coxnet
# coxnet = CoxnetSurvivalAnalysis()
# print(cross_validate(coxnet, input_train, structured_y, cv=5))

# Grid search
tuned_params = {"l1_ratio": np.linspace(0.01, 0.02, 100),
                "n_alphas": range(140, 160, 1),
                }
grid_search = RandomizedSearchCV(CoxnetSurvivalAnalysis(), tuned_params, cv=5, n_jobs=4, n_iter=1000)
grid_search.fit(input_train, structured_y)
print(grid_search.best_score_)
best_params = grid_search.best_params_
print(best_params)

# Prediction
def predict(model, X, threshold=0.9):
    prediction = model.predict_survival_function(X)
    y_pred = []
    for pred in prediction:
        time = pred.x
        survival_prob = pred.y
        i_pred = 0
        while i_pred < len(survival_prob) - 1 and survival_prob[i_pred] > threshold:
            i_pred += 1
        y_pred.append(time[i_pred])
    return pd.DataFrame(np.array([[y, np.nan] for y in y_pred]), index=X.index, columns=['SurvivalTime', 'Event'])

coxph = CoxnetSurvivalAnalysis(**best_params, fit_baseline_model=True)
coxph.fit(input_train, structured_y)
y_pred = predict(coxph, input_test)
y_pred.to_csv('submission.csv')
