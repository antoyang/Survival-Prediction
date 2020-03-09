"""
Baseline random forest script.
Leon Zheng.

ToDo: choose the right feature and fine tune.
"""


import read_data
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import cross_validate

"""
Reading data
"""
radiomics_path = 'data/train/features/radiomics.csv'
clinical_path = 'data/train/features/clinical_data_cleaned.csv'
label_path = 'data/train/y_train.csv'

def load_data(features=None):
    X_df, y_df = read_data.load_owkin_data(radiomics_path, clinical_path, label_path, without_nan=True)
    if features != None:
        X_df = X_df[features]
    X = X_df.to_numpy()
    y = read_data.y_dataframe_to_rsf_input(y_df)
    return X_df, y_df, X, y

X_df, y_df, X, y = load_data()
feature_name = list(X_df.columns.values)

"""
Train model
"""
params = {'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 10}
rsf = RandomSurvivalForest(n_estimators = params['n_estimators'],
                           min_samples_split = params['min_samples_split'],
                           min_samples_leaf = params['min_samples_leaf'],
                           max_features="sqrt",
                           n_jobs=-1
                           )
print(cross_validate(rsf, X, y, cv=5))
