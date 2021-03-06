"""
Baseline random forest script.
Leon Zheng.

ToDo: choose the right feature and fine tune.
"""


import preprocessing
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import cross_validate

"""
Reading data
"""

def load_data(features=None):
    X_df, y_df, _ = preprocessing.load_owkin_data()
    if features != None:
        X_df = X_df[features]
    X = X_df.to_numpy()
    y = preprocessing.y_dataframe_to_rsf_input(y_df)
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
