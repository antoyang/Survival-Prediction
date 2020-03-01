"""
COX-PH Baseline
Leon Zheng
"""

import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.base import BaseEstimator
from sksurv.util import Surv
import numpy as np
from metrics import cindex


class CoxPhRegression(BaseEstimator):

    def __init__(self, alpha=10, threshold=0.55):
        self.alpha = alpha
        self.threshold = threshold
        self.model = CoxPHSurvivalAnalysis(alpha=alpha)

    def set_data(self, input_train, output_train, input_test):
        self.input_train = input_train
        self.output_train = output_train
        self.input_test = input_test

    def fit(self, X, y):
        structured_y = Surv.from_dataframe('Event', 'SurvivalTime', y)
        self.model.fit(X, structured_y)
        return self

    def predict(self, X):
        prediction = self.model.predict_survival_function(X)
        y_pred = []
        for pred in prediction:
            time = pred.x
            survival_prob = pred.y
            i_pred = 0
            while i_pred < len(survival_prob) - 1 and survival_prob[i_pred] > self.threshold:
                i_pred += 1
            y_pred.append(time[i_pred])
        return pd.DataFrame(np.array([[y, np.nan] for y in y_pred]), index=X.index, columns=['SurvivalTime', 'Event'])

    def score(self, X, y):
        y_pred = self.predict(X)
        return cindex(y, y_pred)


if __name__ == '__main__':

    from sklearn.model_selection import cross_val_score
    # Read training set
    radiomics_train = pd.read_csv("data/train/features/radiomics.csv", index_col=0, usecols=[0, 5, 32, 39, 6, 3])
    clinical_train = pd.read_csv("data/train/features/clinical_data.csv", index_col=0, usecols=[0, 3])
    # added_features = pd.read_csv("data/train/features/train_extracted_features.csv", index_col='patient_number', usecols=[])
    # print(added_features)
    input_train = pd.concat([radiomics_train, clinical_train], axis=1)
    output_train = pd.read_csv("data/train/y_train.csv", index_col=0, header=0)

    coxph = CoxPhRegression()
    # cindex_score = make_scorer(cindex)
    print(cross_val_score(coxph, input_train, output_train, cv=5))

    # Grid search
    from sklearn.model_selection import GridSearchCV
    tuned_params = {"alpha": [0.01, 0.1, 1, 10],
                    "threshold": [0.2, 0.4, 0.6, 0.8]}
    grid_search = GridSearchCV(CoxPhRegression(), tuned_params, cv=5, n_jobs=4)
    grid_search.fit(input_train, output_train)
    print(grid_search.best_score_)
    print(grid_search.best_params_)