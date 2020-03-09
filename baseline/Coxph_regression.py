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

    def __init__(self, alpha=1e-8, threshold=0.9):
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
