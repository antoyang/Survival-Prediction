# Survival-Prediction
A Challenge on "Predicting lung cancer survival time" for the Multi-scale models and convolutional neural networks MVA course

Questions:
- Other model than DeepSurv ?

## Linear model
Done :
- Scikit-learn Estimator class: ``CoxPhRegression`` in ``Coxph_regression.py``.
- Script for selecting features, grid search and submission with best params: ``Coxph_baseline.py``

ToDo:
- Find an efficient way to select best features based on the previous baseline.
- Incremental Feature Selection.

## DeepSurv
ToDo:
- Read DeepSurv paper. Analyze model.
- Implement DeepSurv.
