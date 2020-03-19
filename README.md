# Survival-Prediction
A Challenge on "Predicting lung cancer survival time" for the Multi-scale models and convolutional neural networks MVA course

## Computer Vision Pipeline
Done:
- Pyradiomics Feature Extraction Reproduction (130 instead of 5X)
- 3D Scan / Mask Visualization or 2D Slice Visualization
- 2D discriminative slice extraction and preprocessing (to make it compatible with standard pretrained CNNs)
- CNN (Resnet18) Feature Extraction from either best slice among each dim (3 times 512 feature vector per train / test scan), either the best slice only (512 feature vector per scan)
- End to End pipeline based on pycox, a debug model, a ConvNet from scratch and a Resnet + MLP model.
Tried with clean dataloader for data augmentation, but problem with compute baseline hazard then.
- Binary Classifier (with clean data augmentation and dataloader)
- Grad-CAM of end2end training / binary classifier 
- 3D Segmentation training, 3D plot, recovering empty masks + feature extraction and corresponding evaluation 
- LoG and Wavelet visualizations


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
