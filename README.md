# Survival-Prediction
This repository presents our code for a challenge on "Predicting lung cancer survival time" (proposed by Owkin) for the Multi-scale models and convolutional neural networks MVA course (teached by Stéphane Mallat), done jointly with Léon Zheng. Our work is summarized in Report.pdf. Rankings are available at https://challengedata.ens.fr/participants/challenges/33/. Our best solution relies on feature selection with LASSO regression using a combination of features (pyradiomics features including provided ones and additional ones, Laplacian of Gaussian features and wavelet features).

## Feature extraction
Extraction of more Pyradiomics Features (~130 instead of the ~50 provided), Laplacian of Gaussian and wavelet features: see notebook Feature_extraction/Feature_Extraction_CT.ipynb.

## Feature Selection
We tried 3 different feature selection procedures:
- Recursive Feature Elimination
- LASSO regression: see notebook lasso/FeatureSelectionLasso.ipynb
- Random survival forest: see notebook rsf/rsf_feature_selection.ipynb

## Computer Vision approach
The following functionalities aim to provide solutions to the prediction of survival time from images only with a deep learning based approach - without relying on pyradiomics features - and are provided in the notebook ScanCNN/ScanCNN.ipynb:
- Visualization of the tumor in various forms: video, 2D slice, 3D shape
- Simplification to a 2D problem with extraction of the 2D "most discriminative" slice and feature extraction with a pretrained ResNet-18 model
- Simplification to binary classification between patients that have high risks or low risks
- End-to-end pipeline with negative log partial likelihood to finetune a ResNet-18 while learning a 3-layer perceptron or an attention-based model predicting the risk of patients on top of it
- Analysis of these approaches with Grad-Cam
- Binary Image Semantic Segmentation with a 3D U-Net to recover missing masks and for 3D feature extraction

To try:
3D model pretrained on additional data: https://github.com/Tencent/MedicalNet


