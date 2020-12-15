# Survival-Prediction
This repository presents our code for the 2020 challenge "Predicting lung cancer survival time" (proposed by Owkin and hosted by ENS - Collège de France), done jointly with Léon Zheng. We took part to this challenge as part of the Multi-scale models and convolutional neural networks MVA course (teached by Stéphane Mallat). Our work is summarized in Report.pdf. 

We finished first on the private leaderboard among the 98 teams that participated (https://challengedata.ens.fr/participants/challenges/33/), with a C-index of 77.47. Our best solution relies on feature selection with LASSO regression using a combination of features (pyradiomics features including provided ones and additional ones, Laplacian of Gaussian features and wavelet features), plus a 3D U-Net trained for binary semantic segmentation to recover corrupted masks upstream of extraction.

## Feature extraction
Extraction of more pyradiomics features (~130 instead of the ~50 provided), Laplacian of Gaussian and wavelet features: see notebook Feature_extraction/Feature_Extraction_CT.ipynb.

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
One approach that could be interesting to extend this work would consist in using 3D models pretrained on additional data like https://github.com/Tencent/MedicalNet.


