# Survival-Prediction
A Challenge on "Predicting lung cancer survival time" (proposed by Owkin) for the Multi-scale models and convolutional neural networks MVA course (teached by Stéphane Mallat), done with Léon Zheng. Our work is summarized in Report.pdf. Rankings are available at https://challengedata.ens.fr/participants/challenges/33/. Our best solution relies on feature selection with LASSO regression using radiomics features (pyradiomics features including provided ones and additional ones, Laplacian of Gaussian features and wavelet features) 

## Feature extraction
Extraction of more Pyradiomics Features (~130 instead of the ~50 provided), Laplacian of Gaussian and wavelet features: see notebook Feature_extraction/Feature_Extraction_CT.ipynb.

## Feature Selection
We tried 3 different feature selection procedures:
- Recursive Feature Elimination
- LASSO regression: see notebook lasso/FeatureSelectionLasso.ipynb
- Random survival forest: see notebook rsf/rsf_feature_selection.ipynb

## Computer Vision approach
The following functionalities aim to provide solutions to the prediction of survival time from images only - without relying on pyradiomics features - and are provided in the notebook ScanCNN/ScanCNN.ipynb:
- Visualization of the tumor in various forms: video, 2D slice, 3D shape
- Simplification to a 2D problem: In order to apply standard 2D CNN to the problem, one approach we tried consists in extracting only the 2D "most discriminative" slice with maximum number of tumor pixels (either along each dimension, or across all dimensions). We then preprocess them, and extract ResNet-18 features from these slices.
- End-to-end pipeline: based on PyCox and inspired by DeepSurv, we tried finetuning a ResNet-18 while learning a 3-layer perceptron or an attention-based model predicting the risk of patients on top of it, with negative log partial likelihood. Additional analysis of the final model is done with Grad-Cam.
- Binary classification: we tried classifying from the 2D "most discriminative" slice over two classes (one with patients that have high risks, one with patients that have low risks) to understand the difficulty of the task. Additional analysis of the final model is done with Grad-Cam.
- Binary Image Semantic Segmentation: We tried training a 3D U-Net for semantic segmentation both to recover missing masks and to obtain a good 3D feature extractor. 

To try:
3D model pretrained on additional data: https://github.com/Tencent/MedicalNet


