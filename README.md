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

Questions:
- Impact of having smaller images than usual / using repeat x3 instead of standard RGB on the efficiency of the CNNs ?
- Possibility of using 3D images ? seems hard, not many images + no pretrained CNNs
- Performances improvement brought by extracted features without finetuning ? (need PCA + prediction)
- Performances of the end2end CNN: so so despite all the tuning. Can be improved ?

To Do:
- Explore Ondelettes 
- Segmentation Model U-Net like for feature extraction
- Image multiplied by mask ?

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
