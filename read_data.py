"""
Helper methods for reading data
Leon Zheng
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

def read_input(file_radiomics, file_clinical):
    """
    Read radiomics and clinical feature and return dataframe.
    :param file_radiomics: filename
    :param file_clinical: filename
    :return:
    """
    radiomics = pd.read_csv(file_radiomics, index_col=0)
    clinical = pd.read_csv(file_clinical, index_col=0)
    clinical = cleaning_clinical(clinical)
    input = pd.concat([radiomics, clinical], axis=1)
    return input


def cleaning_clinical(clinical):
    """
    Cleaning the clinical dataframe.
    :param clinical:
    :return:
    """
    # Encoding label for SourceDataset
    le = preprocessing.LabelEncoder()
    le.fit(clinical['SourceDataset'])
    le.transform(clinical['SourceDataset'])
    clinical['SourceDataset'] = le.transform(clinical['SourceDataset'])
    # Cleaning Histology
    clinical.replace("NSCLC NOS (not otherwise specified)", "nos", inplace=True)
    clinical.replace("Adenocarcinoma", "adenocarcinoma", inplace=True)
    clinical.replace("Squamous cell carcinoma", "squamous cell carcinoma", inplace=True)
    # Dummies for Histology
    clinical = pd.get_dummies(clinical)
    # Fill age nan
    clinical['age'].fillna((clinical['age'].mean()), inplace=True)

    return clinical

def read_output(file_output):
    """
    Return dataframe for event + survival time.
    :param file_output:
    :return:
    """
    output = pd.read_csv(file_output, index_col=0, header=0)
    return output

def clean_clinical_data(file, newfile):
    """
    Cleaning clinical data.
    :param file: clinical data file
    :param newfile: path for the cleaned data file.
    :return:
    """
    clinical = pd.read_csv(file, index_col=0)
    cleaned = cleaning_clinical(clinical)
    cleaned.to_csv(newfile)

def y_dataframe_to_rsf_input(y_df):
    """
    Input for random survival forest.
    :param y_df: event + survival time dataframe.
    :return:
    """
    y_array = []
    Y = y_df.to_numpy()
    for y in Y:
        tuple = (bool(y[1]), y[0])
        y_array.append(tuple)
    return np.array(y_array, dtype = [(f'{y_df.columns[1]}', np.bool), (f'{y_df.columns[0]}', np.float)])

def load_owkin_data(radiomics_path_train="data/train/features/radiomics.csv",
                    clinical_path_train="data/train/features/clinical_data.csv",
                    label_path_train='data/train/y_train.csv',
                    radiomics_path_test="data/test/features/radiomics.csv",
                    clinical_path_test="data/test/features/clinical_data.csv"):
    """
    Load Owkin data: return PyRadiomics + clinical features of training set in dataframe,
                    event + time of training set in dataframe, and
                    PyRadiomics + clinical features of testing set in dataframe,
    """
    input_train = read_input(radiomics_path_train, clinical_path_train)
    output_train = read_output(label_path_train)
    input_test = read_input(radiomics_path_test, clinical_path_test)
    return input_train, output_train, input_test

if __name__ == '__main__':
    input_train, output_train, input_test = load_owkin_data()
    print(input_train, output_train, input_test)
