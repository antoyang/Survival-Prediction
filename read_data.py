"""
Helper methods for reading data
Leon Zheng
"""

import pandas as pd
import numpy as np

def read_input(file_radiomics, file_clinical):
    """
    Read radiomics and clinical feature and return dataframe.
    :param file_radiomics: filename
    :param file_clinical: filename
    :return:
    """
    radiomics = pd.read_csv(file_radiomics, index_col=0)
    clinical = pd.read_csv(file_clinical, index_col=0)
    clinical = pd.get_dummies(clinical)
    input = pd.concat([radiomics, clinical], axis=1)
    return input

def clean_clinical_data(file, newfile):
    """
    Cleaning clinical data.
    :param file: clinical data file
    :param newfile: path for the cleaned data file.
    :return:
    """
    with open(file) as f:
        content = f.readlines()
    for i in range(len(content)):
        content[i] = content[i].replace("NSCLC NOS (not otherwise specified)", "nos")
        content[i] = content[i].replace("Adenocarcinoma", "adenocarcinoma")
        content[i] = content[i].replace("Squamous cell carcinoma", "squamous cell carcinoma")
    f = open(newfile, "w+")
    for line in content:
        f.write(line)
    f.close()

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

def read_output(file_output):
    """
    Return dataframe for event + survival time.
    :param file_output:
    :return:
    """
    output = pd.read_csv(file_output, index_col=0, header=0)
    return output

def eliminate_nan(input, output):
    """
    Eliminate patient with nan.
    :param input: features dataframe
    :param output: event + survival time dataframe.
    :return:
    """
    input_no_nan = input.dropna()
    output_no_nan = output.loc[input_no_nan.index]
    return input_no_nan, output_no_nan

def load_owkin_data(radiomics_path, clinical_path, label_path, without_nan=True):
    """
    Load Owkin data: radiomics features, clinical features, and
    :param radiomics_path: filename
    :param clinical_path: filename
    :param label_path: filename
    :param without_nan: set True if reading data without nan
    :return: input is a pandas dataframe with all radiomics and clinical features.
            ouput is event + survival time dataframe.
    """
    input = read_input(radiomics_path, clinical_path)
    output = read_output(label_path)
    if without_nan:
        return eliminate_nan(input, output)
    return input, output

if __name__ == '__main__':
    clean_clinical_data("data/train/features/clinical_data.csv", "data/train/features/clinical_data_cleaned.csv")