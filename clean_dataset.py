import pandas as  pd
import numpy as np
from fancyimpute import IterativeImputer

"""
Remove outliers and impute NaN values with mice

"""

imputer = IterativeImputer()

def identify_outliers(data, column):
    q1 = np.percentile(data[column],25)
    q3 = np.percentile(data[column],75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    for i in range(0,len(data)):
        if data.loc[i,column] < lower_bound or data.loc[i,column] > upper_bound:
            data.loc[i,column] = np.NAN
    return data


def replace_outliers(data,imputer):
    for d in data.columns:
        data = identify_outliers(data,d)
    data_imputed = imputer.fit_transform(data)
    data = pd.DataFrame(data_imputed,columns = data.columns)
    return data


def drop_outliers(data):
    for d in data.columns:
        data = identify_outliers(data,d)
    data = data.dropna()
    return data

def cleaning_dataset(dataset):
    for i in range(0,2):
        dataset = replace_outliers(dataset, imputer)
    dataset = drop_outliers(dataset)
    return dataset