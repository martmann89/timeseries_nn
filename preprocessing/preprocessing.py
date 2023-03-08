import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import config as cfg
from utility import ROOT_DIR

def preprocess(df, columns):
    df = scale_data(df, ROOT_DIR + '/outputs/scaler/output_scaler.pckl', columns)
    train_df, val_df, test_df = split_data(df)
    return train_df, val_df, test_df


def split_data(df):
    n = len(df) - (cfg.data['test_data_size']+cfg.nn_pred['input_len'])
    train_df = df[0:int(n * cfg.data['train_data_perc'])]
    val_df = df[int(n * cfg.data['train_data_perc']):n]
    test_df = df[-(cfg.data['test_data_size']+cfg.nn_pred['input_len']):]
    return train_df, val_df, test_df


def scale_data(df, filepath, columns):
    output_scaler = MinMaxScaler(feature_range=(0, 1))
    output_scaler.fit(df[[cfg.label]])
    file_scaler = open(filepath, 'wb')
    pickle.dump(output_scaler, file_scaler)

    scaler_all = MinMaxScaler(feature_range=(0, 1))
    df = pd.DataFrame(scaler_all.fit_transform(df[columns]), columns=columns)
    return df
