import pandas as pd
import polars as pl
import time
import numpy as np
import gc
import torch
from utils_copy import format_time
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_train(ts):

    weights = pd.read_csv('/data01/jhko/LEAP/sample_submission.csv', nrows = 1)
    del weights['sample_id']
    weights = weights.T
    weights = weights.to_dict()[0]

    df_train = pl.read_csv('/data01/jhko/LEAP/train.csv', n_rows = 2_500_500)
    for target in weights:
        df_train = df_train.with_columns(pl.col(target).mul(weights[target]))
    print('time to read dataset:', format_time(time.time()-ts), flush = True)

    FEAT_COLS = df_train.columns[1:557]
    TARGET_COLS = df_train.columns[557:]

    for col in FEAT_COLS:
        df_train = df_train.with_columns(pl.col(col).cast(pl.Float32))

    for col in TARGET_COLS:
        df_train = df_train.with_columns(pl.col(col).cast(pl.Float32))

    x_train = df_train.select(FEAT_COLS).to_numpy()
    y_train = df_train.select(TARGET_COLS).to_numpy()

    del df_train
    gc.collect()
    
    return x_train, y_train, FEAT_COLS, TARGET_COLS

def normalization(x_train, y_train):
    mx = x_train.mean(axis=0)
    sx = np.maximum(x_train.std(axis=0), 1e-8)
    x_train = (x_train - mx.reshape(1,-1)) / sx.reshape(1,-1)

    my = y_train.mean(axis=0)
    sy = np.maximum(np.sqrt((y_train*y_train).mean(axis=0)), 1e-8)
    y_train = (y_train - my.reshape(1,-1)) / sy.reshape(1,-1)

    return x_train, y_train, mx, sx, my, sy


class NumpyDataset(Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0], "x, y sample shape not same"
        self.x = x
        self.y = y
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]).float().to(device), torch.from_numpy(self.y[index]).float().to(device)
        