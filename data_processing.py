import gc
import pandas as pd
import polars as pl
import numpy as np

DATA_PATH = "/data01/jhko/LEAP/" #"/scratch/x2817a02/workspace/kaggle/ClimSim/data/"
MIN_STD = 1e-8

def read_and_preprocess_data():
	weights = pd.read_csv(DATA_PATH + "sample_submission.csv", nrows=1)
	del weights['sample_id']
	weights = weights.T
	weights = weights.to_dict()[0] #가중치 가져옴

	df_train = pl.read_csv(DATA_PATH + "train.csv", n_rows=2_500_000)

	for target in weights:
		df_train = df_train.with_columns(pl.col(target).mul(weights[target])) #가중치 곱함

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

def normalize_data(x_train, y_train):
    mx = x_train.mean(axis=0)
    sx = np.maximum(x_train.std(axis=0), MIN_STD)
    x_train = (x_train - mx.reshape(1,-1)) / sx.reshape(1,-1)

    my = y_train.mean(axis=0)
    sy = np.maximum(np.sqrt((y_train*y_train).mean(axis=0)), MIN_STD)
    y_train = (y_train - my.reshape(1,-1)) / sy.reshape(1,-1)

    return x_train, y_train, mx, sx, my, sy
