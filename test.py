import os
import torch
import time
import polars as pl
import pandas as pd
import numpy as np
from model import FFNN
from data_processing import read_and_preprocess_data, normalize_data
from utils import format_time, seed_everything

DATA_PATH = "/scratch/x2817a02/workspace/kaggle/ClimSim/data/"
BATCH_SIZE = 1024  # 기존 12288에서 줄임
MIN_STD = 1e-8
BEST_MODEL_PATH = "best_model.pth"

def predict_and_save(model, device, mx, sx, my, sy, FEAT_COLS, output_size):
    df_test = pl.read_csv(DATA_PATH + "test.csv")

    for col in FEAT_COLS:
        df_test = df_test.with_columns(pl.col(col).cast(pl.Float32))

    x_test = df_test.select(FEAT_COLS).to_numpy()

    x_test = (x_test - mx.reshape(1, -1)) / sx.reshape(1, -1)

    predt = np.zeros([x_test.shape[0], output_size], dtype=np.float32)

    i1 = 0
    for i in range(10): #10000):
        i2 = np.minimum(i1 + BATCH_SIZE, x_test.shape[0])
        if i1 == i2:
            break

        inputs = torch.from_numpy(x_test[i1:i2, :]).float().to(device)

        with torch.no_grad():
            outputs = model(inputs)
            predt[i1:i2, :] = outputs.cpu().numpy()

        i1 = i2

        if i2 >= x_test.shape[0]:
            break

    for i in range(sy.shape[0]):
        if sy[i] < MIN_STD * 1.1:
            predt[:, i] = 0

    predt = predt * sy.reshape(1, -1) + my.reshape(1, -1)

    ss = pd.read_csv(DATA_PATH + "sample_submission.csv")
    ss.iloc[:, 1:] = predt

    # Save predictions to submission.csv
    ss.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    ts = time.time()
    seed_everything()

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Model file {BEST_MODEL_PATH} does not exist. Please train the model first.")
        exit()

    x_train, y_train, FEAT_COLS, TARGET_COLS = read_and_preprocess_data()
    x_train, y_train, mx, sx, my, sy = normalize_data(x_train, y_train)

    print("Time after processing data:", format_time(time.time()-ts), flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    input_size = x_train.shape[1]
    output_size = y_train.shape[1]
    hidden_size = input_size + output_size
    model = FFNN(input_size, [2*hidden_size, hidden_size, hidden_size, 2*hidden_size], output_size).to(device)

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    print("Loaded existing model from", BEST_MODEL_PATH)

    print("Total time after loading model:", format_time(time.time()-ts))

    # Model evaluation and prediction
    predict_and_save(model, device, mx, sx, my, sy, FEAT_COLS, output_size)
    print("Total time after prediction:", format_time(time.time() - ts))

