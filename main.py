import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_processing import read_and_preprocess_data, normalize_data
from dataset import NumpyDataset
from model import FFNN
from train import train_model
from utils import format_time, seed_everything

DATA_PATH = "/scratch/x2817a02/workspace/kaggle/ClimSim/data/"
BATCH_SIZE = 1024  # 기존 12288에서 줄임
MIN_STD = 1e-8
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 10**(-0.5)
EPOCHS = 2 # 70
PATIENCE = 1 #6
PRINT_FREQ = 100
BEST_MODEL_PATH = "best_model.pth"

if __name__ == "__main__":
    ts = time.time()
    seed_everything()

    x_train, y_train, FEAT_COLS, TARGET_COLS = read_and_preprocess_data()
    x_train, y_train, mx, sx, my, sy = normalize_data(x_train, y_train)

    print("Time after processing data:", format_time(time.time()-ts), flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    input_size = x_train.shape[1]
    output_size = y_train.shape[1]
    hidden_size = input_size + output_size
    model = FFNN(input_size, [2*hidden_size, hidden_size, hidden_size, 2*hidden_size], output_size).to(device)

    if os.path.exists(BEST_MODEL_PATH):
        print(f"Model file {BEST_MODEL_PATH} already exists. Skipping training.")
    else:
        dataset = NumpyDataset(x_train, y_train)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

        best_model_state = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, EPOCHS, PATIENCE, PRINT_FREQ, len(TARGET_COLS), ts)

        # Save the best model
        torch.save(best_model_state, BEST_MODEL_PATH)
        print("Saved best model to", BEST_MODEL_PATH)

    print("Total time after training or loading:", format_time(time.time()-ts))

