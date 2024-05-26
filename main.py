import argparse
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
from train_new import train_model, load_config, setup_logging
from utils import format_time, seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main script for training the model.')
    parser.add_argument('--config', type=str, default='hyper.yaml', help='Path to the config file.')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging('training.log')

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

    if os.path.exists(config['BEST_MODEL_PATH']):
        print(f"Model file {config['BEST_MODEL_PATH']} already exists. Skipping training.")
    else:
        dataset = NumpyDataset(x_train, y_train)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['SCHEDULER_FACTOR'], patience=config['SCHEDULER_PATIENCE'])

        best_model_state = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, len(TARGET_COLS), ts, config)

        # Save the best model
        torch.save(best_model_state, config['BEST_MODEL_PATH'])
        print("Saved best model to", config['BEST_MODEL_PATH'])

    print("Total time after training or loading:", format_time(time.time()-ts))
