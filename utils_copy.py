import json
import os
import logging
import datetime
import numpy as np
import torch
import random
import yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_file):
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(message)s', handlers = [
        logging.FileHandler(log_file, mode = 'a'),
        logging.StreamHandler()
    ])
    
def log_hyperparameters(config):
    for key, value in config.items():
        logging.info(f"{key}: {value}")

def format_time(elapsed):
    """Take a time in seconds and return a string hh:mm:ss."""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def seed_everything(seed_val=1325):
    """Seed everything."""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def activate_kaggle_user(user):
    filepath = '/home/jhko/LEAP_ClimSim/kaggle_copy.json'

    with open(filepath, 'r') as f:
        users = json.load(f)
    
    if user not in users:
        raise ValueError(f"User {user} not found in the kaggle.json file")

    user_info = users[user]

    with open('/home/jhko/.kaggle/kaggle.json', 'w') as f:
        json.dump(user_info, f)

    os.chmod('/home/jhko/.kaggle/kaggle.json', 0o600)
    print(f"Activated Kaggle user: {user}")
