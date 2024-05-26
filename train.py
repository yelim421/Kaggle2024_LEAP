import torch
import torch.optim as optim
from torchmetrics.regression import R2Score
import time
from utils import format_time
import yaml
import logging

def load_config(config_path):
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config

def setup_logging(log_file):
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
		logging.FileHandler(log_file, mode = 'a'),
		logging.StreamHandler()
	])

def train_model(config_path):
	config = laod_config(config_path)
	setup_logging('training.log')
	logging.info(f"Hyperparameters : {config}")

    best_val_loss = float('inf')
    best_model_state = None
    patience_count = config['patience_count']
    r2score = R2Score(num_outputs=num_outputs).to(device)

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        steps = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

            if (batch_idx + 1) % print_freq == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                elapsed_time = format_time(time.time() - start_time)
                print(f'  Epoch: {epoch+1}',\
                      f'  Batch: {batch_idx + 1}/{len(train_loader)}',\
                      f'  Train Loss: {total_loss / steps:.4f}',\
                      f'  LR: {current_lr:.1e}',\
                      f'  Time: {elapsed_time}', flush=True)
                total_loss = 0
                steps = 0

        model.eval()
        val_loss = 0
        y_true = torch.tensor([], device=device)
        all_outputs = torch.tensor([], device=device)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                y_true = torch.cat((y_true, labels), 0)
                all_outputs = torch.cat((all_outputs, outputs), 0)

        r2 = r2score(all_outputs, y_true)
        avg_val_loss = val_loss / len(val_loader)
        print(f'\nEpoch: {epoch+1}  Val Loss: {avg_val_loss:.4f}  R2 score: {r2:.4f}')
		logging.info(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f} R2 score: {r2:4f}")
        
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_count = 0
            print("Validation loss decreased, saving new best model and resetting patience counter.")
        else:
            patience_count += 1
            print(f"No improvement in validation loss for {patience_count} epochs.")
            
        if patience_count >= patience:
            print("Stopping early due to no improvement in validation loss.")
            break

    return best_model_state

