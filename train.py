import logging
import time
import torch
import torch.optim as optim
from torchmetrics.regression import R2Score
from torch.utils.data import DataLoader
import gc


from utils_copy import format_time # dont touch
from load_data import NumpyDataset
from model import FFNN
from loss import LpLoss

def train_model(config, x_train, y_train, FEAT_COLS, TARGET_COLS, device, ts):
    
    dataset = NumpyDataset(x_train, y_train)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
    
    input_size = x_train.shape[1] #556
    output_size = y_train.shape[1] #368
    hidden_size = input_size + output_size
    model = FFNN(input_size, [3*hidden_size, 2*hidden_size, 2*hidden_size, 2*hidden_size, 3*hidden_size], output_size).to(device)
    criterion = LpLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['SCHEDULER_FACTOR'], patience=config['SCHEDULER_PATIENCE'])
    print("Time after all preparations:", format_time(time.time()-ts), flush=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_count = 0
    r2score = R2Score(num_outputs=len(TARGET_COLS)).to(device)
    
    for epoch in range(config['EPOCHS']):
        print(" ")
        model.train()
        total_loss = 0
        steps = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if (batch_idx + 1) % config['PRINT_FREQ'] == 0:
                current_lr = optimizer.param_groups[0]['lr']
                elapsed_time = format_time(time.time() - ts)
                log_message = (
                    f'  Epoch: {epoch+1}',
                    f'  Batch: {batch_idx + 1}/{len(train_loader)}',
                    f'  Train Loss: {total_loss / steps:.4f}',
                    f'  LR: {current_lr:.1e}',
                    f'  Time: {elapsed_time}'
                )
                logging.info(log_message)
                total_loss = 0
                steps = 0

        model.eval()
        val_loss = 0
        y_true = torch.tensor([], device=device)
        all_outputs = torch.tensor([], device=device)
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                y_true = torch.cat((y_true, labels), 0)
                all_outputs = torch.cat((all_outputs, outputs), 0)
        
        r2 = 0
        r2_broken = []
        r2_broken_names = []
        for i in range(output_size):
            r2_i = r2score(all_outputs[:, i], y_true[:, i])
            if r2_i > 1e-6:
                r2 += r2_i
            else:
                r2_broken.append(i)
                r2_broken_names.append(FEAT_COLS[i])
        r2 /= output_size

        avg_val_loss = val_loss / len(val_loader)
        val_log = (f'\nEpoch: {epoch+1}  Val Loss: {avg_val_loss:.4f}  R2 score: {r2:.4f}')
        logging.info(val_log)

        print(f'{len(r2_broken)} targets were excluded during evaluation of R2 score.')

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_count = 0
            log_msg = ("Validation loss decreased, saving new best model and resetting patience counter.")
            logging.info(log_msg)
        else:
            patience_count += 1
            log_msg = (f"No improvement in validation loss for {patience_count} epochs.")
            logging.info(log_msg)

        if patience_count >= config['PATIENCE']:
            log_msg = ("Stopping early due to no improvement in validation loss.")
            logging.info(log_msg)
            break

    del x_train, y_train
    gc.collect()

    if best_model_state is not None:
        torch.save(best_model_state, config['MODEL_PATH'])
        logging.info(f"Model checkpoint saved as {config['MODEL_PATH']}.")
    else:
        logging.info("No valid model state to save.")
    
    print("Total training time:", format_time(time.time() - ts))
