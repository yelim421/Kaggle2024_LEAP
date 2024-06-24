import torch
import pandas as pd
import polars as pl
import numpy as np
import os
import gc


from model import FFNN


def test_model(config, mx, sx, my, sy, FEAT_COLS, TARGET_COLS, device):
    if not os.path.exists(config['MODEL_PATH']):
        print(f"Checkpoint file {config['MODEL_PATH']} not found. Exiting.")
        return
    
    df_test = pl.read_csv('/data01/jhko/LEAP/test.csv')
    for col in FEAT_COLS:
        df_test = df_test.with_columns(pl.col(col).cast(pl.Float32))
    
    x_test = df_test.select(FEAT_COLS).to_numpy()
    x_test = (x_test - mx.reshape(1, -1)) / sx.reshape(1, -1)
    #predt = np.zeros([x_test.shape[0], output_size], dtype=np.float32)

    input_size = x_test.shape[1]
    output_size = len(TARGET_COLS)
    hidden_size = input_size + output_size
    model = FFNN(input_size, [3*hidden_size, 2*hidden_size, 2*hidden_size, 2*hidden_size, 3*hidden_size], output_size).to(device)
    model.load_state_dict(torch.load(config['MODEL_PATH']))
    model.eval()

    predt = np.zeros([x_test.shape[0], output_size], dtype=np.float32)

    i1 = 0
    for i in range(10000):
        i2 = np.minimum(i1 + config['BATCH_SIZE'], x_test.shape[0])
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
        if sy[i] < 1e-8 * 1.1:
            predt[:, i] = 0
            
    predt = predt * sy.reshape(1, -1) + my.reshape(1, -1)

    ss = pd.read_csv("/data01/jhko/LEAP/sample_submission.csv")
    ss.iloc[:, 1:] = ss.iloc[:, 1:].astype('float')
    ss.iloc[:, 1:] = predt

    del predt
    gc.collect()

    use_cols = []
    for i in range(27):
        use_cols.append(f"ptend_q0002_{i}")

    ss2 = pd.read_csv("/data01/jhko/LEAP/sample_submission.csv")
    df_test = df_test.to_pandas()

    for col in use_cols:
        ss[col] = -df_test[col.replace("ptend", "state")] * ss2[col] / 1200.

    test_polars = pl.from_pandas(ss[["sample_id"] + TARGET_COLS])
    submission_file = config['MODEL_PATH'].replace('.pth', '.csv')
    test_polars.write_csv(submission_file)

    print(f"Submission file {submission_file} created.")
    
