"""
LSTM training script for ForecastIQ.
Run once from the project root:  python train_lstm.py
Outputs: models/lstm_model.pt, models/lstm_config.json, models/lstm_scaler.pkl
Updates: models/model_metrics.json with lstm MAE / RMSE
"""
import os, json, time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Hyper-parameters ─────────────────────────────────────────────────────────
SEQ_LEN    = 8
BATCH_SIZE = 1024
EPOCHS     = 25
LR         = 1e-3
HIDDEN     = 128
N_LAYERS   = 2
TRAIN_WEEK = 130

SEQ_FEATURES = [
    'log_orders', 'discount_pct',
    'emailer_for_promotion', 'homepage_featured',
    'category_enc', 'cuisine_enc', 'center_type_enc',
]

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')


# ── Model definition ──────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ── Data loading & feature engineering ───────────────────────────────────────
def load_data():
    train       = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
    meal_info   = pd.read_csv(os.path.join(DATASET_DIR, 'meal_info.csv'))
    center_info = pd.read_csv(os.path.join(DATASET_DIR, 'fulfilment_center_info.csv'))

    df = train.merge(meal_info, on='meal_id', how='left')
    df = df.merge(center_info, on='center_id', how='left')

    df['discount_pct'] = (
        (df['base_price'] - df['checkout_price']) / df['base_price']
    ).clip(lower=0)

    label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    for col in ['category', 'cuisine', 'center_type']:
        le = label_encoders[col]
        df[col + '_enc'] = le.transform(df[col].astype(str))

    df = df.sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)
    df['log_orders'] = np.log1p(df['num_orders'])
    return df


def make_sequences(df):
    Xs, ys, ws = [], [], []
    for _, grp in df.groupby(['center_id', 'meal_id']):
        grp  = grp.sort_values('week')
        vals = grp[SEQ_FEATURES].values.astype(np.float32)
        wks  = grp['week'].values
        for i in range(SEQ_LEN, len(vals)):
            Xs.append(vals[i - SEQ_LEN : i])
            ys.append(vals[i, 0])   # log_orders of target step
            ws.append(wks[i])
    return np.array(Xs), np.array(ys, dtype=np.float32), np.array(ws)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print('=' * 55)
    print('  ForecastIQ — LSTM Training')
    print('=' * 55)

    print('\n[1/5] Loading & engineering features...')
    df = load_data()
    print(f'      Rows: {len(df):,}')

    print('\n[2/5] Creating sequences (seq_len={})...'.format(SEQ_LEN))
    X, y, weeks = make_sequences(df)
    print(f'      Total sequences: {len(X):,}')

    train_m = weeks <= TRAIN_WEEK
    val_m   = weeks  > TRAIN_WEEK
    X_tr, y_tr = X[train_m], y[train_m]
    X_va, y_va = X[val_m],   y[val_m]
    print(f'      Train: {len(X_tr):,}  |  Val: {len(X_va):,}')

    print('\n[3/5] Scaling features...')
    n_feat  = len(SEQ_FEATURES)
    scaler  = MinMaxScaler()
    scaler.fit(X_tr.reshape(-1, n_feat))

    X_tr_s = scaler.transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
    X_va_s = scaler.transform(X_va.reshape(-1, n_feat)).reshape(X_va.shape)

    # Scale log_orders target the same way as feature index 0
    lo_min, lo_rng = float(scaler.data_min_[0]), float(scaler.data_max_[0] - scaler.data_min_[0])
    y_tr_s = (y_tr - lo_min) / lo_rng
    y_va_s = (y_va - lo_min) / lo_rng

    tr_loader = DataLoader(SeqDataset(X_tr_s, y_tr_s),
                           batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(SeqDataset(X_va_s, y_va_s),
                           batch_size=BATCH_SIZE * 2, num_workers=0)

    print('\n[4/5] Training LSTM...')
    model     = LSTMForecaster(n_feat, HIDDEN, N_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(X_tr)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                va_loss += criterion(model(xb), yb).item() * len(yb)
        va_loss /= len(X_va)

        scheduler.step(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f'      Epoch {epoch:02d}/{EPOCHS}  '
                  f'train={tr_loss:.5f}  val={va_loss:.5f}  '
                  f'({elapsed:.0f}s elapsed)')

    model.load_state_dict(best_state)

    # Final evaluation in original order-count space
    model.eval()
    preds_s = []
    with torch.no_grad():
        for xb, _ in va_loader:
            preds_s.append(model(xb).numpy())
    preds_s   = np.concatenate(preds_s)
    preds_log = preds_s * lo_rng + lo_min
    preds_ord = np.maximum(np.expm1(preds_log), 0)
    true_ord  = np.maximum(np.expm1(y_va), 0)

    mae  = float(mean_absolute_error(true_ord, preds_ord))
    rmse = float(np.sqrt(mean_squared_error(true_ord, preds_ord)))
    print(f'\n      LSTM  MAE={mae:.2f}  RMSE={rmse:.2f}')

    print('\n[5/5] Saving artifacts...')
    torch.save(model.state_dict(),
               os.path.join(MODELS_DIR, 'lstm_model.pt'))

    cfg = {
        'input_size':  n_feat,
        'hidden_size': HIDDEN,
        'num_layers':  N_LAYERS,
        'seq_len':     SEQ_LEN,
        'seq_features': SEQ_FEATURES,
        'lo_min':      lo_min,
        'lo_rng':      lo_rng,
    }
    with open(os.path.join(MODELS_DIR, 'lstm_config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    joblib.dump(scaler, os.path.join(MODELS_DIR, 'lstm_scaler.pkl'))

    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    with open(metrics_path) as f:
        metrics = json.load(f)
    metrics['lstm'] = {'MAE': round(mae, 4), 'RMSE': round(rmse, 4)}
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    for fname in ['lstm_model.pt', 'lstm_config.json', 'lstm_scaler.pkl']:
        p = os.path.join(MODELS_DIR, fname)
        print(f'      {fname}: {os.path.getsize(p)/1024:.1f} KB')
    print(f'      model_metrics.json: updated')
    print(f'\nDone in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
