import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_sequences(series, n_lags=3):
    """
    Create sequences: X (n_lags steps) -> y (next step)
    """
    X, y = [], []
    for i in range(len(series) - n_lags):
        seq_x = series[i:i+n_lags]
        seq_y = series[i+n_lags]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def scale_data(train_data, test_data):
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled, scaler
