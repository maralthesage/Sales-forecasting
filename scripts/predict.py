import numpy as np

def forecast(model, input_seq, n_forecast, seasonality_future):
    """
    input_seq: last n_lags timesteps (scaled), shape (n_lags, n_features)
    seasonality_future: list of future seasonality values
    """
    output = []
    seq = input_seq.copy()
    for s in seasonality_future:
        x_input = np.expand_dims(seq, axis=0)  # shape (1, n_lags, n_features)
        yhat = model.predict(x_input, verbose=0)
        # Append forecast with seasonality feature
        new_step = [yhat[0,0], s]
        seq = np.vstack([seq[1:], new_step])
        output.append(yhat[0,0])
    return output
