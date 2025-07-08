import pandas as pd
import numpy as np
from scripts import data_loader, preprocess, model as model_script, train as train_script, predict as predict_script

# Constants (validation and forecast remain the same)
VALID_START = '2025-11-01'
VALID_END = '2025-12-01'
FORECAST_MONTHS = 12
N_LAGS = 12

# Load data
sales_df = data_loader.load_sales_data('data/sales_data.csv')
seasonality_dict = data_loader.load_seasonality('seasonality/wg_seasonality_dict.pkl')

results = []

# Process each product
product_ids = sales_df['ART_NR'].unique()

for pid in product_ids:
    print(f"\nProcessing Product {pid}...")

    # Dynamically determine the earliest month this product had data
    product_dates = sales_df[sales_df['ART_NR'] == pid]['MONAT']
    TRAIN_START = product_dates.min().strftime('%Y-%m-%d')
    TRAIN_END = '2025-10-01'

    # Create time series with seasonality
    try:
        ts_train = preprocess.create_time_series(sales_df, pid, TRAIN_START, TRAIN_END, seasonality_dict)
    except Exception as e:
        print(f"Skipping product {pid} due to error: {e}")
        continue
    try:
        ts_valid = preprocess.create_time_series(sales_df, pid, VALID_START, VALID_END, seasonality_dict)
    except Exception as e:
        print(f"Skipping product {pid} due to error: {e}")
        continue

    # Prepare data
    train_scaled, valid_scaled, scaler = train_script.scale_data(ts_train.values, ts_valid.values)
    X_train, y_train = train_script.prepare_sequences(train_scaled, n_lags=N_LAGS)

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = model_script.build_model(input_shape)

    # Train
    lstm_model.fit(X_train, y_train, epochs=30, verbose=0)

    # Prepare input sequence for forecasting
    ts_full = pd.concat([ts_train, ts_valid])
    full_scaled, _, _ = train_script.scale_data(ts_train.values, ts_full.values)
    input_seq = full_scaled[-N_LAGS:]

    # Future seasonality
    product_category = sales_df[sales_df['ART_NR'] == pid]['WG_NAME'].iloc[0]
    future_seasonality = []
    for i in range(FORECAST_MONTHS):
        month = (ts_full.index[-1] + pd.DateOffset(months=i+1)).month
        s = seasonality_dict[product_category][month]

        future_seasonality.append(s)

    # Forecast
    yhat_scaled = predict_script.forecast(lstm_model, input_seq, n_forecast=FORECAST_MONTHS, seasonality_future=future_seasonality)

    # Inverse scaling
    forecasts = []
    for s, y_scaled in zip(future_seasonality, yhat_scaled):
        inverse = scaler.inverse_transform([[y_scaled, s]])[0][0]
        forecasts.append(max(0, inverse))

    # Append results
    forecast_months = [(ts_full.index[-1] + pd.DateOffset(months=i+1)).strftime('%Y-%m') for i in range(FORECAST_MONTHS)]
    for m, f in zip(forecast_months, forecasts):
        results.append({'ART_NR': pid, 'FORECAST_MONTH': m, 'FORECAST_UNITS': f})

# Save output
forecast_df = pd.DataFrame(results)
forecast_df.to_csv('forecast_results.csv', index=False)
print("\nForecast saved to forecast_results.csv")
