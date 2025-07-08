# 📈 Product Sales Forecasting with LSTM
This project predicts monthly sales units for each product using historical sales data and category-specific seasonality profiles.It uses a Keras LSTM model to learn time series patterns and outputs forecasts for the next 12 months.

## 🚀 Features
* Historical data ingestion: Loads and processes all available sales history per product.
* Seasonality factors: Integrates per-category (warengruppe) monthly seasonality into the model input.
* Dynamic time series generation: Automatically fills gaps in monthly sales records.
* Lagged sequences: Uses configurable lookback windows (N_LAGS) for prediction.
* Future forecasting: Produces 12-month forecasts per product.
* CSV export: Outputs results in a clean tabular format.

## 📂 Project Structure
```product_forecasting/
├── data/
│   └── sales_data.csv
├── seasonality/
│   └── wg_seasonality_dict.pkl (or .json)
├── scripts/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── main.py
└── README.md```


## 🛠️ Requirements
* Python 3.8+
* pandas
* numpy
* scikit-learn
* tensorflow