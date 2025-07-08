import pandas as pd
import pickle

def load_sales_data(path):
    df = pd.read_csv(path)
    df['MONAT'] = pd.to_datetime(df['MONAT'])
    return df

def load_seasonality(path):
    with open(path, 'r') as f:
        text = f.read()
    seasonality_dict = eval(text)
    return seasonality_dict


