import pandas as pd
import numpy as np

def create_time_series(df, product_id, start_date, end_date, seasonality_dict):
    """
    For a single product, create the time series array between start_date and end_date.
    """
    product_df = df[df['ART_NR'] == product_id].copy()
    
    if product_df.empty:
        raise ValueError(f"No data found for product ID {product_id}")
    
    if product_df['WG_NAME'].isnull().any():
        raise ValueError(f"Missing WG_NAME for product ID {product_id}")

    all_months = pd.date_range(start=start_date, end=end_date, freq='MS')
    product_df = product_df.set_index('MONAT').reindex(all_months)
    
    # Use assignment instead of inplace fillna
    product_df['UNITS'] = product_df['UNITS'].fillna(0)
    
    product_df['MONTH_NUM'] = all_months.month
    
    category = df[df['ART_NR'] == product_id]['WG_NAME'].iloc[0]
    
    if pd.isna(category):
        raise ValueError(f"WG_NAME is NaN for product ID {product_id}")
    
    seasonality = seasonality_dict.get(category)
    
    if seasonality is None:
        raise KeyError(f"Category '{category}' not found in seasonality_dict")
    
    product_df['SEASONALITY'] = product_df['MONTH_NUM'].apply(lambda m: seasonality[m])
    
    return product_df[['UNITS', 'SEASONALITY']]


