import pandas as pd
import numpy as np

def preprocess_data(df):
    """Preprocess the data."""
    # Convert the date column to datetime format
    df["order_date"] = pd.to_datetime(df["order_date"])
    
    # Check for missing values
    print("Missing Values:")
    print(df.isnull().sum())
    
    # Clean the missing values
    df.dropna(inplace=True)
    
    return df
