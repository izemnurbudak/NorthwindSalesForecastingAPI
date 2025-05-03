# feature_engineering.py

import pandas as pd
import numpy as np

def feature_engineering(df):
    """Creates new features."""
    # Extract year and month features
    # Convert the date column to datetime format
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["month"] = df["order_date"].dt.month 
    df["year"] = df["order_date"].dt.year

    # Calculate total revenue
    df['TotalRevenue'] = df['quantity'] * df['unit_price']

    # Day of the week and weekend information
    df['Weekday'] = df['order_date'].dt.weekday
    df['Quarter'] = df['order_date'].dt.quarter
    df['IsWeekend'] = df['Weekday'].apply(lambda x: "weekend" if x >= 5 else "weekday")

    # First order date
    df['First_Order_Date'] = df.groupby('customer_id')['order_date'].transform('min')
    df['Customer_Type'] = np.where(df['order_date'] == df['First_Order_Date'], 'New Customer', 'Returning Customer')

    # Sales category
    threshold = df["quantity"].mean()
    df["Sales_Category"] = df["quantity"].apply(lambda x: "High" if x > threshold else "Low")

    # Monthly sales growth
    df['Monthly_Sales_Growth'] = df.groupby(['product_id', 'year', 'month'])['TotalRevenue'].pct_change().fillna(0)

    return df


def delete_special_characters(text):
    """Removes special characters and leading zeros from phone numbers."""
    text = text.replace('(', '').replace(')', '')
    while text.startswith('0'):
        text = text[1:]
    return text

def assign_region(text):
    """Assigns region based on the phone number."""
    if text.startswith('1'):
        return 'North America'
    elif text.startswith('33') or text.startswith('49'):
        return 'Europe'
    elif text.startswith('90') or text.startswith('81'):
        return 'Asia'
    else:
        return 'Other'

def customer_region_segmentation(customers):
    """Segments customers based on their phone numbers."""
    customers["phone"] = customers["phone"].apply(delete_special_characters)
    customers["regionSegment"] = customers["phone"].apply(assign_region)
    return customers
