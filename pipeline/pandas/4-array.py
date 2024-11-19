#!/usr/bin/env python3

def array(df):
    # Select the last 10 rows of 'High' and 'Close' columns
    last_10_rows = df[['High', 'Close']].tail(10)
    
    # Convert the selected rows into a NumPy array
    return last_10_rows.to_numpy()
