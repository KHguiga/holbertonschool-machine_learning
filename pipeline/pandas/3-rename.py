import pandas as pd

def rename(df):
    # Rename 'Timestamp' column to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})
    
    # Convert 'Datetime' column from integer (or float) to datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    
    # Select only 'Datetime' and 'Close' columns
    return df[['Datetime', 'Close']]
