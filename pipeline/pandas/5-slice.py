def slice(df):
    # Extract the specified columns
    df_subset = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    
    # Select every 60th row
    df_sliced = df_subset.iloc[::60]
    
    # Return the sliced DataFrame
    return df_sliced
