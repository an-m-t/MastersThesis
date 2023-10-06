import numpy as np
import pandas as pd

def is_dataframe_sorted(df, column_name):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    if column_name not in df.columns:
        raise ValueError("Column '{}' not found in the DataFrame.".format(column_name))
    
    column = df[column_name]
    len_d = len(df[column_name])
    sorted_column = column.is_monotonic_increasing and len_d == len(df[column_name].unique())
    return sorted_column

def clean_weather_data(data):
    data = data.values
    data = data[:5000, :]
    return data
