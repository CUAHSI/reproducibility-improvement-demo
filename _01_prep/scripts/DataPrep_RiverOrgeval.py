#!/usr/bin/env python
# coding: utf-8

# Code to pre-process data for riverlab case - concatenate datasets, gap fill, etc
# save dataframe as csv that is input into the GMM-PCA-IT framework
# Orgeval Version!!!

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
from matplotlib.colors import ListedColormap
# import prep helpers script
import _01_DataPrep.scripts.DataPrep_Helpers as prep_hlp

def load_and_clean_orgeval_data(data_folder: str) -> pd.DataFrame:
    """
    Loads Orgeval river data and performs initial cleaning and type conversion.

    Args:
        data_folder (str): Path to the data folder.

    Returns:
        pd.DataFrame: The cleaned DataFrame with 'Date' as the index.
    """
    df_rivervars = prep_hlp.load_csv(
        file_path=data_folder + 'orgeval_RL.csv',
        date_column='Date',
        localize_tz=None
    )

    # Ensure numeric types for relevant columns
    colnames_keep = ['Date', 'TempRiver', 'Turbidity', 'discharge', 'Magnesium', 'Potassium',
                      'Calcium', 'Sodium', 'Sulfates', 'Nitrates', 'Chlorures']
    
    df = df_rivervars[colnames_keep].copy()
    for c in df.columns:
        if c != 'Date':
            df[c] = pd.to_numeric(df[c], errors='coerce') # Use coerce to turn non-numeric into NaN

    # Replace specific bad values with NaN
    df['discharge'] = np.where(df['discharge'] == 9999, np.nan, df['discharge'])
    df['TempRiver'] = np.where(df['TempRiver'] == 9999, np.nan, df['TempRiver'])
    
    return df.set_index('Date')

def prepare_river_orgeval_data(data_folder: str, 
                               out_data_folder: str, 
                               res: str):
    """
    Orchestrates the data preprocessing for River Orgeval data,
    including loading, cleaning, resampling, and calculating derived variables
    before saving the final DataFrame to a CSV.

    Args:
        data_folder (str): The path to the folder containing the raw data files.
        out_data_folder (str): The path to the folder where the output CSV file will be saved.
        res (str): The resampling frequency string (e.g., '1H', '30T').
    """
    df = load_and_clean_orgeval_data(data_folder)

    # Resample to desired resolution and interpolate
    df = prep_hlp.resample_and_interpolate(df, resample_freq=res, interpolation_limit=24) # limit=24 for 30min data

    # Interpolate for remaining NaNs after resampling (original script had this as a loop)
    for c in df.columns:
        df[c] = df[c].interpolate(method='linear', limit=24) # Original script used limit=24

    df = prep_hlp.calculate_doy(df) # Calculate DOY using the index

    # # Apply specific value filters
    # df['discharge'] = prep_hlp.apply_range_filter(df, 'discharge', lower_bound=0)
    # df['TempRiver'] = prep_hlp.apply_range_filter(df, 'TempRiver', lower_bound=-10, upper_bound=40)
    # df['Turbidity'] = prep_hlp.apply_range_filter(df, 'Turbidity', lower_bound=0)

    # Apply datetime filtering if needed (example)
    # df = prep_hlp.filter_by_datetime_range(df, start_date=dt.datetime(2010,1,1), end_date=dt.datetime(2020,12,31))
    # df = prep_hlp.filter_by_datetime_range(df, exclude_year=2015)

    # Calculate LogQ
    df['LogQ'] = np.log10(df['discharge'])

    # Calculate loads
    colnames_responses = ['Calcium', 'Magnesium', 'Potassium', 'Sodium', 'Chlorures', 'Nitrates', 'Sulfates']
    colnames_loads = []
    for c in colnames_responses:
        if c in df.columns and 'discharge' in df.columns:
            df[c + 'Load_g'] = df[c] * df['discharge']
            colnames_loads.append(c + 'Load_g')

    # Define final columns and save
    colnames_drivers = ['discharge', 'LogQ', 'TempRiver', 'Turbidity'] # Simplified for example
    
    final_cols = [col for col in colnames_responses + colnames_loads + colnames_drivers + ['DOY'] if col in df.columns]
    df = df[final_cols].copy()
    df = df.dropna()

    df.to_csv(out_data_folder + 'Data_OrgevalRiver_Hourly.csv')