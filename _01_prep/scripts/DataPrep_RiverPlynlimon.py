#!/usr/bin/env python
# coding: utf-8

# Code to pre-process data for riverlab case - concatenate datasets, gap fill, etc
# save dataframe as csv that is input into the GMM-PCA-IT framework
# Plynlimon, UK version!!!

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
from matplotlib.colors import ListedColormap
# import prep helpers script
import _01_DataPrep.scripts.DataPrep_Helpers as prep_hlp

def load_and_clean_plynlimon_data(data_folder: str) -> pd.DataFrame:
    """
    Loads Plynlimon river data, filters by site, and performs initial cleaning and type conversion.

    Args:
        data_folder (str): Path to the data folder.

    Returns:
        pd.DataFrame: The cleaned DataFrame with 'date_time' as the index.
    """
    df_rivervars = prep_hlp.load_csv(
        file_path=data_folder + 'PlylimonEditedData_KirchnerPNAS.csv',
        date_column='date_time',
        localize_tz=None
    )

    df_rivervars = df_rivervars[df_rivervars["Site"] == 'UHF']

    colnames_keep = ['dayno', 'date_time', 'Flow cumecs', 'NO3-N mg/l', 'SO4 mg/l',
                      'Cl mg/l', 'Na mg/l', 'Mg mg/l', 'K mg/l', 'Ca mg/l']
    
    df = df_rivervars[colnames_keep].copy()
    df = df.set_index('date_time')
    
    # Convert numeric columns, coerce errors to NaN
    for c in df.columns.drop('dayno', errors='ignore'):
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df

def prepare_river_plynlimon_data(data_folder: str, 
                                 out_data_folder: str, 
                                 res: str):
    """
    Orchestrates the data preprocessing for River Plynlimon data,
    including loading, cleaning, resampling, and calculating derived variables
    before saving the final DataFrame to a CSV.

    Args:
        data_folder (str): The path to the folder containing the raw data files.
        out_data_folder (str): The path to the folder where the output CSV file will be saved.
        res (str): The resampling frequency string (e.g., '1H', '30T').
    """
    df = load_and_clean_plynlimon_data(data_folder)

    # Resample to desired resolution and interpolate
    df = prep_hlp.resample_and_interpolate(df, resample_freq=res)

    # Apply specific value filters
    colnames_responses = ['Ca mg/l', 'Mg mg/l', 'K mg/l', 'NO3-N mg/l', 'Cl mg/l', 'Na mg/l', 'SO4 mg/l']
    for c in colnames_responses:
        df[c] = np.where(df[c] < 0, 0, df[c])

    df = prep_hlp.calculate_doy(df)

    # Calculate LogQ
    df['LogQ'] = np.log10(df['Flow cumecs'])
    df['LogQ'] = np.where(df['LogQ'] < -10, np.nan, df['LogQ'])

    # Calculate loads
    colnames_loads = []
    for c in colnames_responses:
        if c in df.columns and 'Flow cumecs' in df.columns:
            df[c + 'Load_g'] = df[c] * df['Flow cumecs']
            colnames_loads.append(c + 'Load_g')

    # Define final columns and save
    colnames_drivers = ['Flow cumecs', 'LogQ']
    
    final_cols = [col for col in colnames_responses + colnames_loads + colnames_drivers + ['DOY'] if col in df.columns]
    df = df[final_cols].copy()
    df = df.dropna()

    df.to_csv(out_data_folder + 'Data_PlynlimonRiver_Hourly.csv')