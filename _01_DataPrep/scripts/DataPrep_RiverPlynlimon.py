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
import _01_DataPrep.scripts.data_prep_helpers as hlp

def load_and_clean_plynlimon_data(data_folder: str) -> pd.DataFrame:
    """
    Loads Plynlimon river data, filters by site, and performs initial cleaning and type conversion.
    """
    df_rivervars = hlp.load_and_process_csv(
        file_path=data_folder + 'PlylimonEditedData_KirchnerPNAS.csv',
        date_column='date_time',
        localize_tz=None # Original script explicitly dt.tz_localize(None)
    )

    df_rivervars = df_rivervars[df_rivervars["Site"] == 'UHF']

    colnames_keep = ['dayno', 'date_time', 'Flow cumecs', 'NO3-N mg/l', 'SO4 mg/l',
                     'Cl mg/l', 'Na mg/l', 'Mg mg/l', 'K mg/l', 'Ca mg/l']
    
    df = df_rivervars[colnames_keep].copy()
    df = df.set_index('date_time') # Set date_time as index for datetime operations
    
    # Convert numeric columns, coerce errors to NaN
    for c in df.columns.drop('dayno', errors='ignore'): # 'dayno' might not be numeric
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df

def main(data_folder, out_data_folder, res):
    """
    Main function to orchestrate the data preprocessing for River Plynlimon data.
    """
    df = load_and_clean_plynlimon_data(data_folder)

    # Resample to desired resolution and interpolate
    df = hlp.resample_and_interpolate(df, resample_freq=res)

    # Apply specific value filters
    colnames_responses = ['Ca mg/l', 'Mg mg/l', 'K mg/l', 'NO3-N mg/l', 'Cl mg/l', 'Na mg/l', 'SO4 mg/l']
    for c in colnames_responses:
        df[c] = np.where(df[c] < 0, 0, df[c]) # Original script had this for negative values

    # Apply datetime filtering if needed
    # df = hlp.filter_by_datetime_range(df, start_date=dt.datetime(2000,1,1), end_date=dt.datetime(2010,12,31))

    df = hlp.calculate_doy(df) # Calculate DOY using the index

    # Calculate LogQ
    df['LogQ'] = np.log10(df['Flow cumecs'])
    df['LogQ'] = np.where(df['LogQ'] < -10, np.nan, df['LogQ']) # Filter extreme low log values

    # Calculate loads
    colnames_loads = []
    for c in colnames_responses:
        if c in df.columns and 'Flow cumecs' in df.columns:
            df[c + 'Load_g'] = df[c] * df['Flow cumecs'] # Assuming Flow cumecs is volume-like
            colnames_loads.append(c + 'Load_g')

    # Define final columns and save
    colnames_drivers = ['Flow cumecs', 'LogQ'] # Simplified for example
    
    final_cols = [col for col in colnames_responses + colnames_loads + colnames_drivers + ['DOY'] if col in df.columns]
    df = df[final_cols].copy()
    df = df.dropna()

    df.to_csv(out_data_folder + 'Data_PlynlimonRiver_Hourly.csv')

    # Plotting from original script (retained for reference)
    # (fig, ax) = plt.subplots(len(colnames_responses), 1, figsize=(12, 12))
    # for i, c in enumerate(colnames_responses):
    #     ax[i].plot(df[c])
    #     ax[i].set_ylabel(c)
    # plt.show()

if __name__ == '__main__':

    data_folder = '../input/'
    out_data_folder = '../output/'
    res = '7H' # Original script used '7H'

    main(data_folder, out_data_folder, res)