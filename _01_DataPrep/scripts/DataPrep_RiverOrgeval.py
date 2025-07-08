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
import _01_DataPrep.scripts.data_prep_helpers as hlp

def load_and_clean_orgeval_data(data_folder: str) -> pd.DataFrame:
    """
    Loads Orgeval river data and performs initial cleaning and type conversion.
    """
    df_rivervars = hlp.load_and_process_csv(
        file_path=data_folder + 'orgeval_RL.csv',
        date_column='Date',
        localize_tz=None # Original script explicitly dt.tz_localize(None)
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

def main(data_folder, out_data_folder, res):
    """
    Main function to orchestrate the data preprocessing for River Orgeval data.
    """
    df = load_and_clean_orgeval_data(data_folder)

    # Resample to desired resolution and interpolate
    df = hlp.resample_and_interpolate(df, resample_freq=res, interpolation_limit=24) # limit=24 for 30min data

    # Interpolate for remaining NaNs after resampling (original script had this as a loop)
    for c in df.columns:
        df[c] = df[c].interpolate(method='linear', limit=24) # Original script used limit=24

    df = hlp.calculate_doy(df) # Calculate DOY using the index

    # # Apply specific value filters
    # df['discharge'] = hlp.apply_range_filter(df, 'discharge', lower_bound=0)
    # df['TempRiver'] = hlp.apply_range_filter(df, 'TempRiver', lower_bound=-10, upper_bound=40)
    # df['Turbidity'] = hlp.apply_range_filter(df, 'Turbidity', lower_bound=0)

    # Apply datetime filtering if needed (example)
    # df = hlp.filter_by_datetime_range(df, start_date=dt.datetime(2010,1,1), end_date=dt.datetime(2020,12,31))
    # df = hlp.filter_by_datetime_range(df, exclude_year=2015)

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

    # Plotting from original script (retained for reference)
    # (fig, ax) = plt.subplots(len(colnames_responses), 1, figsize=(12, 12))
    # for i, c in enumerate(colnames_responses):
    #     ax[i].plot(df[c])
    #     ax[i].set_ylabel(c)
    # plt.show()

if __name__ == '__main__':

    data_folder = '../input/'
    out_data_folder = '../output/'
    res = '30min' # or '7H' as in original notes

    main(data_folder, out_data_folder, res)