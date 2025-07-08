#!/usr/bin/env python
# coding: utf-8

# MIRZ data pre-processing for input into clustering and IT algorithm

import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import _01_DataPrep.scripts.data_prep_helpers as hlp

def load_and_process_rootsoil_data(file_path: str, site_name: str) -> pd.DataFrame:
    """
    Loads root/soil data, converts timestamp, and performs initial cleaning.
    """
    df = hlp.load_and_process_csv(
        file_path=file_path,
        date_column='TIMESTAMP',
        localize_tz=None # Original script did not localize timezone explicitly
    )
    df = df.set_index('TIMESTAMP') # Set TIMESTAMP as index for time series operations

    # Original script performed some specific filtering/cleaning here
    df = df.loc[df['Cs'] > -100] # Example of original cleaning
    df['site'] = site_name
    
    # Convert all columns to numeric, coercing errors
    for col in df.columns.drop('site', errors='ignore'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def main(data_folder,out_data_folder):
    """
    Main function to orchestrate the data preprocessing for RootSoil Nebraska data.
    """
    df_ag = load_and_process_rootsoil_data(data_folder + 'NEAG_QAQC.csv', 'Ag')
    df_pr = load_and_process_rootsoil_data(data_folder + 'NEPR_QAQC.csv', 'Pr')

    # Apply datetime filtering if needed (e.g., specific date ranges or exclude years)
    # df_ag = hlp.filter_by_datetime_range(df_ag, start_date=dt.datetime(2021,1,1), end_date=dt.datetime(2022,12,31))
    # df_pr = hlp.filter_by_datetime_range(df_pr, exclude_year=2020)

    # Perform outlier removal using helper function
    cols_to_clean = ['Cs', 'Cm', 'Cd', 'SWC_5cm', 'SWC_20cm', 'SWC_50cm', 'SWC_100cm',
                     'Temp_5cm', 'Temp_20cm', 'Temp_50cm', 'Temp_100cm', 'SHF_soil']
    
    df_ag = hlp.remove_outliers_by_quantile(df_ag, cols_to_clean)
    df_pr = hlp.remove_outliers_by_quantile(df_pr, cols_to_clean)

    # Interpolate missing values (limit 48 for hourly data with 2 days gap)
    for c in cols_to_clean:
        if c in df_ag.columns:
            df_ag[c] = df_ag[c].interpolate(method='linear', limit=48)
        if c in df_pr.columns:
            df_pr[c] = df_pr[c].interpolate(method='linear', limit=48)
    
    # Further calculations and aggregations (specific to this script)
    df_ag_agg = df_ag.groupby(df_ag.index.date).mean()
    df_pr_agg = df_pr.groupby(df_pr.index.date).mean()

    df_ag_agg.index = pd.to_datetime(df_ag_agg.index)
    df_pr_agg.index = pd.to_datetime(df_pr_agg.index)
    
    df_ag_agg = hlp.calculate_doy(df_ag_agg)
    df_pr_agg = hlp.calculate_doy(df_pr_agg)

    df_ag_agg = df_ag_agg.drop('site', axis=1, errors='ignore')
    df_pr_agg = df_pr_agg.drop('site', axis=1, errors='ignore')

    df_ag_agg['site'] = 'Ag'
    df_pr_agg['site'] = 'Pr'

    # Combine dataframes
    df_combined = pd.concat([df_ag_agg, df_pr_agg], axis=0, ignore_index=False)
    df_combined = df_combined.dropna()

    # Save processed data
    df_combined.to_csv(out_data_folder + 'Data_RootSoilNebraska_Daily.csv')

if __name__ == '__main__':

    data_folder = '../input/'
    out_data_folder = '../output/'

    main(data_folder,out_data_folder)