#!/usr/bin/env python
# coding: utf-8

# Code to pre-process data for riverlab case - concatenate datasets, gap fill, etc
# save dataframe as csv that is input into the GMM-PCA-IT framework

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
from matplotlib.colors import ListedColormap
# import prep helpers script
import _01_DataPrep.scripts.DataPrep_Helpers as prep_hlp

def load_and_resample_flux_data(fluxdata_folder: str, 
                                res: str) -> pd.DataFrame:
    """
    Loads and resamples flux tower data relevant for river analysis.

    Args:
        fluxdata_folder (str): The path to the folder containing flux data CSVs.
        res (str): The resampling frequency string (e.g., '1H', '30T').

    Returns:
        pd.DataFrame: A DataFrame with resampled flux data, including a smoothed LE column.
    """
    df_fluxtower = prep_hlp.load_csv(
        file_path=fluxdata_folder + 'FluxData_15min_2021_2022.csv',
        date_column='Date',
        localize_tz=True
    )
    df_fluxtower = df_fluxtower[['Date', 'Precip_Tot', 'D5TE_VWC_5cm_Avg', 'D5TE_VWC_100cm_Avg', 'LE_li_wpl']]
    
    # Resample to desired resolution (e.g., 30min)
    df_flux_resampled = prep_hlp.resample_and_interpolate(
        df_fluxtower.set_index('Date'),
        resample_freq=res,
        interpolation_limit=24
    )

    # Smoother version of LE
    df_flux_resampled['LE_li_wpl_smooth'] = df_flux_resampled['LE_li_wpl'].rolling(24 * 3, min_periods=24).mean()
    return df_flux_resampled.reset_index()

def load_and_process_river_data(data_folder: str, 
                                res: str) -> pd.DataFrame:
    """
    Loads and processes river discharge and chemistry data.

    Args:
        data_folder (str): The path to the folder containing river data CSVs.
        res (str): The resampling frequency string (e.g., '1H', '30T').

    Returns:
        pd.DataFrame: A DataFrame with processed and merged river data.
    """
    df_river_Q = prep_hlp.load_csv(
        file_path=data_folder + 'RiverData_Hourly_Monticello.csv',
        date_column='Date',
        localize_tz=True
    )

    # Process Temperature, Turbidity, DO
    df_river_temp = prep_hlp.load_csv(
        file_path=data_folder + 'RiverData_Monticello_Temp_Turb_DO.csv',
        date_column='Date',
        localize_tz=True
    )

    df_river_chem = prep_hlp.load_csv(
        file_path=data_folder + 'RiverData_Monticello_Chem.csv',
        date_column='Date',
        localize_tz=True
    )
    
    # Merge river dataframes
    df = pd.merge(df_river_Q, df_river_temp, on='Date', how='outer')
    df = pd.merge(df, df_river_chem, on='Date', how='outer')

    df = df.set_index('Date')
    df = df.apply(pd.to_numeric, errors='ignore')

    # Replace values outside bounds with NaN, then interpolate
    df = prep_hlp.apply_range_filter(df, 'Discharge', lower_bound=0)
    df = prep_hlp.apply_range_filter(df, 'Precip_gage', lower_bound=0)
    df = prep_hlp.apply_range_filter(df, 'Turbidity', lower_bound=0)
    df = prep_hlp.apply_range_filter(df, 'TempRiver', lower_bound=-10, upper_bound=40)
    df = prep_hlp.apply_range_filter(df, 'Conductivity', lower_bound=0)
    df = prep_hlp.apply_range_filter(df, 'Dissolved Oxygen', lower_bound=0)

    # Interpolate missing values
    df = prep_hlp.resample_and_interpolate(df, res, interpolation_limit=24)

    # Calculate DOY
    df = prep_hlp.calculate_doy(df)
    return df.reset_index()

def prepare_river_monticello_data(data_folder: str, 
                                  out_data_folder: str, 
                                  res: str):
    """
    Orchestrates the data preprocessing for River Monticello data,
    merging flux and river data, calculating derived variables, and saving the
    final DataFrame to a CSV.

    Args:
        data_folder (str): The path to the folder containing the raw data files.
        out_data_folder (str): The path to the folder where the output CSV file will be saved.
        res (str): The resampling frequency string (e.g., '1H', '30T').
    """
    df_flux = load_and_resample_flux_data(data_folder, res)
    df_river = load_and_process_river_data(data_folder, res)

    # Merge flux and river data
    df = pd.merge(df_river, df_flux, on='Date', how='outer')
    df = df.set_index('Date')

    # Filter by datetime range
    df = prep_hlp.filter_by_datetime_range(
        df,
        start_date=dt.datetime(2021, 1, 1),
        end_date=dt.datetime(2023, 12, 31)
    )

    # Calculate cumulative precipitation sums
    df = prep_hlp.calculate_rolling_sums(df, ['Precip_Tot', 'Precip_gage'], [1, 3, 7, 14])

    # Further calculations specific to this script
    df['Q_liters'] = df['Discharge']
    df['LogQ'] = np.log10(df['Discharge'])
    df['LogQ10'] = np.log10(df['Discharge'].rolling(24*10, min_periods=24).mean())

    # Calculate loads
    colnames_responses = ['Calcium', 'Magnesium', 'Potassium', 'Sodium', 'Chlorides', 'Nitrates', 'Sulfates']
    colnames_loads = []
    for c in colnames_responses:
        if c in df.columns and 'Q_liters' in df.columns:
            df[c + 'Load_g'] = df[c] * df['Q_liters']
            colnames_loads.append(c + 'Load_g')

    # Define final columns and save
    colnames_drivers = ['Discharge', 'LogQ', 'Precip_1D', 'Precip_3D', 'Precip_7D', 'Precip_14D',
                        'D5TE_VWC_100cm_Avg', 'TempRiver', 'Turbidity', 'Dissolved Oxygen']
    
    final_cols = [col for col in colnames_responses + colnames_loads + colnames_drivers + ['DOY'] if col in df.columns]
    df = df[final_cols].copy()
    df = df.dropna()

    df.to_csv(out_data_folder + 'Data_MonticelloRiver_Hourly.csv')