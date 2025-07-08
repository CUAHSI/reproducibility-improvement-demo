#!/usr/bin/env python
# coding: utf-8

# Code to pre-process data for riverlab case - concatenate datasets, gap fill, etc
# save dataframe as csv that is input into the GMM-PCA-IT framework

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
from matplotlib.colors import ListedColormap
import _01_DataPrep.scripts.data_prep_helpers as hlp

def load_and_resample_flux_data(fluxdata_folder: str, res: str) -> pd.DataFrame:
    """
    Loads and resamples flux tower data relevant for river analysis.
    """
    df_fluxtower = hlp.load_and_process_csv(
        file_path=fluxdata_folder + 'FluxData_15min_2021_2022.csv',
        date_column='Date',
        localize_tz=True
    )
    df_fluxtower = df_fluxtower[['Date', 'Precip_Tot', 'D5TE_VWC_5cm_Avg', 'D5TE_VWC_100cm_Avg', 'LE_li_wpl']]
    
    # Resample to desired resolution (e.g., 30min)
    df_flux_resampled = hlp.resample_and_interpolate(
        df_fluxtower.set_index('Date'),
        resample_freq=res,
        interpolation_limit=24 # Original script used 24 for 30min data.
    )

    # Smoother version of LE
    df_flux_resampled['LE_li_wpl_smooth'] = df_flux_resampled['LE_li_wpl'].rolling(24 * 3, min_periods=24).mean()
    return df_flux_resampled.reset_index()

def load_and_process_river_data(data_folder: str, res: str) -> pd.DataFrame:
    """
    Loads and processes river discharge and chemistry data.
    """
    df_river_Q = hlp.load_and_process_csv(
        file_path=data_folder + 'RiverData_Hourly_Monticello.csv',
        date_column='Date',
        localize_tz=True
    )

    # Process Temperature, Turbidity, DO
    df_river_temp = hlp.load_and_process_csv(
        file_path=data_folder + 'RiverData_Monticello_Temp_Turb_DO.csv',
        date_column='Date',
        localize_tz=True
    )

    df_river_chem = hlp.load_and_process_csv(
        file_path=data_folder + 'RiverData_Monticello_Chem.csv',
        date_column='Date',
        localize_tz=True
    )
    
    # Merge river dataframes
    df = pd.merge(df_river_Q, df_river_temp, on='Date', how='outer')
    df = pd.merge(df, df_river_chem, on='Date', how='outer')

    # Convert to numeric where necessary (original code did this in a loop)
    # Assuming the merge preserves datetime index if 'Date' is set as index prior to merge or converted after
    df = df.set_index('Date')
    df = df.apply(pd.to_numeric, errors='ignore') # Convert all suitable columns to numeric

    # Replace values outside bounds with NaN, then interpolate
    df = hlp.apply_range_filter(df, 'Discharge', lower_bound=0)
    df = hlp.apply_range_filter(df, 'Precip_gage', lower_bound=0)
    df = hlp.apply_range_filter(df, 'Turbidity', lower_bound=0)
    df = hlp.apply_range_filter(df, 'TempRiver', lower_bound=-10, upper_bound=40)
    df = hlp.apply_range_filter(df, 'Conductivity', lower_bound=0)
    df = hlp.apply_range_filter(df, 'Dissolved Oxygen', lower_bound=0)

    # Interpolate missing values (limit 24 for hourly, original limit was 24, might vary per res)
    df = hlp.resample_and_interpolate(df, res, interpolation_limit=24)

    # Calculate DOY
    df = hlp.calculate_doy(df)
    return df.reset_index()

def main(data_folder,out_data_folder,res):
    """
    Main function to orchestrate the data preprocessing for River Monticello data.
    """
    df_flux = load_and_resample_flux_data(data_folder, res)
    df_river = load_and_process_river_data(data_folder, res)

    # Merge flux and river data
    df = pd.merge(df_river, df_flux, on='Date', how='outer')
    df = df.set_index('Date')

    # Filter by datetime range (e.g., exclude certain years or filter by date)
    # The original script had explicit date filtering, and also `df = df.dropna()`
    # Let's apply a general date range filter as an example:
    df = hlp.filter_by_datetime_range(
        df,
        start_date=dt.datetime(2021, 1, 1),
        end_date=dt.datetime(2023, 12, 31)
    )

    # Calculate cumulative precipitation sums
    df = hlp.calculate_rolling_sums(df, ['Precip_Tot', 'Precip_gage'], [1, 3, 7, 14])

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
                        'D5TE_VWC_100cm_Avg', 'TempRiver', 'Turbidity', 'Dissolved Oxygen'] # Simplified for example
    
    final_cols = [col for col in colnames_responses + colnames_loads + colnames_drivers + ['DOY'] if col in df.columns]
    df = df[final_cols].copy()
    df = df.dropna()

    df.to_csv(out_data_folder + 'Data_MonticelloRiver_Hourly.csv')

    # Plotting from original script (retained for reference)
    # (fig, ax) = plt.subplots(len(colnames_responses), 1, figsize=(12, 12))
    # for i, c in enumerate(colnames_responses):
    #     ax[i].plot(df[c])
    #     ax[i].set_ylabel(c)
    #     ax[i].set_xlim(dt.datetime(2022,1,1),dt.datetime(2023,1,1))
    # plt.show()

if __name__ == '__main__':

    data_folder = '../input/'
    out_data_folder = '../output/'
    res = '30min'

    main(data_folder,out_data_folder,res)