#!/usr/bin/env python
# coding: utf-8

# Code to pre-process data for flux tower case - concatenate datasets, gap fill, etc
# save dataframe as csv that is input into the GMM-PCA-IT framework

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
from matplotlib.colors import ListedColormap
# Import the helper script
import _01_DataPrep.scripts.data_prep_helpers as hlp

def load_and_initial_process_flux_data(data_folder: str) -> pd.DataFrame:
    """
    Loads initial flux data and performs basic date processing and DOY calculation.

    Args:
        data_folder (str): Path to the data folder.

    Returns:
        pd.DataFrame: Processed flux data DataFrame.
    """
    df = hlp.load_and_process_csv(
        file_path=data_folder + 'Corrected_Daily_25m_2016_2023.csv',
        date_column='Date',
        localize_tz=True
    )
    df = hlp.calculate_doy(df, date_column='Date')
    return df

def process_raw_flux_data(data_folder: str) -> pd.DataFrame:
    """
    Loads and processes raw flux data, including outlier removal and resampling.

    Args:
        data_folder (str): Path to the data folder.

    Returns:
        pd.DataFrame: Processed raw flux data DataFrame.
    """
    df_raw = hlp.load_and_process_csv(
        file_path=data_folder + 'FluxData_Raw_ALL.csv',
        date_column='NewDate',
        use_cols=['NewDate', 'tau', 'u_star', 'rslt_wnd_spd', 'T_tmpr_rh_mean'],
        localize_tz=False # NewDate might not need tz_localize(None) based on context
    )

    # Remove outliers using quantiles
    df_raw = hlp.remove_outliers_by_quantile(df_raw, ['tau'], lower_quantile=0, upper_quantile=0.995)
    df_raw = hlp.remove_outliers_by_quantile(df_raw, ['u_star'], lower_quantile=0, upper_quantile=0.95)
    df_raw = hlp.remove_outliers_by_quantile(df_raw, ['rslt_wnd_spd'], lower_quantile=0, upper_quantile=0.995)

    df_raw['UoverUstar'] = df_raw['rslt_wnd_spd'] / df_raw['u_star']

    # Resample to daily and get min/max temperature
    df_raw_1D = hlp.resample_and_interpolate(df_raw, '1D', on_column='NewDate')
    df_raw_maxvals = df_raw.resample('1D', on='NewDate').max()
    df_raw_minvals = df_raw.resample('1D', on='NewDate').min()

    df_raw_1D['Ta_min'] = df_raw_minvals['T_tmpr_rh_mean']
    df_raw_1D['Ta_max'] = df_raw_maxvals['T_tmpr_rh_mean']
    df_raw_1D['Date'] = df_raw_1D.index
    return df_raw_1D.reset_index(drop=True)

def process_ndvi_data(data_folder: str) -> pd.DataFrame:
    """
    Loads and processes MODIS NDVI data from two sources and combines them.

    Args:
        data_folder (str): Path to the data folder.

    Returns:
        pd.DataFrame: Processed NDVI DataFrame.
    """
    df_NDVI_MOD1 = hlp.load_and_process_csv(
        file_path=data_folder + 'MODIS_fluxtower_download/flux-tower-MODIS-NDVI-MOD13A1-061-results.csv',
        date_column='Date',
        localize_tz=True
    )
    df_NDVI_MOD2 = hlp.load_and_process_csv(
        file_path=data_folder + 'MODIS_fluxtower_download/flux-tower-MODIS-NDVI-MYD13A1-061-results.csv',
        date_column='Date', # Original script used df_NDVI_MOD1['Date'] here, assuming it should be df_NDVI_MOD2['Date']
        localize_tz=True
    )

    df_NDVI_MOD1['NDVI'] = df_NDVI_MOD1['MOD13A1_061__500m_16_days_NDVI']
    df_NDVI_MOD2['NDVI'] = df_NDVI_MOD2['MYD13A1_061__500m_16_days_NDVI']

    df_m1 = df_NDVI_MOD1[['Date', 'NDVI']]
    df_m2 = df_NDVI_MOD2[['Date', 'NDVI']]

    dfMOD = pd.concat([df_m1, df_m2], axis=0).drop_duplicates('Date')
    dfMOD = dfMOD.set_index('Date')

    # Interpolate to daily frequency
    dfMOD_1day = hlp.resample_and_interpolate(dfMOD, '1D', interpolation_method='linear')
    return dfMOD_1day.reset_index()

def main(data_folder,out_data_folder):
    """
    Main function to orchestrate the data preprocessing for FluxGC Daily data.
    """
    df_main = load_and_initial_process_flux_data(data_folder)
    df_raw = process_raw_flux_data(data_folder)
    df_ndvi = process_ndvi_data(data_folder)

    # Merge dataframes
    df = pd.merge(df_main, df_ndvi, on='Date', how='outer')
    df = pd.merge(df, df_raw, on='Date', how='outer')

    # Set index before filtering with datetime functions
    df = df.set_index('Date')

    # Filter by date range and exclude 2020 using the new combined function
    start_date = dt.datetime(2016, 4, 15, 0, 0, 0)
    end_date = dt.datetime(2022, 12, 31, 0, 0, 0)
    df = hlp.filter_by_datetime_range(
        df,
        start_date=start_date,
        end_date=end_date,
        exclude_year=2020
    )

    # Interpolate remaining missing values
    df = df.interpolate(method='time')

    # Filter by Day of Year (this remains as it's not a datetime component directly)
    df = df.loc[df['DOY_x'] > 90]
    df = df.loc[df['DOY_x'] < 310]

    print(list(df.columns))

    # Calculate derived variables
    df['gdd'] = (df['Ta_max'] + df['Ta_min']) / 2 - 18
    df['gdd'] = hlp.apply_range_filter(df, 'gdd', lower_bound=0, replace_value=0)

    df = hlp.calculate_cumulative_sum_by_year(df, 'gdd', 'gdd')
    df = hlp.calculate_cumulative_sum_by_year(df, 'Precip_Tot', 'Precip_cum')
    df = hlp.calculate_cumulative_sum_by_year(df, 'ET_corr', 'ET_cum')
    df['P_ET_cumdiff'] = df['Precip_cum'] - df['ET_cum']

    df = hlp.calculate_rolling_sums(df, ['Precip_Tot'], [14, 7, 3, 1])

    # Further calculations and outlier removals (specific to this script)
    # NDVI_peak is tower values (peak of daytime values)
    df['NDVI_peak_interp_roll'] = df['NDVI'].rolling(5, min_periods=1).mean()
    df['NDVI_peak_interp_roll'] = hlp.apply_range_filter(df, 'NDVI_peak_interp_roll', lower_bound=0)

    df['T_tmpr_rh_mean_interp_roll'] = df['T_tmpr_rh_mean'].rolling(5, min_periods=1).mean()
    df['T_tmpr_rh_mean_interp_roll'] = hlp.remove_outliers_by_quantile(df, ['T_tmpr_rh_mean_interp_roll'], lower_quantile=0.005, upper_quantile=0.995)['T_tmpr_rh_mean_interp_roll']

    df['RH_tmpr_rh_mean_interp_roll'] = df['RH_tmpr_rh_mean'].rolling(5, min_periods=1).mean()
    df['RH_tmpr_rh_mean_interp_roll'] = hlp.remove_outliers_by_quantile(df, ['RH_tmpr_rh_mean_interp_roll'], lower_quantile=0.005, upper_quantile=0.995)['RH_tmpr_rh_mean_interp_roll']

    df['VPD'] = 0.611 * np.exp((17.27 * df['T_tmpr_rh_mean_interp_roll']) / (237.3 + df['T_tmpr_rh_mean_interp_roll'])) * (1 - df['RH_tmpr_rh_mean_interp_roll'] / 100)
    df['VPD'] = hlp.apply_range_filter(df, 'VPD', lower_bound=0)

    # More derived variables and cleaning
    df['EF'] = df['LE_corr'] / (df['LE_corr'] + df['H_corr'])
    df['EF'] = hlp.apply_range_filter(df, 'EF', lower_bound=0, upper_bound=1)

    df['ToverET'] = (df['Tr_Wm2']) / df['ET_corr']
    df['ToverET'] = hlp.apply_range_filter(df, 'ToverET', lower_bound=0)

    df['WUE'] = df['GPP_DT'] / df['LE_corr']
    df['WUE'] = hlp.apply_range_filter(df, 'WUE', lower_bound=0)
    df = hlp.remove_outliers_by_quantile(df, ['WUE'])

    df['GPPoverNDVI'] = df['GPP_DT'] / df['NDVI']
    df = hlp.remove_outliers_by_quantile(df, ['GPPoverNDVI'])

    df['Reco_Fraction'] = df['Reco_DT'] / (df['Reco_DT'] + np.abs(df['GPP_DT']))

    # Final column selections and saving
    colnames_responses = ['NEE', 'Reco_Fraction', 'ToverET', 'WUE', 'NDVI', 'GPPoverNDVI', 'EF', 'GPP_DT', 'Reco_DT', 'H_corr', 'ET_corr', 'GPP_DT', 'Reco_DT', 'Tr_Wm2', 'tau', 'UoverUstar', 'DOY']
    colnames_drivers = ['Precip_14D', 'Precip_7D', 'Precip_3D', 'Precip_1D', 'gdd', 'T_tmpr_rh_mean_x', 'D5TE_VWC_5cm_Avg', 'D5TE_VWC_100cm_Avg', 'D5TE_T_5cm_Avg', 'D5TE_T_100cm_Avg', 'short_up_Avg', 'CO2_li_mean', 'RH_tmpr_rh_mean_x', 'VPD', 'NDVI_peak_interp_roll', 'T_tmpr_rh_mean_interp_roll', 'RH_tmpr_rh_mean_interp_roll']

    df = df[colnames_responses + colnames_drivers].copy()
    df = df.dropna()

    df.to_csv(out_data_folder + 'Data_GCFluxTower_Daily.csv')

if __name__ == '__main__':

    data_folder = '../input/'
    out_data_folder = '../output/'

    main(data_folder,out_data_folder)