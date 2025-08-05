#!/usr/bin/env python
# coding: utf-8

# Code to pre-process data for flux tower case - concatenate datasets, gap fill, etc
# save dataframe as csv that is input into the GMM-PCA-IT framework

# Hourly version of flux tower data

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
from matplotlib.colors import ListedColormap
# import prep helpers script
import _01_DataPrep.scripts.DataPrep_Helpers as prep_hlp 

def load_and_merge_gc_data(data_folder: str) -> pd.DataFrame:
    """
    Loads raw GC flux data and merges with GPP/Reco data.

    Args:
        data_folder (str): The path to the folder containing the raw data CSV files.

    Returns:
        pd.DataFrame: A DataFrame containing the merged GC flux, GPP, and Reco data.
    """
    df = prep_hlp.load_csv(
        file_path=data_folder + 'GC_FluxData_RAW_25m_042216_050224.csv',
        date_column='NewDate',
        localize_tz=False,
        index_column='Date'
    )
    df = df.rename(columns={'NewDate': 'Date'})

    dfGPP = prep_hlp.load_csv(
        file_path=data_folder + 'GC_25m_REddyProc_Processed_30min_DaytimePartitioning.csv',
        date_column='Date',
        localize_tz=False,
    )
    dfwithGPP = dfGPP[['Date', 'GPP_DT', 'Reco_DT', 'NEE_U05_fall']]

    df = pd.merge(df, dfwithGPP, on='Date', how='outer')
    df = prep_hlp.calculate_doy(df)
    return df

def process_gc_flux_data(data_folder: str, 
                         df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes GC flux data including NDVI merging, outlier removal, and resampling.

    Args:
        data_folder (str): The path to the folder containing the raw data CSV files.
        df (pd.DataFrame): The DataFrame containing the raw GC flux data.

    Returns:
        pd.DataFrame: The processed GC flux DataFrame.
    """
    df_NDVI_MOD1 = prep_hlp.load_csv(
        file_path=data_folder + 'MODIS_fluxtower_download/flux-tower-MODIS-NDVI-MOD13A1-061-results.csv',
        date_column='Date', localize_tz=True
    )
    df_NDVI_MOD2 = prep_hlp.load_csv(
        file_path=data_folder + 'MODIS_fluxtower_download/flux-tower-MODIS-NDVI-MYD13A1-061-results.csv',
        date_column='Date', localize_tz=True
    )

    df_NDVI_MOD1['NDVI'] = df_NDVI_MOD1['MOD13A1_061__500m_16_days_NDVI']
    df_NDVI_MOD2['NDVI'] = df_NDVI_MOD2['MYD13A1_061__500m_16_days_NDVI']

    df_m1 = df_NDVI_MOD1[['Date', 'NDVI']]
    df_m2 = df_NDVI_MOD2[['Date', 'NDVI']]

    dfMOD = pd.concat([df_m1, df_m2], axis=0).drop_duplicates('Date').set_index('Date')
    dfMOD_1day = prep_hlp.resample_and_interpolate(dfMOD, resample_freq='1D', interpolation_method='linear')
    dfMOD_1day = prep_hlp.calculate_doy(dfMOD_1day)
    df = pd.merge(df, dfMOD_1day, on='Date', how='outer', suffixes=('', '_y'))

    df = prep_hlp.remove_outliers_by_quantile(df, ['tau', 'u_star', 'rslt_wnd_spd', 'T_tmpr_rh_mean', 'RH_tmpr_rh_mean',
                                             'CO2_li_mean', 'H_corr', 'LE_corr', 'ET_corr', 'GPP_DT', 'Reco_DT',
                                             'Tr_Wm2', 'Precip_Tot'])
    for col in df.columns:
        df[col] = np.where(df[col] == -9999, np.nan, df[col])

    df['UoverUstar'] = df['rslt_wnd_spd'] / df['u_star']
    df = df.resample('30T').mean()
    df = df.interpolate(method='linear', limit=20)
    df['site'] = 'GC'
    return df

def process_konza_data(data_folder: str) -> pd.DataFrame:
    """
    Loads and processes Konza flux data, including outlier removal and resampling.

    Args:
        data_folder (str): The path to the folder containing the Konza data CSV file.

    Returns:
        pd.DataFrame: The processed Konza flux DataFrame.
    """
    dfK = prep_hlp.load_csv(
        file_path=data_folder + 'Konza_FluxData_Raw_30min_042216_050224.csv',
        date_column='NewDate',
        localize_tz=False,
        index_column='Date'
    )
    dfK = dfK.rename(columns={'NewDate': 'Date'})

    for c in dfK.columns:
        dfK[c] = np.where(dfK[c] == -9999, np.nan, dfK[c])
    dfK = prep_hlp.remove_outliers_by_quantile(dfK, dfK.columns.drop('Date'))
    dfK = dfK.resample('30T').mean()
    dfK['site'] = 'Kon'
    dfK = dfK.interpolate(method='linear', limit=20)
    return dfK

def prep_hourly_flux_data_both_sites(data_folder: str,
                                     out_data_folder: str):
    """
    Orchestrates the data preprocessing for Flux Both Sites Hourly data,
    merging GC and Konza data, and saving the final processed DataFrame to a CSV.

    Args:
        data_folder (str): The path to the folder containing the raw data files.
        out_data_folder (str): The path to the folder where the output CSV file will be saved.
    """
    df = process_gc_flux_data(load_and_merge_gc_data(data_folder))
    dfK = process_konza_data(data_folder)

    df_combined = pd.concat([df, dfK], axis=0, ignore_index=False)

    df_combined = prep_hlp.calculate_doy(df_combined)

    colnames_responses = ['NEE', 'GPP_DT', 'Reco_DT', 'LE_corr', 'H_corr', 'WUE', 'site']
    final_cols = [col for col in colnames_responses + ['DOY', 'T_tmpr_rh_mean', 'RH_tmpr_rh_mean', 'CO2_li_mean', 'Precip_Tot'] if col in df_combined.columns]

    df_combined = df_combined[final_cols].copy()
    df_combined = df_combined.dropna()

    df_combined.to_csv(out_data_folder + 'Data_FluxBothSites_Hourly.csv')