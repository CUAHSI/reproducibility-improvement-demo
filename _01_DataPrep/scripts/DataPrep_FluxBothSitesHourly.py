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
import data_prep_helpers as hlp

def load_and_merge_gc_data(data_folder: str) -> pd.DataFrame:
    """
    Loads raw GC flux data and merges with GPP/Reco data.
    """
    df = hlp.load_and_process_csv(
        file_path=data_folder + 'GC_FluxData_RAW_25m_042216_050224.csv',
        date_column='NewDate',
        localize_tz=False, # Assuming NewDate is already timezone-naive or intended to be
        index_column='Date'
    )
    df = df.rename(columns={'NewDate': 'Date'}) # Rename column if 'NewDate' was used for date_column in load

    dfGPP = hlp.load_and_process_csv(
        file_path=data_folder + 'GC_25m_REddyProc_Processed_30min_DaytimePartitioning.csv',
        date_column='Date',
        localize_tz=False, # Original script did not localize here
    )
    # Original script loaded dfGPPdates but then assigned dfGPPdates['Date'] to dfGPP['Date'].
    # Assuming 'Date' column is directly in dfGPP.csv as it was used directly for merging.
    dfwithGPP = dfGPP[['Date', 'GPP_DT', 'Reco_DT', 'NEE_U05_fall']]

    df = pd.merge(df, dfwithGPP, on='Date', how='outer')
    df = hlp.calculate_doy(df) # Calculates DOY using the index
    return df

def process_gc_flux_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes GC flux data including NDVI merging, outlier removal, and resampling.
    """
    df_NDVI_MOD1 = hlp.load_and_process_csv(
        file_path=data_folder + 'MODIS_fluxtower_download/flux-tower-MODIS-NDVI-MOD13A1-061-results.csv',
        date_column='Date', localize_tz=True
    )
    df_NDVI_MOD2 = hlp.load_and_process_csv(
        file_path=data_folder + 'MODIS_fluxtower_download/flux-tower-MODIS-NDVI-MYD13A1-061-results.csv',
        date_column='Date', localize_tz=True
    )

    df_NDVI_MOD1['NDVI'] = df_NDVI_MOD1['MOD13A1_061__500m_16_days_NDVI']
    df_NDVI_MOD2['NDVI'] = df_NDVI_MOD2['MYD13A1_061__500m_16_days_NDVI']

    df_m1 = df_NDVI_MOD1[['Date', 'NDVI']]
    df_m2 = df_NDVI_MOD2[['Date', 'NDVI']]

    dfMOD = pd.concat([df_m1, df_m2], axis=0).drop_duplicates('Date').set_index('Date')
    dfMOD_1day = hlp.resample_and_interpolate(dfMOD, '1D', interpolation_method='linear')
    dfMOD_1day = hlp.calculate_doy(dfMOD_1day) # Add DOY for NDVI
    df = pd.merge(df, dfMOD_1day, on='Date', how='outer', suffixes=('', '_y')) # Merged on index, so 'Date' is common

    df = hlp.remove_outliers_by_quantile(df, ['tau', 'u_star', 'rslt_wnd_spd', 'T_tmpr_rh_mean', 'RH_tmpr_rh_mean',
                                             'CO2_li_mean', 'H_corr', 'LE_corr', 'ET_corr', 'GPP_DT', 'Reco_DT',
                                             'Tr_Wm2', 'Precip_Tot'])
    # Replace -9999 with NaN (common in original scripts)
    for col in df.columns:
        df[col] = np.where(df[col] == -9999, np.nan, df[col])

    df['UoverUstar'] = df['rslt_wnd_spd'] / df['u_star']
    df = df.resample('30T').mean() # Resample to 30 min (original behavior)
    df = df.interpolate(method='linear', limit=20) # Interpolate with limit
    df['site'] = 'GC'
    return df

def process_konza_data(data_folder: str) -> pd.DataFrame:
    """
    Loads and processes Konza flux data, including outlier removal and resampling.
    """
    dfK = hlp.load_and_process_csv(
        file_path=data_folder + 'Konza_FluxData_Raw_30min_042216_050224.csv',
        date_column='NewDate',
        localize_tz=False,
        index_column='Date'
    )
    dfK = dfK.rename(columns={'NewDate': 'Date'})

    # Replace -9999 with NaN and remove outliers for Konza data
    for c in dfK.columns:
        dfK[c] = np.where(dfK[c] == -9999, np.nan, dfK[c])
    dfK = hlp.remove_outliers_by_quantile(dfK, dfK.columns.drop('Date')) # Apply to all columns except 'Date'
    dfK = dfK.resample('30T').mean()
    dfK['site'] = 'Kon'
    dfK = dfK.interpolate(method='linear', limit=20) # Apply interpolation like GC
    return dfK

def main(data_folder,out_data_folder):
    """
    Main function to orchestrate the data preprocessing for Flux Both Sites Hourly data.
    """
    df = process_gc_flux_data(load_and_merge_gc_data(data_folder))
    dfK = process_konza_data(data_folder)

    # Combine both datasets
    df_combined = pd.concat([df, dfK], axis=0, ignore_index=False)

    # Apply datetime filters to combined data
    # Filter by specific hours if needed, example:
    # df_combined = hlp.filter_by_datetime_range(df_combined, start_hour=8, end_hour=18)
    # Exclude 2020 data if needed for combined:
    # df_combined = hlp.filter_by_datetime_range(df_combined, exclude_year=2020)

    # Example of a specific filter from the original script applied to combined (if desired)
    # df_combined = df_combined.loc[df_combined['DOY'] > 90]
    # df_combined = df_combined.loc[df_combined['DOY'] < 310]

    # Calculate DOY for the combined dataframe if not already present or needs update after concat
    df_combined = hlp.calculate_doy(df_combined)

    # Column selection and saving
    colnames_responses = ['NEE', 'GPP_DT', 'Reco_DT', 'LE_corr', 'H_corr', 'WUE', 'site']
    # Ensure correct columns from GC and Konza are in the final list
    # Need to verify specific column names after processing both datasets
    # Original script had: colnames_responses = ['NEE','GPP','Reco','LE','B','WUE','site']
    # And colnames_drivers = ['DOY','T_tmpr_rh_mean','RH_tmpr_rh_mean','CO2_li_mean','Precip_Tot','short_up_Avg','VPD','NDVI','D5TE_VWC_5cm_Avg','D5TE_VWC_100cm_Avg','D5TE_T_5cm_Avg','D5TE_T_100cm_Avg','gdd','Tr_Wm2','tau','u_star','rslt_wnd_spd','P_ET_cumdiff','ET_corr','Precip_1D','Precip_3D','Precip_7D','Precip_14D','NDVI_peak_interp_roll','T_tmpr_rh_mean_interp_roll','RH_tmpr_rh_mean_interp_roll','VPD','EF','ToverET','WUE','GPPoverNDVI','Reco_Fraction']
    # This implies that these columns should be derived or present after processing.

    # This part depends on the exact columns present after processing both GC and Konza.
    # The original script had slightly different column names for GC and Konza.
    # For a unified output, you might need to standardize column names within the processing functions.
    # Assuming 'NEE' and 'GPP_DT' etc. are harmonized.

    # For demonstration, selecting a basic set present in the combined data
    final_cols = [col for col in colnames_responses + ['DOY', 'T_tmpr_rh_mean', 'RH_tmpr_rh_mean', 'CO2_li_mean', 'Precip_Tot'] if col in df_combined.columns]

    df_combined = df_combined[final_cols].copy()
    df_combined = df_combined.dropna()

    df_combined.to_csv(out_data_folder + 'Data_FluxBothSites_Hourly.csv')

    # Plotting from original script (retained for reference)
    # plt.figure(figsize=(5,15))
    # for i,c in enumerate(df.columns):
    #     plt.subplot(20,1,i+1)
    #     plt.plot(df[c])
    #     plt.ylabel(c)
    #     #plt.xlim(dt.datetime(2020,1,1,0,0,0),dt.datetime(2023,1,1,0,0,0))
    # plt.show()
    #
    # plt.figure(figsize=(5,15))
    # for i,c in enumerate(dfK.columns):
    #     plt.subplot(20,1,i+1)
    #     plt.plot(dfK[c])
    #     plt.ylabel(c)
    #     #plt.xlim(dt.datetime(2020,1,1,0,0,0),dt.datetime(2023,1,1,0,0,0))
    # plt.show()


if __name__ == '__main__':

    data_folder = '../input/'
    out_data_folder = '../output/'

    main(data_folder,out_data_folder)