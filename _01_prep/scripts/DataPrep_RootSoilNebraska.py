#!/usr/bin/env python
# coding: utf-8

# MIRZ data pre-processing for input into clustering and IT algorithm

import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
# import prep helpers script
import _01_DataPrep.scripts.DataPrep_Helpers as prep_hlp

def load_and_process_rootsoil_data(file_path: str, 
                                   site_name: str) -> pd.DataFrame:
    """
    Loads root/soil data, converts timestamp, and performs initial cleaning.

    Args:
        file_path (str): The path to the root/soil data CSV file.
        site_name (str): The name of the site to be assigned to the 'site' column.

    Returns:
        pd.DataFrame: A DataFrame with the loaded and initially cleaned data.
    """
    df = prep_hlp.load_csv(
        file_path=file_path,
        date_column='TIMESTAMP',
        localize_tz=None
    )
    df = df.set_index('TIMESTAMP')

    df = df.loc[df['Cs'] > -100]
    df['site'] = site_name
    
    for col in df.columns.drop('site', errors='ignore'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def prepare_root_soil_nebraska_data(data_folder: str, 
                                    out_data_folder: str):
    """
    Orchestrates the data preprocessing for RootSoil Nebraska data,
    including loading, cleaning, outlier removal, interpolation, and saving
    the final combined DataFrame to a CSV.

    Args:
        data_folder (str): The path to the folder containing the raw data files.
        out_data_folder (str): The path to the folder where the output CSV file will be saved.

    Returns:
        None
    """
    df_ag = load_and_process_rootsoil_data(data_folder + 'NEAG_QAQC.csv', 'Ag')
    df_pr = load_and_process_rootsoil_data(data_folder + 'NEPR_QAQC.csv', 'Pr')

    cols_to_clean = ['Cs', 'Cm', 'Cd', 'SWC_5cm', 'SWC_20cm', 'SWC_50cm', 'SWC_100cm',
                      'Temp_5cm', 'Temp_20cm', 'Temp_50cm', 'Temp_100cm', 'SHF_soil']
    
    df_ag = prep_hlp.remove_outliers_by_quantile(df_ag, cols_to_clean)
    df_pr = prep_hlp.remove_outliers_by_quantile(df_pr, cols_to_clean)

    for c in cols_to_clean:
        if c in df_ag.columns:
            df_ag[c] = df_ag[c].interpolate(method='linear', limit=48)
        if c in df_pr.columns:
            df_pr[c] = df_pr[c].interpolate(method='linear', limit=48)
    
    df_ag_agg = df_ag.groupby(df_ag.index.date).mean()
    df_pr_agg = df_pr.groupby(df_pr.index.date).mean()

    df_ag_agg.index = pd.to_datetime(df_ag_agg.index)
    df_pr_agg.index = pd.to_datetime(df_pr_agg.index)
    
    df_ag_agg = prep_hlp.calculate_doy(df_ag_agg)
    df_pr_agg = prep_hlp.calculate_doy(df_pr_agg)

    df_ag_agg = df_ag_agg.drop('site', axis=1, errors='ignore')
    df_pr_agg = df_pr_agg.drop('site', axis=1, errors='ignore')

    df_ag_agg['site'] = 'Ag'
    df_pr_agg['site'] = 'Pr'

    df_combined = pd.concat([df_ag_agg, df_pr_agg], axis=0, ignore_index=False)
    df_combined = df_combined.dropna()

    df_combined.to_csv(out_data_folder + 'Data_RootSoilNebraska_Daily.csv')