#!/usr/bin/env python
# coding: utf-8

# # Flux tower seasonal regimes

#Initial setup

# import analysis helper script
import _02_analyze.scripts.Analysis_Helpers as analysis_hlp

def analyze_flux_both_sites_hourly(input_file: str,
                                   output_folder: str,
                                   nc=9,
                                   seed=42):
    """
    Performs the analysis for Flux GC daily data, including
    data loading, scaling, GMM clustering, and calculating conditional probabilities
    by hour, month, and year.

    Args:
        input_file (str): Path to the folder containing the data file.
        output_folder (str): Folder to save the CSV files.
        nc (int): Number of clusters for GMM. Defaults to 9.
        seed (int): Random seed for reproducibility.
    """

    # Define columns for analysis
    colnames_responses = ['NEE', 'GPP', 'Reco', 'LE', 'B', 'WUE']
    colnames_drivers = ['short_up_Avg', 'RH_tmpr_rh_mean', 'D5TE_T_5cm_Avg',
                        'D5TE_T_10cm_Avg', 'D5TE_T_20cm_Avg', 'D5TE_T_50cm_Avg',
                        'D5TE_T_100cm_Avg', 'LST_GC', 'LST_Kon', 'NDVI_GC', 'NDVI_Kon',
                        'SWC_5cm_GC', 'SWC_10cm_GC', 'SWC_20cm_GC', 'SWC_50cm_GC',
                        'SWC_100cm_GC', 'SWC_5cm_Kon', 'SWC_10cm_Kon', 'SWC_20cm_Kon',
                        'SWC_50cm_Kon', 'SWC_100cm_Kon', 'Precip_GC', 'Precip_Kon']

    # Define scaler
    scaler_type = 'MinMaxScaler'

    # Define date column
    date_column='Date'

    # Analyze and export site data
    df_filtered, df_cluster_means, df_hourly_prob, df_monthly_prob, df_yearly_prob, colnames_responses, colnames_drivers = analysis_hlp.analyze_site_data(input_file, 'flux', output_folder, date_column, colnames_responses, colnames_drivers, scaler_type, nc, seed)

    # export site data
    analysis_hlp.export_analysis_results('flux_gc_daily', df_filtered, df_cluster_means, df_hourly_prob, df_monthly_prob, df_yearly_prob, colnames_responses, colnames_drivers, output_folder)