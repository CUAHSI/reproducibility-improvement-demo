#!/usr/bin/env python
# coding: utf-8

# MIRZ case study (toot soil gas concentration gradients: Nebraska prairie and ag sites)

# import analysis helper script
import _02_analyze.scripts.Analysis_Helpers as analysis_hlp

def analyze_root_soil_nebraska(input_file: str,
                               output_folder: str,
                               nc=9,
                               seed=42):
    """
    Performs the analysis for Root Soil Nebraska data, including
    data loading, scaling, GMM clustering, and calculating conditional probabilities
    by hour, month, and year.

    Args:
        input_file (str): Path to the folder containing the data file.
        output_folder (str): Folder to save the CSV files.
        nc (int): Number of clusters for GMM. Defaults to 9.
        seed (int): Random seed for reproducibility.
    """

    # Define columns for analysis
    colnames_responses = ['Cs', 'Cm', 'Cd', 'Os', 'Om', 'Od']
    colnames_drivers = ['TempC_s', 'TempC_m', 'TempC_d', 'VWC_s', 'VWC_m', 'VWC_d']

    # Define scaler
    scaler_type = 'MinMaxScaler'

    # Define date column
    date_column='TIMESTAMP'

    # Analyze and export site data
    df_filtered, df_cluster_means, df_hourly_prob, df_monthly_prob, df_yearly_prob, colnames_responses, colnames_drivers = analysis_hlp.analyze_site_data(input_file, 'soil', output_folder, date_column, colnames_responses, colnames_drivers, scaler_type, nc, seed)

    # export site data
    analysis_hlp.export_analysis_results('root_soil_nebraska', df_filtered, df_cluster_means, df_hourly_prob, df_monthly_prob, df_yearly_prob, colnames_responses, colnames_drivers, output_folder)