#!/usr/bin/env python
# coding: utf-8

# RiverLab Case: Orgeval watershed, France

# import analysis helper script
import _02_analyze.scripts.Analysis_Helpers as analysis_hlp

def analyze_river_orgeval(input_file: str,
                          output_folder: str,
                          nc=9,
                          seed=42):
    """
    Performs the analysis for River Monticello data, including
    data loading, scaling, GMM clustering, and calculating conditional probabilities
    based on flow ranges.

    Args:
        input_file (str): Path to the folder containing the data file.
        output_folder (str): Folder to save the CSV files.
        nc (int): Number of clusters for GMM. Defaults to 9.
        seed (int): Random seed for reproducibility.
    """

    # Define columns for analysis
    colnames_responses = ['Calcium', 'Magnesium', 'Potassium', 'Sodium', 'Chlorures', 'Nitrates', 'Sulfates']
    colnames_drivers = ['LogQ', 'LogQ10', 'Temp_C', 'SpCond_uScm', 'DO_mgL', 'pH', 'Turbidity_FNU']

    # Define scaler
    scaler_type = 'MinMaxScaler'

    # Define date column
    date_column='Date'

    # Analyze and export site data
    df_filtered, df_cluster_means, df_hourly_prob, df_monthly_prob, df_yearly_prob, colnames_responses, colnames_drivers = analysis_hlp.analyze_site_data(input_file, 'river', output_folder, date_column, colnames_responses, colnames_drivers, scaler_type, nc, seed)

    # export site data
    analysis_hlp.export_analysis_results('river_monticello', df_filtered, df_cluster_means, df_hourly_prob, df_monthly_prob, df_yearly_prob, colnames_responses, colnames_drivers, output_folder)
