#!/usr/bin/env python
# coding: utf-8

# RiverLab Case: Plynlimon, UK, Upper Hafren 7-hour observations

# import analysis helper script
import _02_analyze.scripts.Analysis_Helpers as analysis_hlp

def analyze_river_plynlimon(input_file: str,
                            output_folder: str,
                            nc=9,
                            seed=42):
    """
    Performs the analysis for River Plynlimon data, including
    data loading, scaling, GMM clustering, and calculating conditional probabilities
    by hour, month, and year.

    Args:
        input_file (str): Path to the folder containing the data file.
        output_folder (str): Folder to save the CSV files.
        nc (int): Number of clusters for GMM. Defaults to 9.
        seed (int): Random seed for reproducibility.
    """

    # Define columns for analysis
    colnames_responses = ['Ca mg/l', 'Mg mg/l', 'K mg/l', 'NO3-N mg/l', 'Cl mg/l', 'Na mg/l', 'SO4 mg/l']
    colnames_drivers = ['Temp C', 'pH', 'Cond uS/cm', 'DO mg/l', 'Turbidity NTU']

    # Define scaler
    scaler_type = 'MinMaxScaler'

    # Define date column
    date_column='date_time'

    # Analyze and export site data
    df_filtered, df_cluster_means, df_hourly_prob, df_monthly_prob, df_yearly_prob, colnames_responses, colnames_drivers = analysis_hlp.analyze_site_data(input_file, 'river', output_folder, date_column, colnames_responses, colnames_drivers, scaler_type, nc, seed)

    # export site data
    analysis_hlp.export_analysis_results('river_plynlimon', df_filtered, df_cluster_means, df_hourly_prob, df_monthly_prob, df_yearly_prob, colnames_responses, colnames_drivers, output_folder)