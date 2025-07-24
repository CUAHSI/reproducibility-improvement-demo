import pandas as pd
import numpy as np
import _02_Analysis.scripts.cluster_funcs as cf
from _02_Analysis.scripts.analysis_helpers import load_data, scale_data, calculate_conditional_probabilities
import os

def export_flux_gc_daily_results(df_filtered, df_cluster_means, df_monthly_prob, df_yearly_prob,
                                 df_wet_dry_prob, colnames_responses, colnames_drivers,
                                 output_folder='DATA/AnalysisResults/'):
    """
    Exports the analysis results to CSV files.

    Args:
        df_filtered (pd.DataFrame): The original DataFrame with cluster indices.
        df_cluster_means (pd.DataFrame): DataFrame of cluster means.
        df_monthly_prob (pd.DataFrame): Monthly conditional probabilities.
        df_yearly_prob (pd.DataFrame): Yearly conditional probabilities.
        df_wet_dry_prob (pd.DataFrame): Wet/Dry year conditional probabilities.
        colnames_responses (list): List of response column names.
        colnames_drivers (list): List of driver column names.
        output_folder (str): Folder to save the CSV files. Defaults to 'DATA/AnalysisResults/'.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Save df_filtered (main analysis results)
    df_filtered.to_csv(os.path.join(output_folder, 'flux_gc_daily_filtered_data.csv'), index=True)

    # Save cluster means
    df_cluster_means.to_csv(os.path.join(output_folder, 'flux_gc_daily_cluster_means.csv'), index=True)

    # Save monthly probabilities
    df_monthly_prob.to_csv(os.path.join(output_folder, 'flux_gc_daily_monthly_prob.csv'), index=False)

    # Save yearly probabilities
    df_yearly_prob.to_csv(os.path.join(output_folder, 'flux_gc_daily_yearly_prob.csv'), index=False)

    # Save wet/dry probabilities
    df_wet_dry_prob.to_csv(os.path.join(output_folder, 'flux_gc_daily_wet_dry_prob.csv'), index=False)

    # Save column names
    with open(os.path.join(output_folder, 'flux_gc_daily_colnames.txt'), 'w') as f:
        f.write("colnames_responses=" + str(colnames_responses) + "\n")
        f.write("colnames_drivers=" + str(colnames_drivers) + "\n")
    print(f"Analysis results saved to {output_folder}")

def run_flux_gc_daily_analysis(nc=6, data_folder='DATA/Processed/',
                               datafile_name='ProcessedData_GCFluxTowerDaily.csv',
                               seed=42):
    """
    Performs the analysis for Flux Tower Goose Creek Daily data, including
    data loading, scaling, GMM clustering, and calculating conditional probabilities.

    Args:
        nc (int): Number of clusters for GMM. Defaults to 6.
        data_folder (str): Path to the folder containing the data file.
        datafile_name (str): Name of the CSV data file.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
               - df_filtered (pd.DataFrame): The original DataFrame filtered for NaNs,
                                             with 'balance_idx' added.
               - df_cluster_means (pd.DataFrame): DataFrame of cluster means for response variables.
               - df_monthly_prob (pd.DataFrame): Conditional probabilities of clusters by month.
               - df_yearly_prob (pd.DataFrame): Conditional probabilities of clusters by year.
               - df_wet_dry_prob (pd.DataFrame): Conditional probabilities of clusters by wet/dry years.
               - colnames_responses (list): List of response column names used in analysis.
               - colnames_drivers (list): List of driver column names used in analysis.
    """
    # Load data
    df = load_data(data_folder, datafile_name, date_column='Date')

    # Define columns for analysis
    colnames_responses = ['GPP_DT', 'Tr_Wm2', 'NDVI']
    colnames_drivers = ['Rn_Wm2', 'Tair_C', 'VPD_kPa', 'SWC_5cm', 'SWC_10cm', 'SWC_20cm',
                        'SWC_50cm', 'SWC_100cm', 'Precip_mm']

    all_cols = colnames_responses + colnames_drivers

    # Filter out rows with NaN values in relevant columns for clustering
    df_filtered = df.dropna(subset=all_cols).copy() # Use .copy() to avoid SettingWithCopyWarning

    # Scale data
    df_scaled, scaler = scale_data(df_filtered, all_cols, scaler_type='MinMaxScaler')

    # Prepare data for GMM
    allvars_responses = [df_scaled[col].values for col in colnames_responses]

    # Run GMM clustering
    gmm_model, balance_idx = cf.GMMfun(allvars_responses, nc, seed)
    df_filtered['balance_idx'] = balance_idx # Add cluster index to original filtered data

    # Calculate cluster means for original (unscaled) response variables
    cluster_means = df_filtered.groupby('balance_idx')[colnames_responses].mean()
    df_cluster_means = cluster_means.T
    df_cluster_means.columns = [f'Cluster {i}' for i in range(1, nc + 1)]

    # Calculate conditional probabilities by month
    df_filtered['month'] = df_filtered.index.month
    months = np.unique(df_filtered.index.month)
    df_monthly_prob = calculate_conditional_probabilities(df_filtered, 'balance_idx', 'month', months)
    # Map month numbers to names, handling cases where not all months are present
    all_month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_monthly_prob['Month'] = [all_month_names[m-1] for m in months]

    # Calculate conditional probabilities by year
    df_filtered['year'] = df_filtered.index.year
    years = np.unique(df_filtered.index.year)
    df_yearly_prob = calculate_conditional_probabilities(df_filtered, 'balance_idx', 'year', years)
    df_yearly_prob['Year'] = [str(y) for y in years]

    # Calculate conditional probabilities for wet vs. dry years
    wet_years = [2016, 2018, 2022]
    dry_years = [2017, 2019, 2021]

    # Create a 'year_type' column
    df_filtered['year_type'] = ''
    df_filtered.loc[df_filtered['year'].isin(wet_years), 'year_type'] = 'Wet'
    df_filtered.loc[df_filtered['year'].isin(dry_years), 'year_type'] = 'Dry'

    year_types = ['Wet', 'Dry']
    # Filter df_filtered to only include 'Wet' or 'Dry' year_types before calculating probabilities
    df_filtered_wet_dry = df_filtered[df_filtered['year_type'].isin(year_types)].copy()
    df_wet_dry_prob = calculate_conditional_probabilities(df_filtered_wet_dry, 'balance_idx', 'year_type', year_types)
    df_wet_dry_prob['Year Type'] = year_types

    export_flux_gc_daily_results(df_filtered, df_cluster_means, df_monthly_prob, df_yearly_prob,
                                    df_wet_dry_prob, colnames_responses, colnames_drivers)



if __name__ == '__main__':
    nc_value = 6 # Example number of clusters
    run_flux_gc_daily_analysis(nc=nc_value)