import pandas as pd
import numpy as np
import _02_Analysis.scripts.cluster_funcs as cf
from _02_Analysis.scripts.analysis_helpers import load_data, scale_data, calculate_conditional_probabilities
import os

def export_flux_both_sites_hourly_results(df_filtered, df_cluster_means, df_monthly_prob, df_hourly_prob,
                                          colnames_responses, colnames_drivers, output_folder):
    """
    Exports the analysis results to CSV files.

    Args:
        df_filtered (pd.DataFrame): The original DataFrame with cluster indices.
        df_cluster_means (pd.DataFrame): DataFrame of cluster means.
        df_monthly_prob (pd.DataFrame): Monthly conditional probabilities.
        df_hourly_prob (pd.DataFrame): Hourly conditional probabilities.
        colnames_responses (list): List of response column names.
        colnames_drivers (list): List of driver column names.
        output_folder (str): Folder to save the CSV files. Defaults to 'DATA/AnalysisResults/'.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Save df_filtered (main analysis results)
    df_filtered.to_csv(os.path.join(output_folder, 'flux_both_sites_hourly_filtered_data.csv'), index=True)

    # Save cluster means
    df_cluster_means.to_csv(os.path.join(output_folder, 'flux_both_sites_hourly_cluster_means.csv'), index=True)

    # Save monthly probabilities
    df_monthly_prob.to_csv(os.path.join(output_folder, 'flux_both_sites_hourly_monthly_prob.csv'), index=False)

    # Save hourly probabilities
    df_hourly_prob.to_csv(os.path.join(output_folder, 'flux_both_sites_hourly_hourly_prob.csv'), index=False)

    # Save column names as a simple text file or JSON if more complex structure needed
    with open(os.path.join(output_folder, 'flux_both_sites_hourly_colnames.txt'), 'w') as f:
        f.write("colnames_responses=" + str(colnames_responses) + "\n")
        f.write("colnames_drivers=" + str(colnames_drivers) + "\n")
    print(f"Analysis results saved to {output_folder}")


def run_flux_both_sites_hourly_analysis(data_folder, output_folder, nc=7,
                                         datafile_name='ProcessedData_GCKonTowers30min.csv',
                                         seed=42):
    """
    Performs the analysis for Flux Tower Both Sites Hourly data, including
    data loading, scaling, GMM clustering, and calculating conditional probabilities.

    Args:
        nc (int): Number of clusters for GMM. Defaults to 7.
        data_folder (str): Path to the folder containing the data file.
        datafile_name (str): Name of the CSV data file.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
               - df_filtered (pd.DataFrame): The original DataFrame filtered for NaNs,
                                             with 'balance_idx' and 'site_idx' added.
               - df_cluster_means (pd.DataFrame): DataFrame of cluster means for response variables.
               - df_monthly_prob (pd.DataFrame): Conditional probabilities of clusters by month.
               - df_hourly_prob (pd.DataFrame): Conditional probabilities of clusters by hour.
               - colnames_responses (list): List of response column names used in analysis.
               - colnames_drivers (list): List of driver column names used in analysis.
    """
    # Load data
    df = load_data(data_folder, datafile_name, date_column='Date')

    # Define columns for analysis
    colnames_responses = ['NEE', 'GPP', 'Reco', 'LE', 'B', 'WUE']
    colnames_drivers = ['short_up_Avg', 'RH_tmpr_rh_mean', 'D5TE_T_5cm_Avg',
                        'D5TE_T_10cm_Avg', 'D5TE_T_20cm_Avg', 'D5TE_T_50cm_Avg',
                        'D5TE_T_100cm_Avg', 'LST_GC', 'LST_Kon', 'NDVI_GC', 'NDVI_Kon',
                        'SWC_5cm_GC', 'SWC_10cm_GC', 'SWC_20cm_GC', 'SWC_50cm_GC',
                        'SWC_100cm_GC', 'SWC_5cm_Kon', 'SWC_10cm_Kon', 'SWC_20cm_Kon',
                        'SWC_50cm_Kon', 'SWC_100cm_Kon', 'Precip_GC', 'Precip_Kon']

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

    # Add site index based on 'site' column
    df_filtered['site_idx'] = df_filtered['site'].astype('category').cat.codes + 1

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


    # Calculate conditional probabilities by hour
    df_filtered['hour'] = df_filtered.index.hour
    hours = np.unique(df_filtered.index.hour)
    df_hourly_prob = calculate_conditional_probabilities(df_filtered, 'balance_idx', 'hour', hours)
    df_hourly_prob['Hour'] = [str(h) for h in hours] # Ensure hour column is string for plotting

    export_flux_both_sites_hourly_results(df_filtered, df_cluster_means, df_monthly_prob, df_hourly_prob,
                                          colnames_responses, colnames_drivers,output_folder)

if __name__ == '__main__':

    nc_value = 7 # Example number of clusters
    run_flux_both_sites_hourly_analysis(nc=nc_value)