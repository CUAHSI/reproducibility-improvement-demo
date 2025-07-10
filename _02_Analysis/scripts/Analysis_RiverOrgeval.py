import pandas as pd
import numpy as np
import cluster_funcs as cf
from _02_Analysis.scripts.analysis_helpers import load_data, scale_data, calculate_flow_conditional_probabilities
import os

def run_river_orgeval_analysis(nc=6, data_folder='DATA/Processed/',
                               datafile_name='ProcessedData_RiverOrgeval.csv',
                               seed=42):
    """
    Performs the analysis for River Orgeval data, including
    data loading, scaling, GMM clustering, and calculating conditional probabilities
    based on flow ranges.

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
               - df_flow_prob (pd.DataFrame): Conditional probabilities of clusters by flow quantile.
               - colnames_responses (list): List of response column names used in analysis.
               - colnames_drivers (list): List of driver column names used in analysis.
    """
    # Load data
    df = load_data(data_folder, datafile_name, date_column='Date')

    # Define columns for analysis
    colnames_responses = ['Calcium', 'Magnesium', 'Potassium', 'Sodium', 'Chlorures', 'Nitrates', 'Sulfates']
    colnames_drivers = ['LogQ', 'LogQ10', 'Temp_C', 'SpCond_uScm', 'DO_mgL', 'pH', 'Turbidity_FNU']

    all_cols = colnames_responses + colnames_drivers + ['discharge'] # Include discharge for flow analysis

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

    # Calculate conditional probabilities by flow ranges
    flow_ranges = np.asarray(df_filtered['discharge'].quantile([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]))
    df_flow_prob = calculate_flow_conditional_probabilities(df_filtered, 'balance_idx', 'discharge', flow_ranges)

    return df_filtered, df_cluster_means, df_flow_prob, colnames_responses, colnames_drivers

def export_river_orgeval_results(df_filtered, df_cluster_means, df_flow_prob,
                                 colnames_responses, colnames_drivers, output_folder='DATA/AnalysisResults/'):
    """
    Exports the analysis results to CSV files.

    Args:
        df_filtered (pd.DataFrame): The original DataFrame with cluster indices.
        df_cluster_means (pd.DataFrame): DataFrame of cluster means.
        df_flow_prob (pd.DataFrame): Flow conditional probabilities.
        colnames_responses (list): List of response column names.
        colnames_drivers (list): List of driver column names.
        output_folder (str): Folder to save the CSV files. Defaults to 'DATA/AnalysisResults/'.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Save df_filtered (main analysis results)
    df_filtered.to_csv(os.path.join(output_folder, 'river_orgeval_filtered_data.csv'), index=True)

    # Save cluster means
    df_cluster_means.to_csv(os.path.join(output_folder, 'river_orgeval_cluster_means.csv'), index=True)

    # Save flow probabilities
    df_flow_prob.to_csv(os.path.join(output_folder, 'river_orgeval_flow_prob.csv'), index=False)

    # Save column names
    with open(os.path.join(output_folder, 'river_orgeval_colnames.txt'), 'w') as f:
        f.write("colnames_responses=" + str(colnames_responses) + "\n")
        f.write("colnames_drivers=" + str(colnames_drivers) + "\n")
    print(f"Analysis results saved to {output_folder}")


if __name__ == '__main__':
    nc_value = 6 # Example number of clusters
    df_filtered, df_cluster_means, df_flow_prob, colnames_responses, colnames_drivers = \
        run_river_orgeval_analysis(nc=nc_value)

    export_river_orgeval_results(df_filtered, df_cluster_means, df_flow_prob,
                                 colnames_responses, colnames_drivers)

    print("Analysis Complete. Results exported to CSV files for visualization.")