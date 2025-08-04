import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Any, Dict
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator # Import BaseEstimator for scaler type hinting

# import prep helper script
import _01_DataPrep.scripts.DataPrep_Helpers as prep_hlp 
# import analysis clustering script
import _02_analyze.scripts.cluster_funcs as cf

def scale_data(df: pd.DataFrame,
               columns: List[str],
               scaler_type: str = 'MinMaxScaler') -> Tuple[pd.DataFrame, BaseEstimator]:
    """
    Scales specified columns of a DataFrame using different scaling methods.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (List[str]): A list of column names to be scaled.
        scaler_type (str): The type of scaler to use ('MinMaxScaler',
                           'StandardScaler', 'RobustScaler').
                           Defaults to 'MinMaxScaler'.

    Returns:
        Tuple[pd.DataFrame, BaseEstimator]: A new DataFrame with the specified columns scaled,
                                           and the fitted scaler object.
    """
    scaled_df = df.copy()
    scaler: BaseEstimator # Declare scaler with a type hint
    if scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'RobustScaler':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaler_type. Choose from 'MinMaxScaler', 'StandardScaler', 'RobustScaler'.")

    scaled_df[columns] = scaler.fit_transform(scaled_df[columns])
    return scaled_df, scaler

def calculate_conditional_probabilities(df: pd.DataFrame,
                                        cluster_col: str,
                                        time_col: str,
                                        time_units: List[Union[int, str]]) -> pd.DataFrame:
    """
    Calculates the conditional probabilities of cluster frequency by a given time unit.

    Args:
        df (pd.DataFrame): The input DataFrame with cluster indices and time information.
        cluster_col (str): The name of the column containing cluster indices.
        time_col (str): The name of the column containing the time unit (e.g., 'month', 'hour', 'year').
        time_units (List[Union[int, str]]): A list of unique time units to iterate through (e.g., [1,2,..,12] for months).

    Returns:
        pd.DataFrame: A DataFrame containing the conditional probabilities.
    """
    nc = df[cluster_col].nunique()
    frequencies = np.zeros((nc, len(time_units)))

    for i in range(1, nc + 1):  # loop through classes
        for m_ind, m in enumerate(time_units):  # loop through time units
            df_small = df.loc[(df[cluster_col] == i) & (df[time_col] == m)]
            frequencies[i - 1, m_ind] = len(df_small)

    p_frequencies = frequencies / np.sum(frequencies)
    p_time = np.sum(p_frequencies, axis=0)
    # Handle division by zero for time units with no data
    conditional_prob = np.divide(p_frequencies, p_time, out=np.zeros_like(p_frequencies), where=p_time!=0)

    dfp = pd.DataFrame(data=conditional_prob.T, columns=[str(j) for j in range(1, nc + 1)])
    dfp[time_col.capitalize()] = [str(unit) for unit in time_units]
    return dfp

def calculate_flow_conditional_probabilities(df: pd.DataFrame,
                                             cluster_col: str,
                                             flow_col: str,
                                             flow_ranges: np.ndarray) -> pd.DataFrame:
    """
    Calculates the conditional probabilities of cluster frequency based on flow ranges.

    Args:
        df (pd.DataFrame): The input DataFrame with cluster indices and flow data.
        cluster_col (str): The name of the column containing cluster indices.
        flow_col (str): The name of the column containing flow values.
        flow_ranges (np.ndarray): An array of flow quantile ranges.

    Returns:
        pd.DataFrame: A DataFrame containing the conditional probabilities.
    """
    nc = df[cluster_col].nunique()
    flows = np.zeros((nc, len(flow_ranges) - 1))

    for i in range(1, nc + 1):  # loop through classes
        for m_ind, m in enumerate(flow_ranges[:-1]):  # loop through flow quantiles
            df_small = df.loc[(df[cluster_col] == i) & (df[flow_col] > m) & (df[flow_col] <= flow_ranges[m_ind + 1])]
            flows[i - 1, m_ind] = len(df_small)

    p_flows = flows / np.sum(flows)
    p_f = np.sum(p_flows, axis=0)
    # Handle division by zero for flow ranges with no data
    conditional_prob = np.divide(p_flows, p_f, out=np.zeros_like(p_flows), where=p_f!=0)

    dfp = pd.DataFrame(data=conditional_prob.T, columns=[str(j) for j in range(1, nc + 1)])
    dfp['Flow Quantile'] = [f"{idx+1}" for idx in range(len(flow_ranges) - 1)]
    return dfp

def export_analysis_results(site_name: str,
                            df_filtered: pd.DataFrame,
                            df_cluster_means: pd.DataFrame,
                            colnames_responses: List[str],
                            colnames_drivers: List[str],
                            output_folder: str,
                            prob_dataframes: Dict[str, pd.DataFrame]):
    """
    Exports the analysis results to CSV files.

    Args:
        site_name (str): The name of the site being analyzed.
        df_filtered (pd.DataFrame): The original DataFrame with cluster indices.
        df_cluster_means (pd.DataFrame): DataFrame of cluster means.
        colnames_responses (List[str]): List of response column names.
        colnames_drivers (List[str]): List of driver column names.
        output_folder (str): Folder to save the CSV files.
        prob_dataframes (Dict[str, pd.DataFrame]): A dictionary of conditional probability
                                                   DataFrames to export. The keys will be used
                                                   to name the files.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Save df_filtered (main analysis results)
    df_filtered.to_csv(os.path.join(output_folder, site_name + '_filtered_data.csv'), index=True)

    # Save cluster means
    df_cluster_means.to_csv(os.path.join(output_folder, site_name + '_cluster_means.csv'), index=True)
    
    # Save conditional probabilities
    for name, df in prob_dataframes.items():
        df.to_csv(os.path.join(output_folder, f'{site_name}_{name}.csv'), index=False)

    # Save column names
    with open(os.path.join(output_folder, site_name + '_colnames.txt'), 'w') as f:
        f.write("colnames_responses=" + str(colnames_responses) + "\n")
        f.write("colnames_drivers=" + str(colnames_drivers) + "\n")
    print(f"Analysis results saved to {output_folder}")

def analyze_site_data(site_name: str,
                      site_type: str,
                      input_file: str,
                      date_column: str,
                      colnames_responses: List[str],
                      colnames_drivers: List[str],
                      scaler_type: str,
                      nc: int,
                      seed: int):
    """
    Performs the core analysis steps: loading, scaling, clustering, and
    calculating cluster means.

    Args:
        site_name (str): The name of the site being analyzed.
        site_name (str): Type of site being analyzed (e.g., river)
        input_file (str): Path to the folder containing the data file.
        date_column (str): The name of the date/time column in the data file.
        colnames_responses (List[str]): List of response column names.
        colnames_drivers (List[str]): List of driver column names.
        scaler_type (str): The type of scaler to use.
        nc (int): Number of clusters for GMM.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
               - df_filtered (pd.DataFrame): The original DataFrame filtered for NaNs,
                                             with 'balance_idx' added.
               - df_cluster_means (pd.DataFrame): DataFrame of cluster means for response variables.
               - colnames_responses (list): List of response column names used in analysis.
               - colnames_drivers (list): List of driver column names used in analysis.
    """

    # Add logic in this function for site type specific analysis
    
    # Load data
    df = prep_hlp.load_csv(input_file)
    
    # Set the date column as index
    if date_column in df.columns:
        df.set_index(pd.to_datetime(df[date_column]), inplace=True)
        df.drop(columns=[date_column], inplace=True)
    
    # Define columns for analysis
    all_cols = colnames_responses + colnames_drivers

    # Filter out rows with NaN values in relevant columns for clustering
    df_filtered = df.dropna(subset=all_cols).copy() # Use .copy() to avoid SettingWithCopyWarning

    # Scale data
    df_scaled, scaler = scale_data(df_filtered, all_cols, scaler_type)

    # Prepare data for GMM
    allvars_responses = [df_scaled[col].values for col in colnames_responses]

    # Run GMM clustering
    gmm_model, balance_idx = cf.GMMfun(allvars_responses, nc, seed)
    df_filtered['balance_idx'] = balance_idx # Add cluster index to original filtered data

    # Calculate cluster means for original (unscaled) response variables
    cluster_means = df_filtered.groupby('balance_idx')[colnames_responses].mean()
    df_cluster_means = cluster_means.T
    df_cluster_means.columns = [f'Cluster {i}' for i in range(1, nc + 1)]

    return df_filtered, df_cluster_means, colnames_responses, colnames_drivers
