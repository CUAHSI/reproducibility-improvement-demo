import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import numpy as np

def load_data(data_folder, datafile_name, date_column, index_column=None):
    """
    Loads data from a CSV file, converts a specified column to datetime objects,
    and sets it as the DataFrame index.

    Args:
        data_folder (str): The path to the folder containing the data file.
        datafile_name (str): The name of the CSV data file.
        date_column (str): The name of the column to be converted to datetime
                           and set as the index.
        index_column (str, optional): An alternative column to set as the index
                                      if different from date_column after conversion.
                                      Defaults to None, meaning date_column is used.

    Returns:
        pd.DataFrame: The loaded and preprocessed DataFrame.
    """
    df = pd.read_csv(f"{data_folder}{datafile_name}")
    df[date_column] = pd.to_datetime(df[date_column])
    if index_column:
        df = df.set_index(df[index_column])
    else:
        df = df.set_index(df[date_column])
    return df

def scale_data(df, columns, scaler_type='MinMaxScaler'):
    """
    Scales specified columns of a DataFrame using different scaling methods.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to be scaled.
        scaler_type (str): The type of scaler to use ('MinMaxScaler',
                           'StandardScaler', 'RobustScaler').
                           Defaults to 'MinMaxScaler'.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns scaled.
        object: The fitted scaler object.
    """
    scaled_df = df.copy()
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

def calculate_conditional_probabilities(df, cluster_col, time_col, time_units):
    """
    Calculates the conditional probabilities of cluster frequency by a given time unit.

    Args:
        df (pd.DataFrame): The input DataFrame with cluster indices and time information.
        cluster_col (str): The name of the column containing cluster indices.
        time_col (str): The name of the column containing the time unit (e.g., 'month', 'hour', 'year').
        time_units (list): A list of unique time units to iterate through (e.g., [1,2,..,12] for months).

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

def calculate_flow_conditional_probabilities(df, cluster_col, flow_col, flow_ranges):
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