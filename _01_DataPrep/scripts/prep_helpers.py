import pandas as pd
import numpy as np
import datetime as dt

def load_and_process_csv(
    file_path: str,
    date_column: str,
    index_column: str = None,
    localize_tz: bool = True,
    use_cols: list = None
) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame, converts a specified column to datetime,
    localizes timezone (optional), and sets an index.

    Args:
        file_path (str): The path to the CSV file.
        date_column (str): The name of the column to convert to datetime.
        index_column (str, optional): The column to set as the DataFrame index.
                                      Defaults to None (keeps existing index or creates default).
        localize_tz (bool, optional): Whether to localize the datetime column to None timezone.
                                      Defaults to True.
        use_cols (list, optional): A list of column names to load from the CSV.
                                   Defaults to None (loads all columns).

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    df = pd.read_csv(file_path, usecols=use_cols)
    df[date_column] = pd.to_datetime(df[date_column])
    if localize_tz:
        df[date_column] = df[date_column].dt.tz_localize(None)

    if index_column:
        df = df.set_index(index_column)
    return df

def resample_and_interpolate(
    df: pd.DataFrame,
    resample_freq: str,
    on_column: str = None,
    interpolation_method: str = 'linear',
    interpolation_limit: int = None
) -> pd.DataFrame:
    """
    Resamples a DataFrame to a specified frequency and interpolates missing values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        resample_freq (str): The frequency string for resampling (e.g., '1D', '30T', '1H').
        on_column (str, optional): The column to resample on if the DataFrame is not indexed by time.
                                   Defaults to None.
        interpolation_method (str, optional): The interpolation method (e.g., 'linear', 'time').
                                              Defaults to 'linear'.
        interpolation_limit (int, optional): Maximum number of consecutive NaNs to fill. Defaults to None.

    Returns:
        pd.DataFrame: The resampled and interpolated DataFrame.
    """
    if on_column:
        df_resampled = df.resample(resample_freq, on=on_column).mean() # Assuming mean for resampling aggregation
    else:
        df_resampled = df.resample(resample_freq).mean() # Assuming mean for resampling aggregation
    return df_resampled.interpolate(method=interpolation_method, limit=interpolation_limit)

def remove_outliers_by_quantile(
    df: pd.DataFrame,
    columns: list,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995
) -> pd.DataFrame:
    """
    Removes outliers from specified columns based on quantiles. Values outside the
    specified quantile range are replaced with NaN.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to process.
        lower_quantile (float, optional): The lower quantile threshold. Defaults to 0.005.
        upper_quantile (float, optional): The upper quantile threshold. Defaults to 0.995.

    Returns:
        pd.DataFrame: The DataFrame with outliers replaced by NaN.
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            maxval = df_copy[col].quantile(upper_quantile)
            minval = df_copy[col].quantile(lower_quantile)
            df_copy[col] = np.where(df_copy[col] > maxval, np.nan, df_copy[col])
            df_copy[col] = np.where(df_copy[col] < minval, np.nan, df_copy[col])
    return df_copy

def apply_range_filter(
    df: pd.DataFrame,
    column: str,
    lower_bound: float = None,
    upper_bound: float = None,
    replace_value=np.nan
) -> pd.DataFrame:
    """
    Applies a numerical range filter to a specified column, replacing values outside
    the bounds with a specified value (default is NaN).

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to filter.
        lower_bound (float, optional): The lower bound for filtering. Defaults to None (no lower bound).
        upper_bound (float, optional): The upper bound for filtering. Defaults to None (no upper bound).
        replace_value: The value to replace filtered data with. Defaults to np.nan.

    Returns:
        pd.DataFrame: The DataFrame with the filter applied.
    """
    df_copy = df.copy()
    if column in df_copy.columns:
        if lower_bound is not None:
            df_copy[column] = np.where(df_copy[column] < lower_bound, replace_value, df_copy[column])
        if upper_bound is not None:
            df_copy[column] = np.where(df_copy[column] > upper_bound, replace_value, df_copy[column])
    return df_copy

def filter_by_datetime_range(
    df: pd.DataFrame,
    start_date: dt.datetime = None,
    end_date: dt.datetime = None,
    start_month: int = None,
    end_month: int = None,
    start_hour: int = None,
    end_hour: int = None,
    exclude_year: int = None
) -> pd.DataFrame:
    """
    Filters a DataFrame by various datetime components using its datetime index.
    Can filter by a full date range, month range, hour range, or exclude a specific year.

    Args:
        df (pd.DataFrame): The input DataFrame with a datetime index.
        start_date (datetime.datetime, optional): The start date for filtering. Defaults to None.
        end_date (datetime.datetime, optional): The end date for filtering. Defaults to None.
        start_month (int, optional): The starting month (1-12) for filtering. Defaults to None.
        end_month (int, optional): The ending month (1-12) for filtering. Defaults to None.
        start_hour (int, optional): The starting hour (0-23) for filtering. Defaults to None.
        end_hour (int, optional): The ending hour (0-23) for filtering. Defaults to None.
        exclude_year (int, optional): A specific year to exclude from the DataFrame. Defaults to None.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    df_filtered = df.copy()

    if not isinstance(df_filtered.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index for datetime filtering.")

    if start_date is not None and end_date is not None:
        df_filtered = df_filtered.loc[(df_filtered.index >= start_date) & (df_filtered.index <= end_date)]

    if start_month is not None and end_month is not None:
        df_filtered = df_filtered.loc[(df_filtered.index.month >= start_month) & (df_filtered.index.month <= end_month)]

    if start_hour is not None and end_hour is not None:
        # Note: end_hour is exclusive in Python's range behavior, so use < end_hour
        df_filtered = df_filtered.loc[(df_filtered.index.hour >= start_hour) & (df_filtered.index.hour < end_hour)]

    if exclude_year is not None:
        df_filtered = df_filtered[df_filtered.index.year != exclude_year]

    return df_filtered

def calculate_rolling_sums(df: pd.DataFrame, columns: list, window_days: list) -> pd.DataFrame:
    """
    Calculates rolling sums for specified columns over given window days.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names for which to calculate rolling sums.
        window_days (list): A list of integers representing the number of days for the rolling window.

    Returns:
        pd.DataFrame: The DataFrame with new rolling sum columns.
    """
    df_copy = df.copy()
    for col in columns:
        for window in window_days:
            df_copy[f'{col}_{window}D'] = df_copy[col].rolling(window, min_periods=1).sum()
    return df_copy

def calculate_cumulative_sum_by_year(df: pd.DataFrame, column: str, new_column_name: str) -> pd.DataFrame:
    """
    Calculates the cumulative sum of a column, resetting at the beginning of each year.

    Args:
        df (pd.DataFrame): The input DataFrame with a datetime index.
        column (str): The name of the column to calculate the cumulative sum for.
        new_column_name (str): The name for the new cumulative sum column.

    Returns:
        pd.DataFrame: The DataFrame with the new cumulative sum column.
    """
    df_copy = df.copy()
    if column in df_copy.columns:
        df_copy[new_column_name] = df_copy.groupby(df_copy.index.year)[column].transform('cumsum')
    return df_copy

def calculate_doy(df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
    """
    Calculates the Day of Year (DOY) for a given datetime column or index.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_column (str, optional): The column to use for DOY calculation.
                                     If None, uses the DataFrame index. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with a 'DOY' column.
    """
    df_copy = df.copy()
    if date_column and date_column in df_copy.columns:
        df_copy['DOY'] = df_copy[date_column].dt.dayofyear
    elif isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy['DOY'] = df_copy.index.dayofyear
    else:
        raise ValueError("DataFrame must have a datetime index or a specified date_column.")
    return df_copy

def clean_data_by_masking_values(
    df: pd.DataFrame,
    columns: list,
    min_value: float = None,
    max_value: float = None,
    replace_value=np.nan
) -> pd.DataFrame:
    """
    Cleans data in specified columns by masking values that fall outside a given range.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to clean.
        min_value (float, optional): The minimum allowed value. Defaults to None.
        max_value (float, optional): The maximum allowed value. Defaults to None.
        replace_value: The value to replace masked data with. Defaults to np.nan.

    Returns:
        pd.DataFrame: The DataFrame with cleaned columns.
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            if min_value is not None:
                df_copy[col] = df_copy[col].mask(df_copy[col] < min_value, replace_value)
            if max_value is not None:
                df_copy[col] = df_copy[col].mask(df_copy[col] > max_value, replace_value)
    return df_copy