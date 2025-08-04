# import analysis helper script
import _02_analyze.scripts.Analysis_Helpers as analysis_hlp

def analyze_flux_gc_daily(input_file: str,
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
    colnames_responses = ['GPP_DT', 'Tr_Wm2', 'NDVI']
    colnames_drivers = ['Rn_Wm2', 'Tair_C', 'VPD_kPa', 'SWC_5cm', 'SWC_10cm', 'SWC_20cm',
                        'SWC_50cm', 'SWC_100cm', 'Precip_mm']

    # Define scaler
    scaler_type = 'MinMaxScaler'

    # Define date column
    date_column='Date'

    # Analyze and export site data
    df_filtered, df_cluster_means, df_hourly_prob, df_monthly_prob, df_yearly_prob, colnames_responses, colnames_drivers = analysis_hlp.analyze_site_data(input_file, 'flux', output_folder, date_column, colnames_responses, colnames_drivers, scaler_type, nc, seed)

    # export site data
    analysis_hlp.export_analysis_results('flux_gc_daily', df_filtered, df_cluster_means, df_hourly_prob, df_monthly_prob, df_yearly_prob, colnames_responses, colnames_drivers, output_folder)