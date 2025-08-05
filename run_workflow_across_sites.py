# import functions from prep, analysis and visualization scripts (treated as subpackages)
from _01_prep.scripts import prepare_hourly_flux_data_both_sites, prepare_flux_gc_daily_data, prepare_river_monticello_data, prepare_river_orgeval_data, prepare_river_plynlimon_data, prepare_root_soil_nebraska_data
from _02_analyze.scripts import analyze_flux_both_sites_hourly, analyze_flux_gc_daily, analyze_river_monticello, analyze_river_orgeval, analyze_river_plynlimon, analyze_root_soil_nebraska
from _03_visualize.scripts import visualize_clustering

def prep_data(in_folder: str, 
              out_folder: str, 
              res: str):
    """
    Prepare flux tower, riverlab and root soil datasets using pre-processing routines.

    Args:
        in_folder  (str): The path to the folder containing the raw data CSV files.
        out_folder (str): The path to export the intermediate processed data.
        res        (str): Temporal resolution for processing data
    """

    # pre-process data for flux tower case - concatenate datasets, gap fill, etc    
    try:
        prepare_hourly_flux_data_both_sites(in_folder,out_folder)
    except FileNotFoundError as e:
        print(e)    
    try:    
        prepare_flux_gc_daily_data(in_folder,out_folder)
    except FileNotFoundError as e:
        print(e)     

    # pre-process data for riverlab case - concatenate datasets, gap fill, etc
    try:
        prepare_river_monticello_data(in_folder,out_folder, res)
    except FileNotFoundError as e:
        print(e)
    try:
        prepare_river_orgeval_data(in_folder,out_folder, res)
    except FileNotFoundError as e:
        print(e)
    try:
        prepare_river_plynlimon_data(in_folder,out_folder, res)
    except FileNotFoundError as e:
        print(e)

    # MIRZ data pre-processing for input into clustering and IT algorithm
    try:
        prepare_root_soil_nebraska_data(in_folder,out_folder)
    except FileNotFoundError as e:
        print(e)

def analyze_data(prepared_flux_both_sites_hourly_csv: str,
                 prepared_flux_gc_daily_csv: str,
                 prepared_river_monticello_csv: str,
                 prepared_river_orgeval_csv: str,
                 prepared_river_plynlimon_csv: str,
                 prepared_root_soil_nebraska_csv: str,
                 out_folder: str,
                 res: str):
    """
    Analyze flux tower, riverlab and root soil datasets using clustering from gaussian mixture models, PCA analysis, and IT methods.

    Args:
        prepared_flux_both_sites_hourly_csv  (str): Full path to csv file of prepared houly datasets for both flux sites and hourly.
        prepared_flux_gc_daily_csv           (str): Full path to csv file of prepared daily datasets for gc flux site.
        prepared_river_monticello_csv        (str): Full path to csv file of prepared datasets for river monticello site.
        prepared_river_orgeval_csv           (str): Full path to csv file of prepared datasets for river orgeval site.
        prepared_river_plynlimon_csv         (str): Full path to csv file of prepared datasets for river plynlimon site.
        prepared_root_soil_nebraska_csv      (str): Full path to csv file of prepared datasets for root soil nebraska site.
        out_folder                           (str): Output folder for results.
    """

    # Analyze flux tower data and apply clustering
    try:
        analyze_flux_both_sites_hourly(prepared_flux_both_sites_hourly_csv, out_folder)
    except FileNotFoundError as e:
        print(e)
    try:
        analyze_flux_gc_daily(prepared_flux_gc_daily_csv, out_folder)
    except FileNotFoundError as e:
        print(e)

    # Analyze river data and apply clustering
    try:
        analyze_river_monticello(prepared_river_monticello_csv, out_folder)
    except FileNotFoundError as e:
        print(e)    
    try:
        analyze_river_orgeval(prepared_river_orgeval_csv, out_folder)
    except FileNotFoundError as e:
        print(e)
    try:
        analyze_river_plynlimon(prepared_river_plynlimon_csv, out_folder)
    except FileNotFoundError as e:
        print(e)

    # Analyze root soil data and apply clustering
    try:
        analyze_root_soil_nebraska(prepared_root_soil_nebraska_csv, out_folder)
    except FileNotFoundError as e:
        print(e)

def visualize_data(analyzed_flux_both_sites_hourly_csv: str,
                   analyzed_flux_gc_daily_csv: str,
                   analyzed_river_monticello_csv: str,
                   analyzed_river_orgeval_csv: str,
                   analyzed_river_plynlimon_csv: str,
                   analyzed_root_soil_nebraska_csv: str,
                   out_folder: str):
    """
    Visualize clustering results of flux tower, riverlab and root soil datasets.

    Args:
        analyzed_flux_both_sites_hourly_csv  (str): Full path to csv file of analyzed houly datasets for both flux sites and hourly.
        analyzed_flux_gc_daily_csv           (str): Full path to csv file of analyzed daily datasets for gc flux site.
        analyzed_river_monticello_csv        (str): Full path to csv file of analyzed datasets for river monticello site.
        analyzed_river_orgeval_csv           (str): Full path to csv file of analyzed datasets for river orgeval site.
        analyzed_river_plynlimon_csv         (str): Full path to csv file of analyzed datasets for river plynlimon site.
        analyzed_root_soil_nebraska_csv      (str): Full path to csv file of analyzed datasets for root soil nebraska site.
        out_folder                           (str): Output folder for figures
    """
    
    # Visualize clustering of flux tower data
    try:
        visualize_clustering(analyzed_flux_both_sites_hourly_csv, out_folder)
    except FileNotFoundError as e:
        print(e)
    try:
        visualize_clustering(analyzed_flux_gc_daily_csv, out_folder)
    except FileNotFoundError as e:
        print(e)

    # Visualize clustering of river data
    try:
        visualize_clustering(analyzed_river_monticello_csv, out_folder)
    except FileNotFoundError as e:
        print(e)    
    try:
        visualize_clustering(analyzed_river_orgeval_csv, out_folder)
    except FileNotFoundError as e:
        print(e)
    try:
        visualize_clustering(analyzed_river_plynlimon_csv, out_folder)
    except FileNotFoundError as e:
        print(e)

    # Visualize clustering of root soil data
    try:
        visualize_clustering(analyzed_root_soil_nebraska_csv, out_folder)
    except FileNotFoundError as e:
        print(e)


def main():
    """
    Main function.
    """

    # Commenting this out for now, but the main function should execute each phase of workflow, e.g.,
    # prep_data(**args)
    # analyze_data(**args)
    # visualize_data(**args)

# Everything after this if statement is ran when the python file is run directly
if __name__ == '__main__': 
    main()
