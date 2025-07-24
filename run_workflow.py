# import prep/pre-processing scripts as modules
from _01_DataPrep.scripts import prepare_hourly_flux_data_both_sites, prepare_flux_gc_daily_data, prepare_river_orgeval_data, prepare_river_plynlimon_data, prepare_river_monticello_data, prepare_root_soil_nebraska_data
# import analysis/plotting scripts as modules
from _02_Analysis.scripts import run_flux_both_sites_hourly_analysis, run_flux_gc_daily_analysis, run_river_monticello_analysis, run_river_orgeval_analysis, run_river_plynlimon_analysis, run_root_soil_nebraska_analysis

def prepare_data(in_folder, out_folder, res):
    """
    Prepare flux tower, riverlab and root soil datasets using pre-processing routines.
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

def analyze_data(in_folder, out_folder):
    """
    Analyze flux tower, riverlab and root soil datasets using clustering from gaussian mixture models, PCA analysis, and IT methods.
    """

    # add commnets here  
    try:
        run_flux_both_sites_hourly_analysis(in_folder,out_folder)
    except FileNotFoundError as e:
        print(e)
    try:
        run_flux_gc_daily_analysis(in_folder,out_folder)
    except FileNotFoundError as e:
        print(e)    
    try:
        run_river_monticello_analysis(in_folder,out_folder)
    except FileNotFoundError as e:
        print(e)    
    try:
        run_river_orgeval_analysis(in_folder,out_folder)
    except FileNotFoundError as e:
        print(e)
    try:
        run_river_plynlimon_analysis(in_folder,out_folder)
    except FileNotFoundError as e:
        print(e)
    try:
        run_root_soil_nebraska_analysis(in_folder,out_folder)
    except FileNotFoundError as e:
        print(e)  

def main():
    """
    Main function.
    """

    # prepare datasets using preprocessing
    prepare_data(in_folder="_01_DataPrep/input/", 
                 out_folder="_01_DataPrep/output/", 
                 res = '30min')
    
    # analyze datasets
    analyze_data(in_folder="_01_DataPrep/output/", 
                 out_folder="_02_Analysis/output/")

if __name__ == '__main__':
    main()