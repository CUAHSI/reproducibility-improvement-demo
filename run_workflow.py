# import prep/pre-processing scripts as modules
from _01_DataPrep.scripts import prepare_hourly_flux_data_both_sites, prepare_flux_gc_daily_data, prepare_river_orgeval_data, prepare_river_plynlimon_data, prepare_river_monticello_data, prepare_root_soil_nebraska_data
# import analysis/plotting scripts as modules

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

def prepare_data(in_folder, out_folder, res):
    """
    Prepare flux tower, riverlab and root soil datasets using pre-processing routines.
    """

def main():
    """
    Main function.
    """
    prepare_data(in_folder="_01_DataPrep/input/", 
                 out_folder="_01_DataPrep/output/", 
                 res = '30min')

if __name__ == '__main__':
    main()