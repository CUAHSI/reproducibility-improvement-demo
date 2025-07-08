from _01_DataPrep.scripts import DataPrep_FluxBothSitesHourly, DataPrep_FluxGCDaily, DataPrep_RiverMonticello, DataPrep_RiverOrgeval, DataPrep_RiverPlynlimon, DataPrep_RootSoilNebraska

def prepare_data(in_folder, out_folder, res):

    # pre-process data for flux tower case - concatenate datasets, gap fill, etc    
    try:
        DataPrep_FluxBothSitesHourly.main(in_folder,out_folder)
    except FileNotFoundError as e:
        print(e)    
    try:    
        DataPrep_FluxGCDaily.main(in_folder,out_folder)
    except FileNotFoundError as e:
        print(e)     

    # pre-process data for riverlab case - concatenate datasets, gap fill, etc
    try:
        DataPrep_RiverMonticello.main(in_folder,out_folder, res)
    except FileNotFoundError as e:
        print(e)
    try:
        DataPrep_RiverOrgeval.main(in_folder,out_folder, res)
    except FileNotFoundError as e:
        print(e)
    try:
        DataPrep_RiverPlynlimon.main(in_folder,out_folder, res)
    except FileNotFoundError as e:
        print(e)

    # MIRZ data pre-processing for input into clustering and IT algorithm
    try:
        DataPrep_RootSoilNebraska.main(in_folder,out_folder)
    except FileNotFoundError as e:
        print(e)

def main():
    prepare_data(in_folder="_01_DataPrep/input/", 
                 out_folder="_01_DataPrep/output/", 
                 res = '30min')

if __name__ == '__main__':
    main()