import os 
# import prep helper script
import _01_DataPrep.scripts.DataPrep_Helpers as prep_hlp 
# import visualize helper script
import _03_visualize.scripts.Visualize_Helpers as viz_hlp

def visualize_clustering(input_file: str,
                         output_folder: str,
                         site_name: str):
    """
    Performs clustering analysis and generates visualizations based on the helper script.

    Args:
        input_file (str): Path to the results data.
        output_folder (str): Path to the output folder for figures.
        site_name (str): Name of site.
    """

    results = prep_hlp.load_csv(input_file)

    try:
        # the handoff between these functions is not correct at the momenet and would need to be modified
        # passing in less arguements than needed for now to throw a TypeError
        viz_hlp.plot_scaled_and_original_data(results, os.join(output_folder, site_name + '_scaled_and_original.svg'))
        viz_hlp.plot_gmm_aic_bic(results, os.join(output_folder, site_name + '_gmm_aic_bic.svg'))
        viz_hlp.plot_all_scatter_plots(results, os.join(output_folder, site_name + '_all_scatter_plots.svg'))
        viz_hlp.plot_mini_scatter_plots(results, os.join(output_folder, site_name + '_mini_scatter_plots.svg'))

    except TypeError:
        pass