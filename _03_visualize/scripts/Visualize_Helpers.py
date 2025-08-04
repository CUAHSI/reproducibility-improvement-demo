import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns

def setup_plot_style():
    """Sets up global font sizes and seaborn colormap."""
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    sns_list = sns.color_palette('deep').as_hex()
    sns_list.insert(0, '#ffffff')
    return ListedColormap(sns_list)


def plot_scaled_and_original_data(X_responses_scaled: np.array, 
                                  allvars_responses: list, 
                                  labels_responses: list,
                                  X_drivers_scaled: np.array, 
                                  allvars_drivers: list, 
                                  labels_drivers: list,
                                  figname: str):
    """
    Plots the scaled and original time series for response and driver variables.
    
    Args:
        X_responses_scaled (np.array): Scaled response variables.
        allvars_responses (list): List of original response variable Series.
        labels_responses (list): Labels for response variables.
        X_drivers_scaled (np.array): Scaled driver variables.
        allvars_drivers (list): List of original driver variable Series.
        labels_drivers (list): Labels for driver variables.
        figname (str): Base filename for saving the figure.
    """
    ct = 1
    plt.figure(figsize=(6, 12))
    
    for i, a in enumerate(X_responses_scaled):
        plt.subplot(20, 2, ct)
        plt.plot(a)
        plt.ylabel(labels_responses[i])
        plt.xticks([])
        if ct == 1:
            plt.title('Scaled')
        ct += 1
        plt.subplot(20, 2, ct)
        plt.plot(allvars_responses[i])
        plt.xticks([])
        if ct == 2:
            plt.title('Original')
        ct += 1

    for i, a in enumerate(X_drivers_scaled):
        plt.subplot(20, 2, ct)
        plt.plot(a)
        plt.ylabel(labels_drivers[i])
        plt.xticks([])
        ct += 1
        plt.subplot(20, 2, ct)
        plt.plot(allvars_drivers[i])
        plt.xticks([])
        ct += 1
    
    plt.tight_layout()
    plt.savefig(figname + '_ScaledOriginalData.svg')
    plt.show()


def plot_gmm_aic_bic(X_responses_scaled: np.array,
                     seed: int,
                     nc: int,
                     figname: str,
                     AIC: np.array,
                     BIC: np.array):
    """
    Plots the AIC and BIC values to help select the optimal number of clusters.

    Args:
        X_responses_scaled (np.array): Scaled response variables.
        seed (int): Random seed for reproducibility.
        nc (int): The selected number of clusters.
        figname (str): Base filename for saving the figure.
        AIC (np.array): AIC value for selecting optimal number of clusters.
        BIC (np.array): BIC value for selecting optimal number of clusters.
    """
    nc_range = range(2, 12)

    fig = plt.figure(figsize=(2.2, 2))
    plt.plot(nc_range, AIC, 'r')
    plt.plot(nc_range, BIC, 'b')
    plt.vlines(nc, ymin=np.min(AIC), ymax=np.max(AIC), color='k', linestyle=':')
    plt.legend(['AIC', 'BIC'])
    plt.xlabel('number of clusters')
    plt.xticks(nc_range)
    plt.tight_layout()
    fig.savefig(figname + 'GMM_AICBIC.svg')
    plt.show()


def plot_all_scatter_plots(features: np.array, 
                           balance_idx: list, 
                           labels_all: list, 
                           cm: ListedColormap, 
                           figname: str):
    """
    Creates a matrix of scatter plots for all variables, colored by cluster.

    Args:
        features (np.array): All data features.
        balance_idx (list): Cluster index for each data point.
        labels_all (list): Labels for all variables.
        cm (ListedColormap): Colormap for clusters.
        figname (str): Base filename for saving the figure.
    """
    nvars_all = np.shape(features)[1]
    ct = 1
    plt.figure(figsize=(10, 10))
    for i in range(nvars_all):
        for j in range(nvars_all):
            if j >= i:
                ct += 1
                continue
            else:
                plt.subplot(nvars_all, nvars_all, ct)
                plt.scatter(features[:, j], features[:, i], .5, balance_idx, cmap=cm, rasterized=True)
                if i == nvars_all - 1:
                    plt.xlabel(labels_all[j], fontsize=10, fontname='Arial')
                else:
                    plt.xlabel('')
                if j == 0:
                    plt.ylabel(labels_all[i], fontsize=10, fontname='Arial')
                else:
                    plt.ylabel('')
                plt.clim([-.5, cm.N - 0.5])
                plt.xticks([])
                plt.yticks([])
                plt.grid()
                plt.gca().tick_params(labelsize=10)
                plt.gca().tick_params(labelsize=10)
                ct += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(figname + '_ScatterPlotAllVars.svg', dpi=300)
    plt.show()


def plot_mini_scatter_plots(features: np.array, 
                            feat_inds: list, 
                            balance_idx: list, 
                            labels_all: list, 
                            cm: ListedColormap, 
                            figname: str):
    """
    Generates a smaller matrix of scatter plots for a selected subset of variables.

    Args:
        features (np.array): All data features.
        feat_inds (list): Indices of features to plot.
        balance_idx (list): Cluster index for each data point.
        labels_all (list): Labels for all variables.
        cm (ListedColormap): Colormap for clusters.
        figname (str): Base filename for saving the figure.
    """
    plt.figure(figsize=(3, 3))
    feats_small = features[:, feat_inds]
    labs = [l for i, l in enumerate(labels_all) if i in feat_inds]
    nvars = len(feat_inds)

    ct = 1
    for i in range(nvars):
        for j in range(nvars):
            if j >= i:
                ct += 1
                continue
            else:
                plt.subplot(nvars, nvars, ct)
                plt.scatter(feats_small[:, j], feats_small[:, i], .5, balance_idx, cmap=cm, rasterized=True)
                if i == nvars - 1:
                    plt.xlabel(labs[j], fontsize=10, fontname='Arial')
                else:
                    plt.xlabel('')
                if j == 0:
                    plt.ylabel(labs[i], fontsize=10, fontname='Arial')
                else:
                    plt.ylabel('')
                plt.clim([-.5, cm.N - 0.5])
                plt.xticks([])
                plt.yticks([])
                plt.grid()
                plt.gca().tick_params(labelsize=10)
                plt.gca().tick_params(labelsize=10)
                ct += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(figname + '_MiniScatterPlots.svg', dpi=300)
    plt.show()