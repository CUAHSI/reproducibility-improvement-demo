#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions for: gaussian mixture models, PCA analysis, and IT methods
CINet clustering interfaces project

@author: Allison Goodwell
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import SparsePCA, PCA
import seaborn as sns
from scipy.stats import gaussian_kde

np.seterr(all='ignore')

def GMMpick(allvars_responses, seed, nc_range):
    """
    Calculates AIC and BIC measures for a range of cluster numbers using Gaussian Mixture Models.

    Args:
        allvars_responses (list): A list of numpy arrays, where each array represents
                                  a time series variable. These will be stacked
                                  to form the features for GMM.
        seed (int): Random seed for reproducibility.
        nc_range (range): A range object specifying the number of clusters to test
                          (e.g., range(1, 10)).

    Returns:
        tuple: A tuple containing two lists: (AIC values, BIC values).
    """
    # Stack the response variables and transpose to get features as columns
    features = 1e3 * np.vstack(allvars_responses).T
    AIC = []
    BIC = []
    for nc in nc_range:
        gmm_model = GaussianMixture(n_components=nc, random_state=seed)
        gmm_model.fit(features)
        AIC.append(gmm_model.aic(features))
        BIC.append(gmm_model.bic(features))
    return AIC, BIC

def GMMfun(allvars_responses, nc, seed, plot_cov=0, labels=0):
    """
    Applies Gaussian Mixture Model (GMM) clustering to the input data.

    Args:
        allvars_responses (list): A list of numpy arrays, where each array represents
                                  a time series variable. These will be stacked
                                  to form the features for GMM.
        nc (int): The number of clusters to form.
        seed (int): Random seed for reproducibility.
        plot_cov (int, optional): If 1, plots the covariance matrix. Defaults to 0.
        labels (int, optional): Placeholder for future use. Defaults to 0.

    Returns:
        tuple: A tuple containing:
               - gmm_model (sklearn.mixture.GaussianMixture): The fitted GMM model.
               - balance_idx (np.ndarray): Array of predicted cluster indices for each data point.
    """
    # Stack the response variables and transpose to get features as columns
    features = 1e3 * np.vstack(allvars_responses).T
    gmm_model = GaussianMixture(n_components=nc, random_state=seed)
    gmm_model.fit(features)
    balance_idx = gmm_model.predict(features) + 1 # Add 1 to make cluster indices 1-based

    if plot_cov == 1:
        # Plotting covariance matrix (example, can be expanded)
        plt.figure()
        plt.imshow(gmm_model.covariances_[0], cmap='viridis') # Plotting first covariance matrix
        plt.title('Covariance Matrix of First Component')
        plt.colorbar()
        plt.show()

    return gmm_model, balance_idx

def PCAfun(allvars_responses, nc, seed, plot_pca=0, labels=0):
    """
    Applies Principal Component Analysis (PCA) to the input data.

    Args:
        allvars_responses (list): A list of numpy arrays, where each array represents
                                  a time series variable. These will be stacked
                                  to form the features for PCA.
        nc (int): The number of principal components to compute.
        seed (int, optional): Random seed (not directly used by PCA, but kept for consistency).
        plot_pca (int, optional): If 1, plots the explained variance ratio. Defaults to 0.
        labels (int, optional): Placeholder for future use. Defaults to 0.

    Returns:
        tuple: A tuple containing:
               - pca_model (sklearn.decomposition.PCA): The fitted PCA model.
               - pca_components (np.ndarray): The principal components.
               - transformed_features (np.ndarray): The data transformed into the PCA space.
    """
    features = np.vstack(allvars_responses).T
    pca_model = PCA(n_components=nc, random_state=seed)
    transformed_features = pca_model.fit_transform(features)
    pca_components = pca_model.components_

    if plot_pca == 1:
        plt.figure(figsize=(8, 4))
        plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance Ratio')
        plt.grid(True)
        plt.show()

    return pca_model, pca_components, transformed_features

def shannon_entropy(data_list, nbins, num_vars):
    """
    Calculates Shannon entropy for given data.

    Args:
        data_list (list): A list of numpy arrays, each representing a variable.
        nbins (int): Number of bins for histogram creation (for density estimation).
        num_vars (int): Number of variables in the data_list.

    Returns:
        tuple: A tuple containing various entropy measures (H_3, H_x1x2, H_x1y, H_x2y, H_s1, H_s2, H_tar).
               The specific meaning depends on the context of information_partitioning.
    """
    # This function is typically a helper for information_partitioning
    # and its outputs are specific to that context.
    # The current implementation directly calculates joint and marginal entropies.

    # For simplicity, assuming num_vars is 3 for (x1, x2, y) as in the original code
    # This function needs to be robust for different num_vars if used generally.

    # Example for 3 variables:
    if num_vars == 3:
        x1, x2, y = data_list[0], data_list[1], data_list[2]

        # Joint histogram for (x1, x2, y)
        hist3d, _ = np.histogramdd((x1, x2, y), bins=nbins)
        p_xyz = hist3d / hist3d.sum()
        H_3 = -np.sum(p_xyz * np.log2(p_xyz + np.finfo(float).eps)) # Joint entropy H(X1,X2,Y)

        # Joint histogram for (x1, x2)
        hist_x1x2, _ = np.histogramdd((x1, x2), bins=nbins)
        p_x1x2 = hist_x1x2 / hist_x1x2.sum()
        H_x1x2 = -np.sum(p_x1x2 * np.log2(p_x1x2 + np.finfo(float).eps)) # Joint entropy H(X1,X2)

        # Joint histogram for (x1, y)
        hist_x1y, _ = np.histogramdd((x1, y), bins=nbins)
        p_x1y = hist_x1y / hist_x1y.sum()
        H_x1y = -np.sum(p_x1y * np.log2(p_x1y + np.finfo(float).eps)) # Joint entropy H(X1,Y)

        # Joint histogram for (x2, y)
        hist_x2y, _ = np.histogramdd((x2, y), bins=nbins)
        p_x2y = hist_x2y / hist_x2y.sum()
        H_x2y = -np.sum(p_x2y * np.log2(p_x2y + np.finfo(float).eps)) # Joint entropy H(X2,Y)

        # Marginal entropies
        hist_s1, _ = np.histogram(x1, bins=nbins)
        p_s1 = hist_s1 / hist_s1.sum()
        H_s1 = -np.sum(p_s1 * np.log2(p_s1 + np.finfo(float).eps)) # H(X1)

        hist_s2, _ = np.histogram(x2, bins=nbins)
        p_s2 = hist_s2 / hist_s2.sum()
        H_s2 = -np.sum(p_s2 * np.log2(p_s2 + np.finfo(float).eps)) # H(X2)

        hist_tar, _ = np.histogram(y, bins=nbins)
        p_tar = hist_tar / hist_tar.sum()
        H_tar = -np.sum(p_tar * np.log2(p_tar + np.finfo(float).eps)) # H(Y)

        return H_3, H_x1x2, H_x1y, H_x2y, H_s1, H_s2, H_tar
    else:
        raise ValueError("shannon_entropy currently supports only 3 variables for information partitioning.")


def information_partitioning(dfinit, source_1, source_2, target, nbins, reshuffle=0):
    """
    Performs information partitioning analysis to decompose mutual information.

    Args:
        dfinit (pd.DataFrame): The input DataFrame.
        source_1 (str): Name of the first source variable column.
        source_2 (str): Name of the second source variable column.
        target (str): Name of the target variable column.
        nbins (int): Number of bins for entropy calculation.
        reshuffle (int, optional): If 1, reshuffles source variables to calculate
                                   redundancy and synergy for shuffled data. Defaults to 0.

    Returns:
        tuple: A tuple containing:
               - mi_s1_tar (float): Mutual information between source_1 and target.
               - mi_s2_tar (float): Mutual information between source_2 and target.
               - redundancy (float): Redundancy measure.
               - synergy (float): Synergy measure.
               - unique_s1 (float): Unique information from source_1.
               - unique_s2 (float): Unique information from source_2.
               - total_mi (float): Total mutual information.
    """
    df = dfinit.copy()

    if reshuffle == 1:
        df[source_1] = np.random.permutation(df[source_1].values)
        df[source_2] = np.random.permutation(df[source_2].values)
    else:
        df[source_1] = df[source_1].values
        df[source_2] = df[source_2].values
    df[target] = df[target].values

    x1 = df[source_1].values
    x2 = df[source_2].values
    y = df[target].values

    H_3, H_x1x2, H_x1y, H_x2y, H_s1, H_s2, H_tar = shannon_entropy([x1, x2, y], nbins, 3)

    mi_s1_s2 = H_s1 + H_s2 - H_x1x2  # I(X1;X2)
    mi_s1_tar = H_s1 + H_tar - H_x1y # I(X1;Y)
    mi_s2_tar = H_s2 + H_tar - H_x2y # I(X2;Y)
    mi_s1s2_tar = H_s1 + H_s2 + H_tar - H_x1x2 - H_x1y - H_x2y + H_3 # I(X1,X2;Y)

    # Calculate redundancy and synergy based on Williams and Beer decomposition
    redundancy = mi_s1_tar + mi_s2_tar - mi_s1s2_tar
    synergy = mi_s1s2_tar - mi_s1_tar - mi_s2_tar + redundancy # This is the interaction information I(X1;X2;Y)

    # Unique information
    unique_s1 = mi_s1_tar - redundancy
    unique_s2 = mi_s2_tar - redundancy
    total_mi = mi_s1s2_tar

    return mi_s1_tar, mi_s2_tar, redundancy, synergy, unique_s1, unique_s2, total_mi

def calculate_mmi(df, source_col, target_col, nbins):
    """
    Calculates the Mutual Information (MI) between a source and a target variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        source_col (str): Name of the source variable column.
        target_col (str): Name of the target variable column.
        nbins (int): Number of bins for entropy calculation.

    Returns:
        float: The mutual information value.
    """
    x = df[source_col].values
    y = df[target_col].values

    # Calculate joint histogram
    hist_xy, _, _ = np.histogram2d(x, y, bins=nbins)
    p_xy = hist_xy / hist_xy.sum()

    # Calculate marginal histograms
    hist_x, _ = np.histogram(x, bins=nbins)
    p_x = hist_x / hist_x.sum()

    hist_y, _ = np.histogram(y, bins=nbins)
    p_y = hist_y / hist_y.sum()

    # Calculate entropies
    H_x = -np.sum(p_x * np.log2(p_x + np.finfo(float).eps))
    H_y = -np.sum(p_y * np.log2(p_y + np.finfo(float).eps))
    H_xy = -np.sum(p_xy * np.log2(p_xy + np.finfo(float).eps))

    # Mutual Information I(X;Y) = H(X) + H(Y) - H(X,Y)
    mi = H_x + H_y - H_xy
    return mi

def calculate_s_dependency(df, source_col, target_col, nbins):
    """
    Calculates the S-dependency metric, which is a normalized mutual information.

    Args:
        df (pd.DataFrame): The input DataFrame.
        source_col (str): Name of the source variable column.
        target_col (str): Name of the target variable column.
        nbins (int): Number of bins for entropy calculation.

    Returns:
        float: The S-dependency value.
    """
    x = df[source_col].values
    y = df[target_col].values

    hist_x, _ = np.histogram(x, bins=nbins)
    p_x = hist_x / hist_x.sum()
    H_x = -np.sum(p_x * np.log2(p_x + np.finfo(float).eps))

    hist_y, _ = np.histogram(y, bins=nbins)
    p_y = hist_y / hist_y.sum()
    H_y = -np.sum(p_y * np.log2(p_y + np.finfo(float).eps))

    mi = calculate_mmi(df, source_col, target_col, nbins)

    # S-dependency = MI / min(H(X), H(Y))
    s_dependency = mi / min(H_x, H_y)
    return s_dependency

def normalize_s_dependency(s_dependency, r_min, r_mmi):
    """
    Normalizes the S-dependency value to a range between 0 and 1,
    based on minimum and maximum mutual information values.

    Args:
        s_dependency (float): The S-dependency value to normalize.
        r_min (float): The minimum mutual information value (e.g., from shuffled data).
        r_mmi (float): The maximum mutual information value (e.g., from original data).

    Returns:
        float: The normalized S-dependency value.
    """
    if r_mmi - r_min == 0:
        return 0.0 # Avoid division by zero if min and max are the same
    norm_s_dependency = (s_dependency - r_min) / (r_mmi - r_min)
    return norm_s_dependency