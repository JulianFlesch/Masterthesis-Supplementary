import os
import pandas as pd
import numpy as np
from numpy.random import default_rng

import anndata as ad
from scanpy import read_h5ad

data_dir = os.path.join(os.path.dirname(__file__), "data")

def load_h5ad(h5ad_file):
    if os.path.exists(h5ad_file):
        d = read_h5ad(h5ad_file)
        return d
    else:
        print("Data could not be loaded. No such file:", h5ad_file)

def load_acinar(acinar_file=None):
    if not acinar_file:
        acinar_file = os.path.join(data_dir, "acinar_sce.h5ad")
    return load_h5ad(acinar_file)


def load_hesc(hesc_file=None):
    if not hesc_file:
        hesc_file = os.path.join(data_dir, "hesc_sce.h5ad")
    return load_h5ad(hesc_file)


def load_beta(beta_file=None):
    if not beta_file:
        beta_file = os.path.join(data_dir, 'beta_sce.h5ad')
    return load_h5ad(beta_file)


def load_germ(germ_file=None):
    if not germ_file:
        germ_file = os.path.join(data_dir, 'germ_sce.h5ad')
    return load_h5ad(germ_file)


def load_colon(colon_file=None):
    if not colon_file:
        colon_file = os.path.join(data_dir, 'colon_sce.h5ad')
    return load_h5ad(colon_file)


def load_mef(mef_file=None):
    if not mef_file:
        mef_file = os.path.join(data_dir, 'mef_sce.h5ad')
    return load_h5ad(mef_file)


def load_simdata(simdata_file=None):
    if not simdata_file:
        simdata_file = os.path.join(data_dir, "simdata.h5ad")
    return load_h5ad(simdata_file)

 
def restructure_y_to_bin(y_orig):
    '''
    The labels are converted to binary, such that the threshold from 0-1
    corresponds from changing from label $l_i$ to $l_{i+1}$. 
    $k$ copies of the label vector are concatenated such that for every
    vector $j$ the labels  $l_i$ with $i<j$ are converted to 0 and the 
    labels $i\ge j$ are converted to 1.
    '''

    y_classes = np.unique(y_orig)
    k = len(y_classes)

    y_bin = []
    for ki in range(1,k):
        thresh = y_classes[ki]
        y_bin += [int(x >= thresh) for x in y_orig]

    y_bin = np.array(y_bin)

    return y_bin


def restructure_X_to_bin(X_orig, k):
    '''
    The count matrix is extended with copies of itself, to fit the converted label
    vector FOR NOW. For big problems, it could suffice to have just one label 
    vector and perform and iterative training.
    To train the thresholds, $k$ columns are added to the count matrix and 
    initialized to zero. Each column column represents the threshold for a 
    label $l_i$ and is set to 1, exactly  where that label $l_1$ occurs.
    '''

    # X training matrix
    X_bin = np.concatenate([X_orig.copy()] * (k-1))
    # Add thresholds
    num_el = X_orig.shape[0] * (k-1)

    for ki in range(k-1):
        temp = np.repeat(0, num_el).reshape(X_orig.shape[0], (k-1))
        temp[:,ki] = 1
        if ki > 0:
            thresholds = np.concatenate([thresholds, temp])
        else:
            thresholds = temp

    X_bin = np.concatenate([X_bin, thresholds], axis=1)

    return X_bin
