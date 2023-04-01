from scanpy import read_h5ad


DATA_DIR = "/home/julian/Uni/MasterThesis/psupplementary/data/"

def load_acinar():
    acinar = read_h5ad(DATA_DIR + "acinar_sce.h5ad")
    return acinar


def load_hesc():
    hesc = read_h5ad(DATA_DIR + "hesc_sce.h5ad")
    return hesc


def load_beta():
    beta = read_h5ad(DATA_DIR + 'beta_sce.h5ad')
    return beta


def load_germ():
    germ = read_h5ad(DATA_DIR + 'germ_sce.h5ad')
    return germ


def load_colon():
    colon = read_h5ad(DATA_DIR + 'colon_sce.h5ad')
    return colon


def load_mef():
    mef = read_h5ad(DATA_DIR + 'mef_sce.h5ad')
    return mef
