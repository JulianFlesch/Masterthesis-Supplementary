import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
import scanpy as sc
import anndata as ad


def transform_labels(y, labels=None):
    """
    Transforms a target vector, such that it contains successive labels starting at 0.
    """

    # if labels are not given, compute
    if labels is None: 
        labels = np.unique(y)

    transf = dict(zip(labels,
                      np.arange(0, len(labels))))

    y_trans = np.array([transf[e] for e in y])
    return y_trans


def calculate_weights(y):
    """
    Calculates the weights for each label in a target vector.
    The weight of label is the number of occurances 
    divided by the total number of labels.
    """
    _, counts = np.unique(y, return_counts=True)
    weights = [counts[el] / len(y) for el in y]
    return weights


class Preprocessing(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 log: bool = False,
                 scale: bool = False,
                 normalize: bool = False,
                 smooth: bool = False,
                 select_genes: str = "all",
                 gene_list: ArrayLike | None = None,
                 min_gene_mean: float = 0.1,
                 max_gene_mean: float = 3,
                 hvg_min_dispersion: float = 0.5,
                 hvg_max_dispersion: float = np.inf,
                 hvg_n_top_genes: int | None = None):
        
        self.scale = scale
        self.log = log
        self.normalize = normalize
        self.smooth = smooth

        select_genes_options = {"all", "hvg", "tf_mouse", "tf_human"}
        if select_genes not in select_genes_options and \
            not isinstance(select_genes, ArrayLike):
            raise ValueError("Parameter select_genes must be one of %s or a list of gene ids." % select_genes_options)
        self.select_genes = select_genes

        self.gene_list = gene_list
        self.min_gene_mean = min_gene_mean
        self.max_gene_mean = max_gene_mean
        self.hvg_min_dispersion = hvg_min_dispersion
        self.hvg_max_dispersion = hvg_max_dispersion
        self.hvg_n_top_genes = hvg_n_top_genes

    def fit_transform(self, adata: ad.AnnData, y: ArrayLike | None = None, **fit_params) -> ndarray:
        
        # filter genes by their minimum mean counts
        cell_thresh = np.ceil(0.01 * adata.n_obs)
        sc.pp.filter_genes(adata, min_counts=cell_thresh)

        # log transform: adata.X = log(adata.X + 1)
        if self.log: sc.pp.log1p(adata)

        # select highly-variable genes
        if self.select_genes == "hvg":
            sc.pp.highly_variable_genes(
                adata, flavor='seurat', n_top_genes=self.hvg_n_top_genes,
                min_disp=self.hvg_min_dispersion, max_disp=self.hvg_max_dispersion, inplace=True
            )
            adata = adata[:,adata.var.highly_variable].copy()
        
        # select mouse transcription factors
        elif self.select_genes == "tf_mouse":
            raise NotImplemented("select_genes='tf_mouse'")

        # select human transcription factors
        elif self.select_genes == "tf_human":
            raise NotImplemented("select_genes='tf_human'")

        # select curated genes from list
        # TODO: Check for array
        elif self.gene_list is not None:
            raise NotImplemented("gene_list")
        
        # select all genes
        elif self.select_genes == "all":
            pass

        # normalize with total UMI count per cell
        # this helps keep the parameters small
        if self.normalize: sc.pp.normalize_per_cell(adata)

        # scale to unit variance and shift to zero mean
        if self.scale: sc.pp.scale(adata)

        if self.smooth:
            raise NotImplemented("smoothing is not implemented")

        return adata
