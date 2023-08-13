import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
import scanpy as sc
import anndata as ad
from sklearn.utils import class_weight


def transform_labels(y, labels=None):
    """
    Transforms a target vector, such that it contains successive labels starting at 0.
    """

    # TODO: Check if y is numerical (i.e. int) -> Raise error otherwise
    # BUG: Passing string labels that represent a numerical ordering can cause issues, 
    # because np.unique sorts implicitly and string sorting != int sorting


    # if labels are not given, compute
    if labels is None: 
        labels = np.unique(y)

    transf = dict(zip(labels,
                      np.arange(0, len(labels))))

    y_trans = np.array([transf[e] for e in y])
    return y_trans


def calculate_weights(y):
    """
    Calculates weights from the classes in y. 
    Returns an array the same length as y, 
    where the class is replaced with their respective weight

    Calculates balanced class weights according to
    `n_samples / (n_classes * np.bincount(y))`
    as is done in sklearn.
    """
    classes = np.unique(y)
    weights = class_weight.compute_class_weight("balanced", classes=classes, y=y)
    
    transf = dict(zip(classes, weights))

    return np.array([transf[e] for e in y])


def smooth(adata, k=10):
    from scipy import stats

    knn = 10

    # corellate all cells
    cor_mat = np.corrcoef(adata.X)

    # calculate the ranks of cell correlations
    order_mat = np.argsort(cor_mat, axis=1)
    rank_mat = np.argsort(order_mat, axis=1)

    # indicate the knn closest neighbours
    idx_mat = rank_mat <= knn

    # calculate the neighborhood average
    avg_knn_mat = idx_mat / np.sum(idx_mat, axis=1, keepdims=True)
    assert np.all(np.sum(avg_knn_mat, axis=1) == 1)

    # 
    imputed_mat = np.dot(avg_knn_mat, adata.X)
    adata.X = imputed_mat

    

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
        sc.pp.filter_genes(adata, min_cells=cell_thresh)

        # log transform: adata.X = log(adata.X + 1)
        if self.log: sc.pp.log1p(adata, copy=False)

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
