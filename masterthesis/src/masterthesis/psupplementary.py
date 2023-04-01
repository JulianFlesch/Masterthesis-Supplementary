import os
from .data import data_dir
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from rpy2.robjects.lib import ggplot2


psupertime = importr("psupertime")
grdevices = importr("grDevices")  # to close graphical windows again


def close_rgraphics():
    grdevices.dev_off()


def load_acinar():
    #load acinar_sce data set into the r context
    acinar_sce = r.data(os.path.join(data_dir, "acinar_sce.rda"))


def run_psupertime():
    # psuper_obj = psupertime.psupertime(r.acinar_sce, y_age)
    r('psuper_obj = psupertime::psupertime(acinar_sce, acinar_sce$donor_age)')


def identify_hvg():
    r('''
        hvg_params = list(hvg_cutoff=0.1, bio_cutoff=0.5, span=0.1)
        sel_genes = psupertime:::.calc_hvg_genes(acinar_sce, hvg_params)
        acinar_hvg = acinar_sce[sel_genes, ]
    ''')


def calc_umap_projection():
    r('''
        library('umap')
        x = t(SummarizedExperiment::assay(acinar_hvg, 'logcounts'))
        wellKey_vector = SingleCellExperiment::colData(acinar_hvg)$wellKey
        label_vector = factor(SingleCellExperiment::colData(acinar_hvg)[['donor_age']])
        proj_umap = psupplementary:::calc_umap(x, wellKey_vector)
        u1 = proj_umap$umap1
        u2 = proj_umap$umap2
    ''')


def plot_hvg_umap():

    data_table = importr("data.table")

    plot_dt = data_table.data_table(
            y_var = r("label_vector")
            ,dim1 = r("proj_umap$umap1")
            ,dim2 = r("proj_umap$umap2")
            )

    col_vals = psupertime._make_col_vals(plot_dt[0])

    labels = r.c('umap1', 'umap2', 'Donor age (years)')

    pp = ggplot2.ggplot(plot_dt) + \
            ggplot2.aes( x=plot_dt[1], y=plot_dt[2], colour=plot_dt[0] ) + \
            ggplot2.geom_point() + \
            ggplot2.scale_colour_manual( values=col_vals ) + \
            ggplot2.labs(
                x = labels[0],
                y = labels[1],
                colour = labels[2]
            ) + \
            ggplot2.theme_bw() + \
            ggplot2.theme(
                axis_text = ggplot2.element_blank()
            )


def run_acinar():

    print("Loading Data")
    load_acinar()


    print("Identifying highly variant genes")
    identify_hvg()

    print("UMAP projection of highly variant genes")
    calc_umap_projection()

    print("Plot highly variant genes")
    plot_hvg_umap()

    print("Running psupertime")
    run_psupertime()

    print("Plot psupertime genes")

    return grdevices
