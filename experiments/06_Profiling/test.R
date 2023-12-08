#!/bin/Rscript

load("/home/julian/Uni/MasterThesis/data/acinar_sce.rda")
psupertime::psupertime(acinar_sce, acinar_sce$donor_age, sel_genes="all")

