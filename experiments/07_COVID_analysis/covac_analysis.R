library("SingleCellExperiment")
covac_sce <- zellkonverter::readH5AD("../data/COVAC_POSTQC.h5ad")

counts(covac_sce) <- assay(covac_sce)

# normalize 
sf <- 2^rnorm(ncol(covac_sce))
sf <- sf / mean(sf)
normcounts(covac_sce) <- t(t(counts(covac_sce)) / sf)
logcounts(covac_sce) <- log2(normcounts(covac_sce) + 1)

# create ordinal data vector
covac_sce$ordinal_label <- covac_sce$timepoint
levels(covac_sce$ordinal_label) <- list("t1"=0, "t2"=1, "t3"=2, "t4"=3, "d1"=0, "d28"=1, "d56"=2, "M6"=3)
covac_sce$ordinal_label <- as.numeric(covac_sce$ordinal_label)

# subset to only the P cohort
covac_subset <- covac_sce[, covac_sce$cohort == "P"]

# Run Psupertime
pobj <- psupertime::psupertime(covac_subset, covac_subset$ordinal_label)
