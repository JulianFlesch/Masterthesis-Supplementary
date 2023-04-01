
load("/home/julian/Uni/MasterThesis/psupplementary/data/acinar_sce.rda")

lambdas 	= 10^seq(from=0, to=-4, by=-0.1)
xt <- t(SingleCellExperiment::logcounts(acinar_sce))
y <- acinar_sce$donor_age

newx <- psupertime:::.restructure_propodds(xt, y, rep(1, length(y)))
