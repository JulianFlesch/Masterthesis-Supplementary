
mean_abs_error <- function(y, predicted) {
  mean(abs(y - predicted))
}

accuracy <- function(y, predicted) {
  sum(y == predicted) / length(y)
}

balanced_accuracy <- function(y, predicted) {
  y = y + 1
  predicted = predicted + 1
  
  w <- length(y) / (length(unique(y)) * y)
  temp = c()
  for (i in 1:length(y)) { temp[i] = sum(y[i] == y) * w[i] }
  w_adj = w / temp
  
  ( 1/sum(w_adj) ) * sum ((y == predicted) * w_adj)  
}


filenames = c('/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.1_SS0.1.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.1_SS0.3.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.1_SS0.5.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.1_SS0.7.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.1_SS0.9.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.3_SS0.1.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.3_SS0.3.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.3_SS0.5.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.3_SS0.7.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.5_SS0.1.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.5_SS0.3.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.5_SS0.5.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.7_SS0.1.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.7_SS0.3.h5ad',
              '/home/julian/Uni/MasterThesis/data/simdata_v2_TS0.9_SS0.1.h5ad')

n_seeds = 5
n_folds = 5
n_lambdas = 40

genes_outfile = "results/genes_R.txt"
results_outfile = "results/results_R.txt"

# collect results
seeds <- c()
all_seeds <- c()
files <- c()
dof <- c()
mae <- c()
acc <- c()
bacc <- c()
cor_spearman <- c()
cor_pearson <- c()

genes = list()

for (j in 1:length(filenames)) {
  f = filenames[j]
  cat("... Reading File ", f, "\n")
  
  for (i in 1:n_seeds) {
    seed <- floor(runif(1, 1, 9999))
    cat("... Iter ", i, " Seed: ", seed, "\n")

    simdata <- zellkonverter::readH5AD(f)
    pobj <- psupertime::psupertime(simdata, simdata$Ordinal_Time_Labels, scale=FALSE, sel_genes="all", seed = seed)
    
    y <- as.integer(pobj$proj_dt$label_input)
    y_pred <- as.integer(pobj$proj_dt$label_psuper)
    
    # collect results
    files <- c(files, f)
    all_seeds <- c(all_seeds, seed)
    dof <- c(dof, sum(pobj$beta_dt$beta != 0))
    mae <- c(mae, mean_abs_error(y, y_pred))
    acc <- c(acc, accuracy(y, y_pred))
    bacc <- c(bacc, balanced_accuracy(y, y_pred))
    cor_spearman <- c(cor_spearman, cor(simdata$Latent_Time, pobj$proj_dt$psuper, method="spearman"))
    cor_pearson <- c(cor_pearson, cor(simdata$Latent_Time, pobj$proj_dt$psuper, method = "pearson"))
    
    genes[[ (j-1) * n_seeds + i ]] = as.character(pobj$beta_dt[pobj$beta_dt$abs_beta != 0]$symbol)
    
  }
}

df <- data.frame(files=files, seed=seeds, dof=dof, mae=mae, acc=acc, bacc=bacc, cor_spearman=cor_spearman, cor_pearson=cor_pearson)
write.csv(df, results_outfile, quote = FALSE)

sink(genes_outfile)
for (i in 1:length(genes)) {
    cat(genes[[i]])
    cat("\n")
}
sink()