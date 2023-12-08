
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

load("~/data/acinar_sce.rda")

n_seeds <- 5

# genes after preprocessing, forwarded to training
n_input_genes <- 13564  

# SPARSE MODEL
# ------------
acc <- c()
bacc <- c()
dof <- c()

for (i in 1:n_seeds) {
  seed <- floor(runif(1, 1, 9999))
  cat(sprintf("Iter %s Seed %s", i, seed))
  psuper <- psupertime::psupertime(acinar_sce,
                                    acinar_sce$donor_age,
                                    smooth=FALSE,
                                    min_expression = 0.01,
                                    scale=TRUE,
                                    seed=seed,
                                    sel_genes="all",
                                    penalization='1se')  # sparse
  dof <- c(dof, sum(psuper$beta_dt$beta != 0))
  acc <- c(acc, accuracy(psuper$proj_dt$label_input, psuper$proj_dt$label_psuper))
  bacc <- c(bacc, balanced_accuracy(psuper$proj_dt$label_input, psuper$proj_dt$label_psuper))
}

mean_acc <- round(mean(acc) * 100, 2)
std_acc <- round(sd(acc) * 100, 1)
mean_sparsity  <- round((1 - (mean(dof) / n_input_genes)) * 100, 2)
std_sparsity <- round((sd(dof) / n_input_genes) * 100, 1)
cat(sprintf("ACINAR SPARSE LATEXROW: $%s \\pm %s$  &  $%s \\pm %s$", mean_acc, std_acc, mean_sparsity, std_sparsity))


# BEST MODEL
# ----------
acc <- c()
bacc <- c()
dof <- c()

for (i in 1:n_seeds) {
  seed <- floor(runif(1, 1, 9999))
  cat(sprintf("Iter %s Seed %s", i, seed))
  psuper <- psupertime::psupertime(acinar_sce,
                                    acinar_sce$donor_age,
                                    smooth=FALSE,
                                    min_expression = 0.01,
                                    scale=TRUE,
                                    seed=seed,
                                    sel_genes="all",
                                    penalization='best')  # best model
  dof <- c(dof, sum(psuper$beta_dt$beta != 0))
  acc <- c(acc, accuracy(psuper$proj_dt$label_input, psuper$proj_dt$label_psuper))
  bacc <- c(bacc, balanced_accuracy(psuper$proj_dt$label_input, psuper$proj_dt$label_psuper))
}


mean_acc <- round(mean(acc) * 100, 2)
std_acc <- round(sd(acc)*100, 1)
mean_sparsity  <- round((1 - (mean(dof) / n_input_genes)) * 100, 2)
std_sparsity <- round((sd(dof) / n_input_genes) * 100, 1)
cat(sprintf("ACINAR BEST LATEXROW: $%s \\pm %s$  &  $%s \\pm %s$", mean_acc, std_acc, mean_sparsity, std_sparsity))



