import datetime
import os
from sklearn import metrics
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from scanpy import read_h5ad
from sklearn import metrics
from pypsupertime import Psupertime

import warnings

start_time = datetime.datetime.now()

# Fit params
preprocessing_params = {"log": False, "normalize": False, "scale": False}
regularization_params = {"n_folds": 5, "n_jobs": 5, "n_params": 40, "scoring": metrics.make_scorer(metrics.accuracy_score)}
estimator_params = {"max_iter": 1000, "early_stopping": True, "penalty": "elasticnet"}

n_batches_options = [1, 5, 10, 15, 20]
l1_ratio_options = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

n_seeds = 5

# Simulation data
data_dir = "/home/ubuntu/data"
filenames = [
    "simdata_v2_TS0.1_SS0.1.h5ad", 
    "simdata_v2_TS0.1_SS0.3.h5ad",
    "simdata_v2_TS0.1_SS0.5.h5ad",
    "simdata_v2_TS0.1_SS0.7.h5ad",
    "simdata_v2_TS0.1_SS0.9.h5ad",
    "simdata_v2_TS0.3_SS0.1.h5ad",
    "simdata_v2_TS0.3_SS0.3.h5ad",
    "simdata_v2_TS0.3_SS0.5.h5ad",
    "simdata_v2_TS0.3_SS0.7.h5ad",
    "simdata_v2_TS0.5_SS0.1.h5ad",
    "simdata_v2_TS0.5_SS0.3.h5ad",
    "simdata_v2_TS0.5_SS0.5.h5ad",
    "simdata_v2_TS0.7_SS0.1.h5ad",
    "simdata_v2_TS0.7_SS0.3.h5ad",
    "simdata_v2_TS0.9_SS0.1.h5ad"
]

from pypsupertime.preprocessing import transform_labels, calculate_weights

# report files
genes_outfile = "genes_py.txt"
results_outfile = "results_py.txt"

warnings.filterwarnings("once")

genes = []
results = {
    "file": [],
    "l1_ratio": [],
    "n_batches": [],
    "seed": [],
    "best_reg": [],
    "dof": [],
    "all_accuracy": [],
    "all_bal_acc": [],
    "all_abs_err": [],
    "spearman_corr": [],
    "pearson_corr": [],
    "precision": [],
    "sensitivity": [],
}

print("[*] Running Simulation")

for f in filenames:
    simfile = os.path.join(data_dir, f)
    anndata = read_h5ad(simfile)
    #anndata.obs["ordinal_label"] = transform_labels(np.array([int(x) for x in anndata.obs.Ordinal_Time_Labels]))
    weights_all = calculate_weights(anndata.obs.Ordinal_Time_Labels)
    
    for l1_ratio in l1_ratio_options:
        for n_batches in n_batches_options:
            for i in range(n_seeds):
        
                seed = np.random.randint(9999)
                print("... L1_ratio=%s, n_batches=%s, File=%s, Seed=%s" % (l1_ratio, n_batches, f, seed))

                estimator_params["random_state"] = seed
                estimator_params["n_batches"] = n_batches
                estimator_params["l1_ratio"] = l1_ratio

                p = Psupertime(estimator_params=estimator_params,
                               preprocessing_params=preprocessing_params,
                               regularization_params=regularization_params)

                _ = p.run(anndata, "Ordinal_Time_Labels")
                
                # Annotate genes weights manually
                p.model.gene_weights(anndata, inplace=True)
                
                genes += [anndata.var.psupertime_weight[anndata.var.psupertime_weight != 0]]
        
                pearsonr = anndata.obs.Latent_Time.corr(anndata.obs.psupertime)
                spearmanr = anndata.obs.Latent_Time.corr(anndata.obs.psupertime, method='spearman')
                kendalltau = anndata.obs.Latent_Time.corr(anndata.obs.psupertime, method='kendall')
        
                results["file"] += [f]
                results["seed"] += [seed]
                results["n_batches"] += [n_batches]
                results["l1_ratio"] += [l1_ratio]
                results["best_reg"] += [p.model.regularization]
                dof = len(np.nonzero(p.model.coef_)[0])
                results["dof"] += [dof]
                
                # scores on all data (for comparison, because psupertime only measures this)
                acc = metrics.accuracy_score(anndata.obs.Ordinal_Time_Labels, anndata.obs.predicted_label)
                bacc = metrics.balanced_accuracy_score(anndata.obs.Ordinal_Time_Labels, anndata.obs.predicted_label)
                abs_err = metrics.mean_absolute_error(anndata.obs.Ordinal_Time_Labels,
                                                      anndata.obs.predicted_label,
                                                      sample_weight=weights_all)
                results["all_accuracy"] += [acc]
                results["all_bal_acc"] += [bacc]
                results["all_abs_err"] += [abs_err]
                
                # correlation
                results["spearman_corr"] += [spearmanr]
                results["pearson_corr"] += [pearsonr]
                
                # identification of significant genes
                TP = sum([g in anndata.var[anndata.var.Setting == "TS"].index for g in anndata.var[anndata.var.psupertime_weight.abs() != 0].index])
                FP = len(anndata.var[anndata.var.psupertime_weight.abs() != 0].index) - TP
                P = anndata.var[anndata.var.Setting == "TS"].shape[0]
                results["sensitivity"] += [TP / P]
                results["precision"] += [TP / (TP + FP) if TP + FP > 0 else 0]
                
                print("... dof:", dof,  "train_bacc:", bacc, "test_bacc", acc, "spear_cor", spearmanr)


# Write results to files
pd.DataFrame(results).to_csv(results_outfile)

# Write Genes and weights
with open(genes_outfile, "w") as f:
    for g in genes:#
        if (len(genes) == 0):
            f.write("\n\n")
        else:
            f.write(", ".join(g.abs().sort_values().index) + "\n")
            f.write(", ".join([str(el) for el in g.abs().sort_values()]) + "\n")

warnings.filterwarnings("always")

# Track timing
elapsed_time = datetime.datetime.now() - start_time
print(elapsed_time)
