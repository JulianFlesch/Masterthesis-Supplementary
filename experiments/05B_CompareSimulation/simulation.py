import os
from sklearn import metrics
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from masterthesis.data import load_h5ad
from masterthesis.preprocessing import calculate_weights, transform_labels
from sklearn.model_selection import train_test_split
from masterthesis.model_selection import RegularizationGridSearch
from masterthesis.model import SGDBinarizedModel
from sklearn import metrics
import warnings

# report files
genes_outfile = "genes.txt"
results_outfile = "results.txt"

# Fit params
n_seeds = 1
n_folds = 3 #5
n_jobs = -1
n_reg_params = 2 #20
reg_params = np.geomspace(1, 0.005, n_reg_params)
scoring = metrics.make_scorer(metrics.accuracy_score)

# Simulation data
data_dir = "/home/julian/Uni/MasterThesis/data"
filenames = [
    "simdata_TS0.1_SS0.1.h5ad", 
#    "simdata_TS0.1_SS0.3.h5ad",
#    "simdata_TS0.1_SS0.5.h5ad",
#    "simdata_TS0.1_SS0.7.h5ad",
#    "simdata_TS0.1_SS0.9.h5ad",
#    "simdata_TS0.3_SS0.1.h5ad",
#    "simdata_TS0.3_SS0.3.h5ad",
#    "simdata_TS0.3_SS0.5.h5ad",
#    "simdata_TS0.3_SS0.7.h5ad",
#    "simdata_TS0.5_SS0.1.h5ad",
#    "simdata_TS0.5_SS0.3.h5ad",
#    "simdata_TS0.5_SS0.5.h5ad",
#    "simdata_TS0.7_SS0.1.h5ad",
#    "simdata_TS0.7_SS0.3.h5ad",
#    "simdata_TS0.9_SS0.1.h5ad"
]

if __name__ == "__main__":
    warnings.filterwarnings("once")

    genes = []
    results = {
        "file": [],
        "seed": [],
        "dof": [],
        "train_accuracy": [],
        "train_bal_acc": [],
        "train_abs_err": [],
        "test_accuracy": [],
        "test_bal_acc": [],
        "test_abs_err": [],
        "spearman_corr": [],
        "kendall_corr": [],
        "pearson_corr": []
    }

    print("[*] Running Simulation")
    print("[*] Regularization Params = ", reg_params)

    for f in filenames:
        simfile = os.path.join(data_dir, f)
        print("[*] Reading file %s ..." % simfile)
        anndata = load_h5ad(simfile)

        anndata.obs["ordinal_label"] = transform_labels(np.array([int(x) for x in anndata.obs.Ordinal_Time_Labels]))
        X_train, X_test, y_train, y_test = train_test_split(anndata.X, anndata.obs["ordinal_label"], 
                                                            test_size=0.1, 
                                                            stratify=anndata.obs["ordinal_label"],
                                                            random_state=1234)
        
        weights_train = calculate_weights(y_train)
        weights_test = calculate_weights(y_test)

        for i in range(n_seeds):
            
            seed = np.random.randint(9999)
            print("... Iteration %s, Seed=%s" % (i, seed))

            print("... Cross Validation")
            sgd = RegularizationGridSearch(estimator=SGDBinarizedModel,
                                           n_folds=n_folds,
                                           n_jobs=n_jobs,
                                           lambdas=reg_params,
                                           scoring=scoring)
            
            estimator_params = {"random_state": seed}
            fit_params = {"sample_weight": weights_train}
            sgd.fit(X_train, y_train, fit_params=fit_params, estimator_params=estimator_params)

            print("... Refitting on training data")
            sparse_model = sgd.get_optimal_model("1se")
            sparse_model.fit(X_train, y_train)

            # genes weights
            anndata.var["psupertime_weights"] = sparse_model.coef_
            genes += [anndata.var.psupertime_weights[anndata.var.psupertime_weights != 0]]

            # calculate psupertime -> adds anndata.obs.psupertime
            sparse_model.predict_psuper(anndata)
            pearsonr = anndata.obs.Latent_Time.corr(anndata.obs.psupertime)
            spearmanr = anndata.obs.Latent_Time.corr(anndata.obs.psupertime, method='spearman')
            kendalltau = anndata.obs.Latent_Time.corr(anndata.obs.psupertime, method='kendall')

            print("Psuper:", anndata.obs.psupertime)
            print("Latent_Time:", anndata.obs.Latent_Time)

            results["file"] += [f]
            results["seed"] += [seed]
            results["dof"] += [len(np.nonzero(sparse_model.coef_)[0])]
            results["train_accuracy"] += [metrics.accuracy_score(y_train, sparse_model.predict(X_train))]
            results["train_bal_acc"] += [metrics.balanced_accuracy_score(y_train, sparse_model.predict(X_train))]
            results["train_abs_err"] += [metrics.mean_absolute_error(y_train, sparse_model.predict(X_train), sample_weight=weights_train)]
            results["test_accuracy"] += [metrics.accuracy_score(y_test, sparse_model.predict(X_test))]
            results["test_bal_acc"] += [metrics.balanced_accuracy_score(y_test, sparse_model.predict(X_test))]
            results["test_abs_err"] += [metrics.mean_absolute_error(y_test, sparse_model.predict(X_test), sample_weight=weights_test)]
            results["spearman_corr"] += [spearmanr]
            results["kendall_corr"] += [kendalltau]
            results["pearson_corr"] += [pearsonr]

    print("[*] Writing results")
    # Write results to files
    pd.DataFrame(results).to_csv(results_outfile)
    
    with open(genes_outfile, "w") as f:
        for g in genes:#
            if (len(genes) == 0):
                f.write("\n")
            else:
                f.write(", ".join(g.abs().sort_values().index) + "\n")
