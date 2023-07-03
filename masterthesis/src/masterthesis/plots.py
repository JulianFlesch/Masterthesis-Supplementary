from .model_selection import RegularizationGridSearch
from .model import BaseModel
from .preprocessing import transform_labels, calculate_weights
import numpy as np
from sklearn import metrics
import seaborn as sns
from matplotlib import pyplot as plt


def plot_grid_search(grid_search: RegularizationGridSearch, title="Grid Search Results"):

    if not isinstance(grid_search, RegularizationGridSearch):
        raise ValueError("The first argument needs to be a completed GridSearch")
    
    if not grid_search.is_fitted_:
        raise ValueError("The grid_search must be run first! Did you call the `.fit()` method?")
    
    best_lambda, best_idx = grid_search.get_optimal_lambda("best")
    print("Best idx:", best_idx, "Best Score:", grid_search.scores[best_idx], "Best Lambda:", grid_search.lambdas[best_idx], "Scores std:", np.std(grid_search.scores))
    ose_lambda, ose_idx = grid_search.get_optimal_lambda("1se")
    print("1SE idx:", ose_idx, "1SE Score:", grid_search.scores[ose_idx], "1SE Lambda:", grid_search.lambdas[ose_idx])

    fig = plt.figure(figsize=(16,4))

    ax1 = fig.add_subplot(131)
    fitted_weights = np.array([e.coef_ for e in grid_search.fitted_estimators])
    ax1.plot(fitted_weights)
    ax1.axvline(best_idx, ymin=0, ymax=1, color="red", label="best", ls="--")
    ax1.axvline(ose_idx, ymin=0, ymax=1, color="blue", label="1se", ls="--")
    ax1.set_xlabel("regularization iteration")
    ax1.set_ylabel("weights")
    #ax1.set_yscale("log")
    ax1.legend()

    ax3 = fig.add_subplot(132)
    ax3.errorbar(x=np.arange(len(grid_search.train_scores))+0.1, y=grid_search.train_scores, yerr=grid_search.train_scores_std, elinewidth=0.5, color="tab:blue", ecolor="skyblue", barsabove=True, label="train scores")
    ax3.errorbar(x=np.arange(len(grid_search.scores)), y=grid_search.scores, yerr=grid_search.scores_std, elinewidth=0.5, color="tab:orange", ecolor="moccasin", barsabove=True, label="test scores")
    ax3.axvline(best_idx, ymin=0, ymax=1, color="red", label="best", ls="--")
    ax3.axvline(ose_idx, ymin=0, ymax=1, color="blue", label="1se", ls="--")
    ax3.set_xlabel("regularization iteration")
    ax3.set_ylabel("mean cv score")
    ax3.legend()

    ax2 = fig.add_subplot(133)
    ax2.plot(grid_search.dof)
    ax2.axvline(best_idx, ymin=0, ymax=1, color="red", label="best", ls="--")
    ax2.axvline(ose_idx, ymin=0, ymax=1, color="blue", label="1se", ls="--")
    ax2.set_xlabel("regularization iteration")
    ax2.set_ylabel("Degrees of Freedom")

    fig.suptitle(title)

    return fig


def plot_model_perf(model, train, test, title="Model Predictions"):

    if not isinstance(model, BaseModel):
        raise ValueError("The first argument needs to be a fitted sklearn model")
    
    if not model.is_fitted_:
        raise ValueError("The grid_search must be run first! Did you call the `.fit()` method?")
    
    if not isinstance(train, tuple) or not isinstance(test, tuple):
        raise ValueError("The parameters `train`, `test` are expected to be tuples of np.array")

    X_test, y_test = test
    X_train, y_train = train
    labels = np.unique(np.concatenate([y_test, y_train]))
    y_test_trans = transform_labels(y_test, labels=labels)
    y_train_trans = transform_labels(y_train, labels=labels)
    
    weights_train = calculate_weights(y_train_trans)
    weights_test = calculate_weights(y_test_trans)

    print("Degrees of freedom", len(np.nonzero(model.coef_)[0]))
    print("Train:")
    print("Accuracy:", metrics.accuracy_score(y_train_trans, model.predict(X_train)))
    print("Balanced accuracy:", metrics.balanced_accuracy_score(y_train_trans, model.predict(X_train)))
    print("Mean absolute delta:", metrics.mean_absolute_error(y_train_trans, model.predict(X_train), sample_weight=weights_train))
    print("Test:")
    print("Accuracy:", metrics.accuracy_score(y_test_trans, model.predict(X_test)))
    print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test_trans, model.predict(X_test)))
    print("Mean absolute delta:", metrics.mean_absolute_error(y_test_trans, model.predict(X_test), sample_weight=weights_test))

    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    sns.heatmap(metrics.confusion_matrix(y_test_trans, model.predict(X_test)), annot=True, ax=ax1)
    sns.heatmap(metrics.confusion_matrix(y_train_trans, model.predict(X_train)), annot=True, ax=ax2)

    ax1.set_xlabel("Test-Split Preditions")
    ax1.set_ylabel("Ground Truth Labels")
    ax2.set_xlabel("Train-Split Predictions")

    fig.suptitle(title)

    return fig


def plot_identified_gene_coefficients(*args, **kwargs):
    #TODO
    pass


def plot_identified_genes_over_psupertime(n = 20, *args, **kwargs):
    # TODO
    pass


def plot_labels_over_psupertime(*args, **kwargs):
    # TODO
    pass
