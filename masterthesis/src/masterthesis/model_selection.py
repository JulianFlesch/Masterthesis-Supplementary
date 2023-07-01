from collections.abc import Iterable
from sklearn.model_selection import cross_validate
import numpy as np
from .model import BaseModel

class RegularizationGridSearch:

    def __init__(self, estimator=BaseModel, scoring=None, lambdas=None, n_lambdas=50, lambda_high=10, lambda_low=0.0001, n_jobs=-1, n_folds=5):
        
        self.is_fitted = False

        if isinstance(scoring, dict) or isinstance(scoring, Iterable):
            print("Warning: multiple scorers were set as `scoring` parameter which is currently not supported.")
        self.scoring = scoring  # for now, only use default scoring
        self.n_jobs = n_jobs
        self.n_folds = n_folds
        self.estimator = estimator

        # regularization path
        if not isinstance(lambdas, Iterable):
            self.lambdas = np.geomspace(lambda_high, lambda_low, n_lambdas)
        else:
            self.lambdas = lambdas
        
        # average cross validation scores for each lambda
        self.scores = []

        # std of cross validation scores for each lambda
        self.scores_std = []

        # average training scores fore each lambda
        self.train_scores = []

        # std of training scores fore each lambda
        self.train_scores_std = []

        # best degrees of freedom 
        self.dof = []

        # best estimators
        self.fitted_estimators = []

    @staticmethod
    def calc_dof(params):
        """
        Counts the degrees of freedom in a list of parameters
        """
        return np.count_nonzero(params != 0)

    def fit(self, X, y, fit_params=dict(), estimator_params=dict()):

        for i, lamb in enumerate(self.lambdas):
            
            print("Regularization: %s/%s" % (i+1, len(self.lambdas)), sep="", end="\r")

            cv = cross_validate(estimator=self.estimator(regularization=lamb, **estimator_params),
                                scoring=self.scoring,
                                n_jobs=self.n_jobs,
                                cv=self.n_folds,
                                X=X,
                                y=y,
                                error_score="raise",
                                return_train_score=True,
                                return_estimator=True,
                                fit_params=fit_params
                                )

            # TODO: Allow arbitrary scoring. Currently uses only accuracy score.
            #if isinstance(self.scoring, dict) or isinstance(self.scoring, Iterable):
            #    score_keys = list(filter(lambda s: s.startswith("test"), cv.keys()))
            #    self.scores.append({k: cv[k].mean() for k in score_keys})
            #    best_idx = np.argmax(cv[score_keys[0]])
            #else:
            #    best_idx = np.argmax(cv["test_score"])
            #    self.scores.append(cv["test_score"].mean())
            
            best_idx = np.argmax(cv["test_score"])
            self.train_scores.append(np.mean(cv["train_score"]))
            self.train_scores_std.append(np.std(cv["train_score"]))
            self.scores.append(np.mean(cv["test_score"]))
            self.scores_std.append(np.std(cv["test_score"]))
            self.fitted_estimators.append(cv["estimator"][best_idx])
            self.dof.append(self.calc_dof(cv["estimator"][best_idx].coef_))

        self.is_fitted_ = True
        return self

    def get_optimal_lambda(self, method="1se"):

        if not method in ["1se", "best"]:
            raise ValueError("The method parameter should be one of '1se' or 'best'")

        if method == "best":
            idx = np.argmax(self.scores)
            return (self.lambdas[idx], idx)
            
        if method == "1se":
            n = len(self.dof)

            # check the effect direction of the regularization parameter
            sparsity_increases_w_idx = np.mean(self.dof[:n//4]) < np.mean(self.dof[-n//4:])
            
            # TODO: Trim the scores where dof is zero at the sparse end?
            #trimmed = np.array(self.scores) != 0
            #trimmed_max = trimmed[("mean", "accuracy")].max()
            #trimmed_std = trimmed[("mean", "accuracy")].std()
            #thresh = trimmed_max - trimmed_std
            #above = trimmed[trimmed[("mean", "accuracy")] > thresh]
            #if lower_increases_reg:
            #    idx = above.iloc[-1].name
            #else:
            #    idx = above.iloc[0].name

            # compute the threshold as the maximum score minus the standard error
            nonzero_idx = np.nonzero(self.dof)
            thresh = np.max(self.scores) - np.std(np.array(self.scores)[nonzero_idx])

            if sparsity_increases_w_idx:
                items = zip(self.scores, self.dof)
            else:
                items = reversed(list(zip(self.scores, self.dof)))

            for i, (s, d) in enumerate(items):
                # exclude models with 0 degrees of freedom
                if s > thresh and d != 0:
                    return (self.lambdas[i], i)
            
            print("Warning: No model for method '1se' with non-zero degrees of freedom could be found. Returning the best scoring model")
            return self.get_optimal_lambda(method="best")
        
    def get_optimal_parameters(self, *args, **kwargs):
        lamb, idx = self.get_optimal_lambda(*args, **kwargs)
        return self.fitted_estimators[idx].get_params()

    def get_optimal_model(self, *args, **kwargs):
        return self.estimator(**self.get_optimal_parameters(*args, **kwargs))

    def get_estimator_weights_by_lambdas(self):
        pass
