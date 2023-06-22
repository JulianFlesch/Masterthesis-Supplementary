from collections.abc import Iterable
from sklearn.model_selection import cross_validate
import numpy as np
from .model import BaseModel

class RegularizationGridSearch:

    def __init__(self, estimator=BaseModel, scoring=None, lambdas=None, n_lambdas=50, lambda_high=10, lambda_low=0.0001, n_jobs=-1, n_folds=5):
        
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

        # degree of freedom of best models trained during cross validation
        self.dof = []

        # collect the best estimators for each tested regularization
        self.estimators = []

    @staticmethod
    def calc_dof(params):
        """
        Counts the degrees of freedom in a list of parameters
        """
        return np.count_nonzero(params != 0)

    def fit(self, X, y, sample_weight=None):

        for i, lamb in enumerate(self.lambdas):
            
            print("Regularization: %s/%s" % (i+1, len(self.lambdas)), sep="", end="\r")

            cv = cross_validate(estimator=self.estimator(regularization=lamb),
                                scoring=self.scoring,
                                n_jobs=self.n_jobs,
                                cv=self.n_folds,
                                X=X,
                                y=y,
                                error_score="raise",
                                return_estimator=True,
                                fit_params={"sample_weight": sample_weight}
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
            self.scores.append(cv["test_score"].mean())
            self.estimators.append(cv["estimator"][best_idx])
            self.dof.append(self.calc_dof(cv["estimator"][best_idx].beta))

        return self

    def get_optimal_lambda(self, method="1se"):

        if not method in ["1se", "best"]:
            raise ValueError("The method parameter should be one of '1se' or 'best'")

        if method == "best":
            idx = np.argmax(self.scores)
            return self.lambdas[idx]
            
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

            max_idx = np.argmax(self.scores)
            thresh = self.scores[max_idx] - np.std(self.scores)

            if sparsity_increases_w_idx:
                scores = reversed(self.scores)
            else:
                scores = self.scores

            for i, s in enumerate(self.scores):
                if s > thresh:
                    return (self.lambdas[i], i)
        