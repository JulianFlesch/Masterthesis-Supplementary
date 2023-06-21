from sklearn.model_selection import cross_validate
import numpy as np
from .model import BaseModel

class RegularizationGridSearch:

    def __init__(self, estimator=BaseModel, scoring=None, lambdas=None, n_lambdas=50, lambda_high=10, lambda_low=0.0001, n_jobs=-1, n_folds=5):
        
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.n_folds = n_folds
        self.estimator = estimator

        # regularization path
        if lambdas:
            self.lambdas = lambdas
        else:
            self.lambdas = np.geomspace(lambda_high, lambda_low, n_lambdas)

        # average cross validation scores for each lambda
        self.scores = np.zeros(len(self.lambdas))

        # average degree of freedom of all models trained during cross validation
        self.dof = np.zeros(len(self.lambdas))

        # collect the best estimators for each tested regularization
        self.estimators = []

    @staticmethod
    def calc_dof(params):
        """
        Counts the degrees of freedom in a list of parameters
        """
        return np.count_nonzero(params != 0)

    def fit(self, X, y):

        for i, lamb in enumerate(self.lambdas):

            cv = cross_validate(estimator=self.estimator(regularization=lamb),
                                scoring=self.scoring,
                                n_jobs=self.n_jobs,
                                cv=self.n_folds,
                                X=X,
                                y=y,
                                error_score="raise",
                                return_estimator=True
                                )

            best_idx = np.argmax(cv["test_score"])
            self.estimators[i] = cv["estimator"][best_idx]
            self.dof[i] = self.calc_dof(cv["estimator"][best_idx])
            
            self.scores[i] = cv["test_score"].mean()

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
            sparsity_increases_w_idx = np.mean(self.dof[:n/4]) > np.mean(self.dof[-n/4:])

            trimmed = res_df.loc[res_df[("mean", "dof")] != 0]
            trimmed_max = trimmed[("mean", "accuracy")].max()
            trimmed_std = trimmed[("mean", "accuracy")].std()
            thresh = trimmed_max - trimmed_std
            above = trimmed[trimmed[("mean", "accuracy")] > thresh]

            if lower_increases_reg:
                idx = above.iloc[-1].name
            else:
                idx = above.iloc[0].name

            print("max:", trimmed_max, "std:", trimmed_std, "thresh:", thresh)
            print("Best average fit:", trimmed.loc[idx])
            print("Best parameter:", reg_params[idx])
            
            return reg_params[idx]        