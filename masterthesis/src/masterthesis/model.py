from abc import ABC, ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
import numpy as np
from numpy.random import default_rng
from .data import restructure_X_to_bin, restructure_y_to_bin


class BaseModel(ClassifierMixin, BaseEstimator, ABC):
    random_state: int = 1234
    beta: np.array
    theta: np.array
    k: int = 0

    @abstractmethod
    def fit(self, data, target, **kwargs):
        print("No fitting implemented")
        self.is_fitted_ = True
        return self

    def cv_fit(self, X, y, n_folds=5):
        #kf = StratifiedKFold(n_splits=n_folds)
        pass
    
    def predict_proba(self, X):

        # TODO: Implement differnet methods "forward", "Backward", "prop-odds"
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        # class probabilities are a matrix of shape (n_cells x n_classs) 
        pred = np.zeros(X.shape[0] * (self.k + 1)).reshape(X.shape[0], self.k + 1)

        # save matrix multiplication
        transform = X @ self.beta

        for i in range(self.k):
            pred[:, i] = 1 / (1 + np.exp(transform - self.theta[i]))
        
        pred[:, self.k] = 1 - np.sum(pred[:, :-1], axis=1)

        for i in range(self.k, 0, -1):
            pred[:, i] = pred[:, i] - pred[:, i-1]

        return pred

    def predict(self, X):
        return np.apply_along_axis(np.argmax, 1, self.predict_proba(X))


class BinaryModelMixin(metaclass=ABCMeta):

    def _check_and_restructure_X_y(self, data, targets):
        # check input data
        X, y = check_X_y(data, targets, accept_sparse=True)
        
        # convert to binary problem
        self.k = np.unique(y).size - 1
        X_bin = restructure_X_to_bin(X, n_thresholds=self.k)
        y_bin = restructure_y_to_bin(y)

        return X_bin, y_bin
    
    def _after_fit(self, model):
        # extract the thresholds and weights
        # from the 2D coefficients matrix in the sklearn model
        self.theta = model.coef_[0, -self.k:][::-1]  # thresholds
        self.beta = model.coef_[0, :-self.k]   # weights

        self.is_fitted_ = True

    @abstractmethod
    def _fit_binary_model(self, X_bin, y_bin):
        return LogisticRegression()

    def fit(self, data, target, **kwargs):

        X, y = self._check_and_restructure_X_y(data, target)
        model = self._fit_binary_model(X, y)
        self._after_fit(model)

        return self


class LinearBinarizedModel(BinaryModelMixin, BaseModel):

    def __init__(self, max_iter=10000, solver="liblinear", random_state=1234, regularization=0.01):
        
        # model hyperparameters
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.regularization = regularization

        # fitting/data parameters
        self.k = None
        self.theta = []
        self.beta = []

    def _fit_binary_model(self, X_bin, y_bin):

        model = LogisticRegression(penalty="l1", 
                                  fit_intercept=False,
                                  max_iter=self.max_iter,
                                  solver=self.solver,
                                  random_state=self.random_state,
                                  C=1 / self.regularization  # Inverse of regularization strength -> controls sparsity in our case!
                                )

        model.fit(X_bin, y_bin)

        return model


class SGDBinarizedModel(BinaryModelMixin, BaseModel):
    def __init__(self, max_iter=5, n_batches=2, random_state=1234, regularization=0.01):
        
        # model hyperparameters
        self.max_iter = max_iter
        self.n_batches = n_batches
        self.random_state = random_state
        self.rng = default_rng(seed=self.random_state)
        self.regularization = regularization

        # fitting/data parameters
        self.k = None
        self.theta = []
        self.beta = []

    def _fit_binary_model(self, X_bin, y_bin):

        model = SGDClassifier(loss="log_loss",
                              random_state=self.random_state,
                              penalty="l1",
                              alpha=self.regularization,
                              fit_intercept=False,
                              n_jobs=1)

        cur_iter = 0

        while cur_iter < self.max_iter:
            if (cur_iter > 0 and cur_iter % 2 == 0):
                print("Iter: ", cur_iter, "Train score: ", model.score(X_batch, y_batch))
            
            cur_iter += 1
            
            # fit from samples of the big matrix
            # TODO: Sampling from the big matrix directly is just for PoP,
            # and eliminates the purpose. Only the binarized y-vector should
            # be created and the indexes taken from the log count matrix.
            sampled_indices = self.rng.integers(X_bin.shape[0], size=X_bin.shape[0])

            start = 0
            for i in range(1, self.n_batches+1):
                end = (i * X_bin.shape[0] // self.n_batches)
                idx = sampled_indices[start:end]
                X_batch = X_bin[idx,:]
                y_batch = y_bin[idx]
                start = end
                model.partial_fit(X_batch, y_batch, classes=np.unique(y_batch))

        return model
