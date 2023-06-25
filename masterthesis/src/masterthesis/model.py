from abc import ABC, ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
import numpy as np
from numpy.random import default_rng


class BaseModel(ClassifierMixin, BaseEstimator, ABC):
    regularization: float
    random_state: int = 1234
    coef_: np.array
    intercept_: np.array
    k: int = 0
    classes_: np.array
    is_fitted_: bool = False

    @abstractmethod
    def fit(self, data, target, **kwargs):
        print("No fitting implemented")
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):

        # TODO: Implement differnet methods "forward", "Backward", "prop-odds"
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        # class probabilities are a matrix of shape (n_cells x n_classs) 
        pred = np.zeros(X.shape[0] * (self.k + 1)).reshape(X.shape[0], self.k + 1)

        # save matrix multiplication
        transform = X @ self.coef_

        for i in range(self.k):
            pred[:, i] = 1 / (1 + np.exp(transform - self.intercept_[i]))
        
        pred[:, self.k] = 1 - np.sum(pred[:, :-1], axis=1)

        for i in range(self.k, 0, -1):
            pred[:, i] = pred[:, i] - pred[:, i-1]

        return pred

    def predict(self, X):
        return np.apply_along_axis(np.argmax, 1, self.predict_proba(X))


class BinaryModelMixin(metaclass=ABCMeta):
    binary_estimator_: BaseEstimator = None

    def _check_X_y(self, data, targets):
        # check input data
        return check_X_y(data, targets, accept_sparse=True)

    @staticmethod
    def restructure_y_to_bin(y_orig):
        '''
        The labels are converted to binary, such that the threshold from 0-1
        corresponds from changing from label $l_i$ to $l_{i+1}$. 
        $k$ copies of the label vector are concatenated such that for every
        vector $j$ the labels  $l_i$ with $i<j$ are converted to 0 and the 
        labels $i\ge j$ are converted to 1.
        '''

        y_classes = np.unique(y_orig)
        k = len(y_classes)

        y_bin = []
        for ki in range(1,k):
            thresh = y_classes[ki]
            y_bin += [int(x >= thresh) for x in y_orig]

        y_bin = np.array(y_bin)

        return y_bin

    @staticmethod
    def restructure_X_to_bin(X_orig, n_thresholds):
        '''
        The count matrix is extended with copies of itself, to fit the converted label
        vector FOR NOW. For big problems, it could suffice to have just one label 
        vector and perform and iterative training.
        To train the thresholds, $k$ columns are added to the count matrix and 
        initialized to zero. Each column column represents the threshold for a 
        label $l_i$ and is set to 1, exactly  where that label $l_1$ occurs.
        '''

        # X training matrix
        X_bin = np.concatenate([X_orig.copy()] * (n_thresholds))
        # Add thresholds
        num_el = X_orig.shape[0] * (n_thresholds)

        for ki in range(n_thresholds):
            temp = np.repeat(0, num_el).reshape(X_orig.shape[0], (n_thresholds))
            temp[:,ki] = 1
            if ki > 0:
                thresholds = np.concatenate([thresholds, temp])
            else:
                thresholds = temp

        X_bin = np.concatenate([X_bin, thresholds], axis=1)

        return X_bin

    def _before_fit(self, data, targets):
        data, targets = check_X_y(data, targets)
        self.classes_ = np.unique(targets)
        self.k = len(self.classes_) - 1
        return data, targets

    def _restructure_X_y(self, data, targets):
        # convert to binary problem
        X_bin = self.restructure_X_to_bin(data, n_thresholds=self.k)
        y_bin = self.restructure_y_to_bin(targets)

        return X_bin, y_bin
    
    def _after_fit(self, model):
        # extract the thresholds and weights
        # from the 2D coefficients matrix in the sklearn model
        self.intercept_ = model.coef_[0, -self.k:][::-1]  # thresholds
        self.coef_ = model.coef_[0, :-self.k]   # weights

        self.is_fitted_ = True

    def _get_estimator(self):
        return self.binary_estimator_

    @abstractmethod
    def fit(self, data, targets, sample_weight=None):
        pass


class LinearBinarizedModel(BinaryModelMixin, BaseModel):

    def __init__(self, max_iter=10000, solver="liblinear", random_state=1234, regularization=0.01):
        
        # model hyperparameters
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.regularization = regularization
        self.binary_estimator_ = None

        # fitting/data parameters
        self.k = None
        self.intercept_ = []
        self.coef_ = []

    def _get_estimator(self):
        if self.binary_estimator_ is not None:
            self.binary_estimator_ =  LogisticRegression(penalty="l1", 
                                                    fit_intercept=False,
                                                    max_iter=self.max_iter,
                                                    solver=self.solver,
                                                    random_state=self.random_state,
                                                    C=1/self.regularization  # Inverse of regularization strength -> controls sparsity in our case!
                                                    )
        
        return self.binary_estimator_

    def fit(self, data, targets, sample_weight=None):
        data, targets = self._before_fit(data, targets)
        data, targets = self._restructure_X_y(data, targets)

        model = self._get_estimator()
        
        weights = np.tile(sample_weight, self.k) if sample_weight is not None else None
        model.fit(data, targets, sample_weight=weights)
        self._after_fit(model)

        return self

class LassoBinarizedModel(LinearBinarizedModel):
    def __init__(self, max_iter=10000, tol=1e-4, random_state=1234, regularization=0.01):
        
        # model hyperparameters
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.regularization = regularization
        self.binary_estimator_ = None

        # fitting/data parameters
        self.k = None
        self.intercept_ = []
        self.coef_ = []

    def _get_estimator(self):
        if self.binary_estimator_ is not None:
            self.binary_estimator_ =  \
                Lasso(fit_intercept=False,
                    tol=self.tol,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    alpha=self.regularization
                    )

        return self.binary_estimator_


class SGDBinarizedModel(BinaryModelMixin, BaseModel):

    def __init__(self, max_iter=10, n_batches=1, random_state=1234, regularization=0.01):
        
        # model hyperparameters
        self.max_iter = max_iter
        self.n_batches = n_batches
        self.random_state = random_state
        self.regularization = regularization
        self.rng = default_rng(seed=self.random_state)

        # fitting/data parameters
        self.k = None
        self.intercept_ = []
        self.coef_ = []

    def _restructure_X_y(self, data, targets):
        # overwrites the superclass method to 
        # only convert the target vector to binary problem
        y_bin = self.restructure_y_to_bin(targets)

        return data, y_bin
    
    def fit(self, X, y, sample_weight=None):
        X, y = self._before_fit(X, y)
        y_bin = self.restructure_y_to_bin(y)

        model = SGDClassifier(loss="log_loss",
                              random_state=self.random_state,
                              penalty="l1",
                              alpha=self.regularization,
                              fit_intercept=False,
                              n_jobs=1)

        # thresholds matrix 
        thresholds = np.identity(self.k)
        n = X.shape[0]

        cur_iter = 0

        while cur_iter < self.max_iter:
            #if (cur_iter > 0 and cur_iter % 2 == 0):
            #    print("Iter: ", cur_iter, "Train score: ", model.score(X_batch, y_batch))
            
            cur_iter += 1
            
            sampled_indices = self.rng.integers(len(y_bin), size=len(y_bin))

            start = 0
            for i in range(1, self.n_batches+1):
                end = (i * len(y_bin) // self.n_batches)
                idx = sampled_indices[start:end]
                idx_mod_n = idx % n
                X_batch = np.concatenate((X[idx_mod_n,:], thresholds[idx // n]), axis=1)
                y_batch = y_bin[idx]
                start = end
                weights = np.array(sample_weight)[idx_mod_n] if sample_weight is not None else None
                model.partial_fit(X_batch, y_batch, classes=np.unique(y_batch), sample_weight=weights)

        self._after_fit(model)
        return self
