from abc import ABC, ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
import numpy as np
from numpy.random import default_rng


class BaseModel(ClassifierMixin, BaseEstimator, ABC):
    random_state: int = 1234
    beta: np.array
    theta: np.array
    k: int = 0
    labels: np.array

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
        self.labels = np.unique(targets)
        return data, targets

    def _restructure_X_y(self, data, targets):
        # convert to binary problem
        X_bin = self.restructure_X_to_bin(data, n_thresholds=self.k)
        y_bin = self.restructure_y_to_bin(targets)

        return X_bin, y_bin
    
    def _after_fit(self, model):
        # extract the thresholds and weights
        # from the 2D coefficients matrix in the sklearn model
        self.theta = model.coef_[0, -self.k:][::-1]  # thresholds
        self.beta = model.coef_[0, :-self.k]   # weights

        self.is_fitted_ = True

    @abstractmethod
    def _get_fitted_model(self, data_bin, targets_bin):
        return LogisticRegression()

    def fit(self, data, targets, restructured_inputs=False):
        self.k = np.unique(targets).size - 1

        data, targets = self._before_fit(data, targets)
        
        # restructure the input data as a binary problem
        if not restructured_inputs:
            data, targets = self._restructure_X_y(data, targets)

        model = self._get_fitted_model(data, targets)
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

    def _get_fitted_model(self, X_bin, y_bin):

        model = LogisticRegression(penalty="l1", 
                                  fit_intercept=False,
                                  max_iter=self.max_iter,
                                  solver=self.solver,
                                  random_state=self.random_state,
                                  C=self.regularization  # Inverse of regularization strength -> controls sparsity in our case!
                                )

        model.fit(X_bin, y_bin)

        return model


class SGDBinarizedModel(BinaryModelMixin, BaseModel):
    def __init__(self, max_iter=5, n_batches=2, random_state=1234, regularization=0.01):
        
        # model hyperparameters
        self.max_iter = max_iter
        self.n_batches = n_batches
        self.random_state = random_state
        self.regularization = regularization
        self.rng = default_rng(seed=self.random_state)

        # fitting/data parameters
        self.k = None
        self.theta = []
        self.beta = []

    def _restructure_X_y(self, data, targets):
        # overwrites the superclass method to 
        # only convert the target vector to binary problem
        y_bin = self.restructure_y_to_bin(targets)

        return data, y_bin
    
    def _get_fitted_model(self, X, y_bin):

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
            if (cur_iter > 0 and cur_iter % 2 == 0):
                print("Iter: ", cur_iter, "Train score: ", model.score(X_batch, y_batch))
            
            cur_iter += 1
            
            sampled_indices = self.rng.integers(len(y_bin), size=len(y_bin))

            start = 0
            for i in range(1, self.n_batches+1):
                end = (i * len(y_bin) // self.n_batches)
                idx = sampled_indices[start:end]
                X_batch = np.concatenate((X[idx % n,:], thresholds[idx // n]), axis=1)
                y_batch = y_bin[idx]
                start = end
                model.partial_fit(X_batch, y_batch, classes=np.unique(y_batch))

        return model
