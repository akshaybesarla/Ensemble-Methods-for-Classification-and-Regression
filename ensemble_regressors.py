"""
ensemble_regressors.py
----------------------
Ensemble regressors built from scratch using NumPy.
All models use DecisionTreeRegressor as the base learner.

Classes
-------
BaggingRegressor           — Bootstrap Aggregating
RandomForestRegressor      — Bagging + random feature subsets
GradientBoostingRegressor  — Sequential residual correction
"""

import numpy as np
from src.models.decision_tree import DecisionTreeRegressor


class BaggingRegressor:
    """
    Bagging Regressor (Bootstrap Aggregating).

    Trains multiple DecisionTreeRegressors on independent bootstrap samples.
    Predictions are the mean of all trees' outputs.

    Parameters
    ----------
    base_estimator : DecisionTreeRegressor instance
    n_estimators   : int — number of bootstrap trees (default 10)
    """

    def __init__(self, base_estimator=None, n_estimators=10):
        self.base_estimator = base_estimator if base_estimator else DecisionTreeRegressor(max_depth=5)
        self.n_estimators   = n_estimators
        self.models         = []

    def fit(self, X, y):
        """Fit n_estimators trees on bootstrap samples of (X, y)."""
        self.models = []
        for _ in range(self.n_estimators):
            indices  = np.random.choice(len(X), len(X), replace=True)
            X_boot, y_boot = X[indices], y[indices]
            model = DecisionTreeRegressor(max_depth=self.base_estimator.max_depth)
            model.fit(X_boot, y_boot)
            self.models.append(model)

    def predict(self, X):
        """Predict by averaging all trees' predictions."""
        predictions = np.zeros((self.n_estimators, len(X)))
        for i, model in enumerate(self.models):
            predictions[i] = model.predict(X)
        return np.mean(predictions, axis=0)


class RandomForestRegressor:
    """
    Random Forest Regressor.

    Extends Bagging by selecting a random subset of features at each split.
    Standard feature count: p/3 for regression.

    Parameters
    ----------
    n_estimators : int       — number of trees (default 10)
    max_depth    : int/None  — depth of each tree
    max_features : int       — features considered per split (default: p//3)
    """

    def __init__(self, n_estimators=10, max_depth=None, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.max_features = max_features
        self.models       = []

    def fit(self, X, y):
        """Fit n_estimators trees with random feature subsets."""
        self.models = []
        n_features = X.shape[1]

        # Default: p/3 features per split — standard for regression
        if self.max_features is None:
            self.max_features = max(1, n_features // 3)

        for _ in range(self.n_estimators):
            indices  = np.random.choice(len(X), len(X), replace=True)
            features = np.random.choice(n_features, self.max_features, replace=False)
            X_boot   = X[indices][:, features]
            y_boot   = y[indices]
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X_boot, y_boot)
            self.models.append((model, features))

    def predict(self, X):
        """Predict by averaging all trees' predictions."""
        predictions = np.zeros((self.n_estimators, len(X)))
        for i, (model, features) in enumerate(self.models):
            predictions[i] = model.predict(X[:, features])
        return np.mean(predictions, axis=0)


class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor.

    Builds DecisionTreeRegressors sequentially. Each tree is fitted to the
    residual errors of all previous trees, reducing bias iteratively.

    Parameters
    ----------
    n_estimators  : int   — number of boosting stages (default 100)
    learning_rate : float — shrinkage per tree contribution (default 0.1)
    max_depth     : int   — depth of each individual tree (default 3)
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.models        = []
        self.loss          = []

    def fit(self, X, y):
        """Fit sequential trees to the residuals of the current ensemble."""
        self.models = []
        residual    = y.copy()
        for _ in range(self.n_estimators):
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, residual)
            self.models.append(model)
            prediction = model.predict(X)
            residual  -= self.learning_rate * prediction

    def predict(self, X):
        """Predict as the sum of all trees' contributions scaled by learning rate."""
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += self.learning_rate * model.predict(X)
        return predictions
