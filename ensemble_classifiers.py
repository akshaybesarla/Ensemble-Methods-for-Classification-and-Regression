"""
ensemble_classifiers.py
-----------------------
Ensemble classifiers built from scratch using NumPy.
All models use DecisionTree as the base learner.

Classes
-------
BaggingClassifier       — Bootstrap Aggregating
RandomForestClassifier  — Bagging + random feature subsets
GradientBoostingClassifier — Sequential residual correction
"""

import numpy as np
from src.models.decision_tree import DecisionTree


class BaggingClassifier:
    """
    Bagging Classifier (Bootstrap Aggregating).

    Trains multiple Decision Trees on independent bootstrap samples of the
    training data. Predictions are made by majority vote across all trees.

    Parameters
    ----------
    base_model   : DecisionTree instance
    n_estimators : int — number of bootstrap trees (default 10)
    """

    def __init__(self, base_model=None, n_estimators=10):
        self.base_model   = base_model if base_model else DecisionTree()
        self.n_estimators = n_estimators
        self.models       = []

    def fit(self, X, y):
        """Fit n_estimators trees on bootstrap samples of (X, y)."""
        self.models = []
        num_samples = X.shape[0]
        for _ in range(self.n_estimators):
            indices    = np.random.choice(num_samples, num_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]
            model = DecisionTree(max_depth=self.base_model.max_depth)
            model.fit(X_boot, y_boot)
            self.models.append(model)

    def predict(self, X):
        """Predict class labels via majority vote across all trees."""
        predictions = np.array([model.predict(X) for model in self.models])
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )


class RandomForestClassifier:
    """
    Random Forest Classifier.

    Extends Bagging by randomly selecting a subset of features at each split,
    reducing correlation between trees and improving generalisation.

    Parameters
    ----------
    n_estimators : int  — number of trees (default 10)
    max_features : int  — features considered at each split (default: sqrt(p))
    """

    def __init__(self, n_estimators=10, max_features=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees        = []

    def fit(self, X, y):
        """Fit n_estimators trees with random feature subsets."""
        self.trees = []
        num_samples, num_features = X.shape

        # Default: sqrt(p) features per split — standard for classification
        if self.max_features is None:
            self.max_features = int(np.sqrt(num_features))
        else:
            self.max_features = min(self.max_features, num_features)

        for _ in range(self.n_estimators):
            features   = np.random.choice(num_features, self.max_features, replace=False)
            X_boot, y_boot = self._bootstrap(X[:, features], y)
            tree = DecisionTree(max_depth=None)
            tree.fit(X_boot, y_boot)
            self.trees.append((tree, features))

    def _bootstrap(self, X, y):
        indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        """Predict class labels via majority vote across all trees."""
        predictions = np.array([
            tree.predict(X[:, features]) for tree, features in self.trees
        ])
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )


class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier.

    Builds Decision Trees sequentially. Each tree is fitted to the residual
    errors of all previous trees, progressively reducing bias.

    Parameters
    ----------
    n_estimators  : int   — number of boosting stages (default 100)
    learning_rate : float — shrinkage applied to each tree's contribution (default 0.1)
    max_depth     : int   — depth of each individual tree (default 3)
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.models        = []

    def fit(self, X, y):
        """Fit sequential trees on residuals of the current ensemble."""
        self.models = []
        residuals   = y.astype(float)
        for _ in range(self.n_estimators):
            model = DecisionTree(max_depth=self.max_depth)
            model.fit(X, residuals)
            predictions = model.predict(X)
            residuals  -= self.learning_rate * predictions
            self.models.append(model)

    def predict(self, X):
        """Predict class labels as rounded sum of all trees' contributions."""
        predictions = np.zeros(X.shape[0], dtype=float)
        for model in self.models:
            predictions += self.learning_rate * model.predict(X)
        return np.round(predictions).astype(int)
