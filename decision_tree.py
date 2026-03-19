"""
decision_tree.py
----------------
Custom Decision Tree implementation from scratch using NumPy.
Supports both classification (Gini impurity) and regression (MSE / variance reduction).
Used as the base learner for all ensemble methods in this project.
"""

import numpy as np


class DecisionTree:
    """
    Decision Tree Classifier.
    Uses Gini impurity to select the best split at each node (CART algorithm).

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree. None means nodes expand until all leaves are pure.
    """

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """Build the decision tree from training data."""
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # Stopping criteria: pure node or max depth reached
        if len(unique_classes) == 1 or (self.max_depth and depth == self.max_depth):
            return unique_classes[0]

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.bincount(y).argmax()

        left_indices  = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] >  best_threshold

        left_subtree  = self._build_tree(X[left_indices],  y[left_indices],  depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        """Find the feature and threshold that minimises weighted Gini impurity."""
        num_samples, num_features = X.shape
        best_feature, best_threshold = None, None
        best_gini = float('inf')

        for feature in range(num_features):
            for threshold in np.unique(X[:, feature]):
                left_idx  = X[:, feature] <= threshold
                right_idx = X[:, feature] >  threshold

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                gini_left  = 1.0 - sum(
                    (np.sum(y[left_idx]  == c) / len(y[left_idx]))  ** 2
                    for c in np.unique(y)
                )
                gini_right = 1.0 - sum(
                    (np.sum(y[right_idx] == c) / len(y[right_idx])) ** 2
                    for c in np.unique(y)
                )
                gini = (
                    (len(y[left_idx])  / len(y)) * gini_left +
                    (len(y[right_idx]) / len(y)) * gini_right
                )

                if gini < best_gini:
                    best_gini      = gini
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        """Predict class labels for samples in X."""
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left_subtree, right_subtree = tree
        if x[feature] <= threshold:
            return self._predict_sample(x, left_subtree)
        else:
            return self._predict_sample(x, right_subtree)


class DecisionTreeRegressor:
    """
    Decision Tree Regressor.
    Uses weighted variance reduction (MSE-based splitting) to build the tree.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree.
    """

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree_ = None

    def fit(self, X, y):
        """Build the regression tree from training data."""
        self.tree_ = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        # Stopping criteria
        if self.max_depth is not None and depth >= self.max_depth:
            return np.mean(y)
        if len(np.unique(y)) == 1:
            return y[0]

        n_samples, n_features = X.shape
        best_split = None
        best_mse   = float('inf')

        for feature_index in range(n_features):
            for threshold in np.unique(X[:, feature_index]):
                left_mask  = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                left_y, right_y = y[left_mask], y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                # Weighted variance — the splitting criterion
                mse = len(left_y) * np.var(left_y) + len(right_y) * np.var(right_y)

                if mse < best_mse:
                    best_mse   = mse
                    best_split = (feature_index, threshold)

        if best_split is None:
            return np.mean(y)

        feature_index, threshold = best_split
        left_mask  = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        return {
            'feature_index': feature_index,
            'threshold':     threshold,
            'left':          self._build_tree(X[left_mask],  y[left_mask],  depth + 1),
            'right':         self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def predict(self, X):
        """Predict continuous target values for samples in X."""
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def _predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature_index']] <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])
