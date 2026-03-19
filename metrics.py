"""
metrics.py
----------
Custom evaluation metric functions used across both classification
and regression experiments. All implemented from scratch in NumPy
to mirror the logic used inside the Jupyter notebooks.
"""

import numpy as np


# ── Regression Metrics ────────────────────────────────────────────────────────

def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error.
    MSE = (1/n) * Σ(yᵢ - ŷᵢ)²

    Parameters
    ----------
    y_true : array-like — ground truth values
    y_pred : array-like — predicted values

    Returns
    -------
    float
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error.
    MAE = (1/n) * Σ|yᵢ - ŷᵢ|

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    float
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    """
    Root Mean Squared Error.
    RMSE = sqrt(MSE)

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    float
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2_score(y_true, y_pred):
    """
    R² (Coefficient of Determination).
    R² = 1 - SS_res / SS_tot

    A value of 1.0 means perfect prediction. 0.0 means the model
    performs no better than predicting the mean. Negative values
    indicate the model is worse than a constant mean predictor.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    float
    """
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    ss_res  = np.sum((y_true - y_pred) ** 2)
    ss_tot  = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


# ── Classification Metrics ────────────────────────────────────────────────────

def accuracy_score(y_true, y_pred):
    """
    Classification accuracy.
    Accuracy = correct predictions / total predictions

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    float
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average='macro'):
    """
    Precision score for multi-class classification.
    Precision = TP / (TP + FP)

    Parameters
    ----------
    y_true  : array-like
    y_pred  : array-like
    average : str — 'macro' (default) averages equally across classes

    Returns
    -------
    float
    """
    y_true   = np.array(y_true)
    y_pred   = np.array(y_pred)
    classes  = np.unique(y_true)
    precisions = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return np.mean(precisions)


def recall_score(y_true, y_pred, average='macro'):
    """
    Recall score for multi-class classification.
    Recall = TP / (TP + FN)

    Parameters
    ----------
    y_true  : array-like
    y_pred  : array-like
    average : str — 'macro' (default)

    Returns
    -------
    float
    """
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return np.mean(recalls)


def f1_score(y_true, y_pred, average='macro'):
    """
    F1 Score — harmonic mean of precision and recall.
    F1 = 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    y_true  : array-like
    y_pred  : array-like
    average : str — 'macro' (default)

    Returns
    -------
    float
    """
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def all_classification_metrics(y_true, y_pred):
    """
    Returns a dict of all classification metrics in one call.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1
    """
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall':    recall_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred),
    }


def all_regression_metrics(y_true, y_pred):
    """
    Returns a dict of all regression metrics in one call.

    Returns
    -------
    dict with keys: mse, mae, rmse, r2
    """
    return {
        'mse':  mean_squared_error(y_true, y_pred),
        'mae':  mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2':   r2_score(y_true, y_pred),
    }
