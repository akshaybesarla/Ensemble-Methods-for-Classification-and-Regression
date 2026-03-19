"""
cross_validation.py
-------------------
K-Fold Cross-Validation utilities used in both classification and regression
experiments. Mirrors the logic used inside the Jupyter notebooks.
"""

import numpy as np
from sklearn.model_selection import KFold


def k_fold_cross_validation_classifier(model, X, y, k=4):
    """
    K-Fold Cross-Validation for classification models.

    Splits data into k folds. On each fold, trains on k-1 folds
    and evaluates on the remaining fold. Returns mean accuracy
    across folds for both training and validation sets.

    Parameters
    ----------
    model : classifier with .fit(X, y) and .predict(X)
    X     : np.ndarray — feature matrix
    y     : np.ndarray — target labels
    k     : int        — number of folds (default 4)

    Returns
    -------
    (mean_train_accuracy, mean_val_accuracy) : tuple of floats
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_accuracies = []
    val_accuracies   = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)

        train_acc = np.mean(model.predict(X_train) == y_train)
        val_acc   = np.mean(model.predict(X_val)   == y_val)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    return np.mean(train_accuracies), np.mean(val_accuracies)


def k_fold_cross_validation_regressor(model, X, y, k_folds=4):
    """
    K-Fold Cross-Validation for regression models.

    Returns mean MSE and MAE across folds for both train and validation sets.

    Parameters
    ----------
    model   : regressor with .fit(X, y) and .predict(X)
    X       : np.ndarray
    y       : np.ndarray
    k_folds : int — number of folds (default 4)

    Returns
    -------
    (mean_train_mse, mean_train_mae, mean_val_mse, mean_val_mae) : tuple of floats
    """
    kf = KFold(n_splits=k_folds)
    mse_train_scores, mae_train_scores = [], []
    mse_val_scores,   mae_val_scores   = [], []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred   = model.predict(X_val)

        mse_train_scores.append(np.mean((y_train - y_train_pred) ** 2))
        mae_train_scores.append(np.mean(np.abs(y_train - y_train_pred)))
        mse_val_scores.append(np.mean((y_val - y_val_pred) ** 2))
        mae_val_scores.append(np.mean(np.abs(y_val - y_val_pred)))

    return (
        np.mean(mse_train_scores),
        np.mean(mae_train_scores),
        np.mean(mse_val_scores),
        np.mean(mae_val_scores),
    )
