"""
grid_search.py
--------------
Grid Search with K-Fold Cross-Validation for hyperparameter tuning.
Mirrors the logic used in the Jupyter notebooks for all four models.
"""

import numpy as np
from itertools import product
from src.evaluation.cross_validation import (
    k_fold_cross_validation_classifier,
    k_fold_cross_validation_regressor,
)


def grid_search_classifier(model_class, param_grid, X, y, k=4, **kwargs):
    """
    Exhaustive Grid Search for classification models.

    Iterates over every combination of parameters in param_grid,
    evaluates each using K-Fold CV, and returns the best parameters.

    Parameters
    ----------
    model_class : class — the model class to instantiate (e.g. DecisionTree)
    param_grid  : dict  — {param_name: [values]} e.g. {'max_depth': [2,3,4,5]}
    X           : np.ndarray
    y           : np.ndarray
    k           : int  — number of CV folds (default 4)
    **kwargs    : additional fixed keyword arguments passed to model_class

    Returns
    -------
    best_params   : dict  — parameter combination with highest validation accuracy
    best_accuracy : float — corresponding validation accuracy

    Example
    -------
    >>> best_params, best_acc = grid_search_classifier(
    ...     DecisionTree,
    ...     {'max_depth': [2, 3, 4, 5, 6]},
    ...     X_train, y_train
    ... )
    """
    best_params   = None
    best_accuracy = 0.0

    keys, values = zip(*param_grid.items())
    for combo in product(*values):
        params      = dict(zip(keys, combo))
        model_args  = {**kwargs, **params}
        model       = model_class(**model_args)
        _, val_acc  = k_fold_cross_validation_classifier(model, X, y, k)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_params   = params

    return best_params, best_accuracy


def detailed_grid_search_classifier(model_class, param_grid, X, y, k=4, **kwargs):
    """
    Detailed Grid Search — returns all results, not just the best.

    Parameters
    ----------
    Same as grid_search_classifier.

    Returns
    -------
    results : list of (params_dict, train_accuracy, val_accuracy)
    """
    results = []
    keys, values = zip(*param_grid.items())

    for combo in product(*values):
        params     = dict(zip(keys, combo))
        model_args = {**kwargs, **params}
        model      = model_class(**model_args)
        train_acc, val_acc = k_fold_cross_validation_classifier(model, X, y, k)
        results.append((params, train_acc, val_acc))
        print(f"Params: {params}  |  Train: {train_acc:.4f}  |  Val: {val_acc:.4f}")

    return results


def grid_search_regressor(model_class, param_grid, X, y, k=4, **kwargs):
    """
    Exhaustive Grid Search for regression models.

    Selects the parameter combination with the lowest validation MSE.

    Parameters
    ----------
    model_class : class — the model class to instantiate
    param_grid  : dict  — {param_name: [values]}
    X           : np.ndarray
    y           : np.ndarray
    k           : int  — number of CV folds (default 4)
    **kwargs    : additional fixed keyword arguments

    Returns
    -------
    best_params : dict  — parameter combination with lowest validation MSE
    best_mse    : float — corresponding validation MSE

    Example
    -------
    >>> best_params, best_mse = grid_search_regressor(
    ...     BaggingRegressor,
    ...     {'n_estimators': [10, 15, 20, 25, 30]},
    ...     X_train, y_train
    ... )
    """
    best_params = None
    best_mse    = float('inf')

    keys, values = zip(*param_grid.items())
    for combo in product(*values):
        params     = dict(zip(keys, combo))
        model_args = {**kwargs, **params}
        model      = model_class(**model_args)
        _, _, val_mse, _ = k_fold_cross_validation_regressor(model, X, y, k)

        if val_mse < best_mse:
            best_mse    = val_mse
            best_params = params

    return best_params, best_mse


def detailed_grid_search_regressor(model_class, param_grid, X, y, k=4, **kwargs):
    """
    Detailed Grid Search for regressors — returns all results.

    Returns
    -------
    results : list of (params_dict, train_mse, train_mae, val_mse, val_mae)
    """
    results = []
    keys, values = zip(*param_grid.items())

    for combo in product(*values):
        params     = dict(zip(keys, combo))
        model_args = {**kwargs, **params}
        model      = model_class(**model_args)
        tr_mse, tr_mae, v_mse, v_mae = k_fold_cross_validation_regressor(model, X, y, k)
        results.append((params, tr_mse, tr_mae, v_mse, v_mae))
        print(
            f"Params: {params}  |  "
            f"Train MSE: {tr_mse:.2f}  Train MAE: {tr_mae:.2f}  |  "
            f"Val MSE: {v_mse:.2f}  Val MAE: {v_mae:.2f}"
        )

    return results
