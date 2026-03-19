"""
preprocessing.py
----------------
Data preprocessing utilities for the ensemble methods project.
Handles feature scaling, train/validation/test splitting, and
feature selection as described in the report methodology.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier as SklearnDT


def split_data(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split dataset into train, validation, and test sets (70/15/15 default).

    Parameters
    ----------
    X            : np.ndarray — feature matrix
    y            : np.ndarray — target vector
    val_size     : float — proportion for validation set (default 0.15)
    test_size    : float — proportion for test set (default 0.15)
    random_state : int

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # Then split the remainder into train and validation
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val=None, X_test=None):
    """
    Standardise features using StandardScaler (zero mean, unit variance).
    Fitted only on the training set to prevent data leakage.

    Parameters
    ----------
    X_train : np.ndarray
    X_val   : np.ndarray or None
    X_test  : np.ndarray or None

    Returns
    -------
    Scaled arrays in the same order as inputs. Scaler also returned.
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    results = [X_train]
    if X_val  is not None: results.append(scaler.transform(X_val))
    if X_test is not None: results.append(scaler.transform(X_test))
    results.append(scaler)
    return tuple(results)


def select_features_rfe(X_train, y_train, n_features=None):
    """
    Recursive Feature Elimination (RFE) using a Decision Tree estimator.
    Selects the top n_features most predictive features.

    Parameters
    ----------
    X_train    : np.ndarray
    y_train    : np.ndarray
    n_features : int or None — number of features to select (default: half)

    Returns
    -------
    selected_indices : np.ndarray — indices of selected features
    rfe              : fitted RFE object
    """
    if n_features is None:
        n_features = max(1, X_train.shape[1] // 2)

    estimator = SklearnDT()
    rfe       = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    selected_indices = np.where(rfe.support_)[0]
    return selected_indices, rfe
