"""
src/evaluation/__init__.py
Exposes all evaluation utilities for easy importing.

Usage
-----
from src.evaluation import mean_squared_error, r2_score
from src.evaluation import k_fold_cross_validation_classifier
from src.evaluation import grid_search_classifier, grid_search_regressor
"""

from src.evaluation.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    all_classification_metrics,
    all_regression_metrics,
)
from src.evaluation.cross_validation import (
    k_fold_cross_validation_classifier,
    k_fold_cross_validation_regressor,
)
from src.evaluation.grid_search import (
    grid_search_classifier,
    grid_search_regressor,
    detailed_grid_search_classifier,
    detailed_grid_search_regressor,
)

__all__ = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r2_score",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "all_classification_metrics",
    "all_regression_metrics",
    "k_fold_cross_validation_classifier",
    "k_fold_cross_validation_regressor",
    "grid_search_classifier",
    "grid_search_regressor",
    "detailed_grid_search_classifier",
    "detailed_grid_search_regressor",
]
