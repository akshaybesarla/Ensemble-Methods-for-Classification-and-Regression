"""
src/models/__init__.py
Exposes all model classes for easy importing.

Usage
-----
from src.models import DecisionTree, DecisionTreeRegressor
from src.models import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from src.models import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
"""

from src.models.decision_tree import DecisionTree, DecisionTreeRegressor
from src.models.ensemble_classifiers import (
    BaggingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from src.models.ensemble_regressors import (
    BaggingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

__all__ = [
    "DecisionTree",
    "DecisionTreeRegressor",
    "BaggingClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "BaggingRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
]
