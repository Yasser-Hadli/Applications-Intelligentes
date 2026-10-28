
"""Modèles de Machine Learning à comparer."""

from monprojet import config

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)


def candidates() -> dict:

    # Cas 1 : problème de classification
    if config.TASK == "classification":

        return {
            "baseline": DummyClassifier(
                strategy="most_frequent"
            ),

            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=config.SEED
            ),

            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=config.SEED
            ),

            "gradient_boosting": GradientBoostingClassifier(
                random_state=config.SEED
            )
        }

    # Cas 2 : problème de régression
    elif config.TASK == "regression":

        return {
            "baseline": DummyRegressor(
                strategy="mean"
            ),

            "ridge": Ridge(
                alpha=1.0
            ),

            "random_forest": RandomForestRegressor(
                n_estimators=100,
                random_state=config.SEED
            ),

            "gradient_boosting": GradientBoostingRegressor(
                random_state=config.SEED
            )
        }

    else:
        raise ValueError("Type de problème inconnu")
