"""Entraîne chaque modèle candidat, l'évalue sur le jeu de test et les compare.

Usage, depuis la racine du projet : python -m monprojet.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from monprojet import config
from monprojet.data import load_data, split
from monprojet.models import candidates

HIGHER_IS_BETTER = {"accuracy": True, "recall": True, "precision": True, "f1": True,
                    "mae": False, "rmse": False, "r2": True}


def build_preprocessing(X: pd.DataFrame) -> ColumnTransformer:
    """Prépare les colonnes : nombres mis à l'échelle, catégories encodées, texte vectorisé. (Fournie.)"""
    text = [config.TEXT_COLUMN] if config.TEXT_COLUMN else []
    numeric = [c for c in X.columns if c not in text and pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in text and c not in numeric]
    steps = []
    if numeric:
        steps.append(("nombres", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric))
    if categorical:
        steps.append(("categories", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                  OneHotEncoder(handle_unknown="ignore")), categorical))
    if text:
        steps.append(("texte", TfidfVectorizer(), config.TEXT_COLUMN))
    return ColumnTransformer(steps, sparse_threshold=0)


def evaluate(model, X_test, y_test) -> dict:
   
    # 1. Faire les prédictions
    y_pred = model.predict(X_test)

    # 2. Cas d'une classification
    if config.TASK == "classification":

        # Binaire si deux classes, macro sinon
        average = (
            "binary" if len(np.unique(y_test)) == 2
            else "macro"
        )

        return {
            "accuracy": round(float(
                accuracy_score(y_test, y_pred)
            ), 3),

            "recall": round(float(
                recall_score(
                    y_test, y_pred,
                    average=average,
                    zero_division=0
                )
            ), 3),

            "precision": round(float(
                precision_score(
                    y_test, y_pred,
                    average=average,
                    zero_division=0
                )
            ), 3),

            "f1": round(float(
                f1_score(
                    y_test, y_pred,
                    average=average,
                    zero_division=0
                )
            ), 3)
        }

    # 3. Cas d'une régression
    elif config.TASK == "regression":

        return {
            "mae": round(float(
                mean_absolute_error(y_test, y_pred)
            ), 3),

            "rmse": round(float(
                np.sqrt(
                    mean_squared_error(y_test, y_pred)
                )
            ), 3),

            "r2": round(float(
                r2_score(y_test, y_pred)
            ), 3)
        }

    else:
        raise ValueError("Type de problème inconnu")



def main() -> None:

    # 1. Charger les données
    df = load_data()

    # 2. Séparer entraînement et test
    X_train, X_test, y_train, y_test = split(df)

    # 3. Préparer le stockage des résultats
    resultats = {}
    modeles_entraines = {}

    # 4. Entraîner chaque modèle
    for nom, estimateur in candidates().items():

        print(f"Entraînement du modèle : {nom}")

        modele = Pipeline([
            ("preparation", build_preprocessing(X_train)),
            ("modele", estimateur)
        ])

        modele.fit(X_train, y_train)

        # Évaluer les performances
        scores = evaluate(modele, X_test, y_test)

        resultats[nom] = scores
        modeles_entraines[nom] = modele

    # 5. Comparer les modèles
    metrique = config.MAIN_METRIC

    if metrique not in HIGHER_IS_BETTER:
        raise ValueError(f"Métrique inconnue : {metrique}")

    ordre_decroissant = HIGHER_IS_BETTER[metrique]

    tableau = pd.DataFrame(resultats).T

    tableau = tableau.sort_values(
        by=metrique,
        ascending=not ordre_decroissant
    )

    print("\nComparaison des modèles :")
    print(tableau.to_string())

    # 6. Sélectionner le meilleur modèle
    meilleur_nom = tableau.index[0]

    meilleur_modele = modeles_entraines[meilleur_nom]

    print(f"\nMeilleur modèle : {meilleur_nom}")

    # 7. Créer le dossier de sauvegarde
    config.MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 8. Sauvegarder le meilleur modèle
    joblib.dump(
        meilleur_modele,
        config.MODELS_DIR / "model.joblib"
    )

    # 9. Sauvegarder les métriques
    rapport = {
        "main_metric": metrique,
        "best": meilleur_nom,
        "results": resultats
    }

    with open(
        config.MODELS_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as fichier:
        json.dump(
            rapport,
            fichier,
            indent=2,
            ensure_ascii=False
        )

    print("\nModèle et métriques sauvegardés !")
