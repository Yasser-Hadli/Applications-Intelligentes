"""Chargement et découpage des données.

À vous d'écrire load_data() et split(). Les tests de tests/test_data.py disent ce qui est attendu.
"""
import pandas as pd
from sklearn.model_selection import train_test_split  # noqa: F401  (utile pour split)

from monprojet import config



def load_data(path=None) -> pd.DataFrame:
    # 1. Lire les données depuis le CSV
    df = pd.read_csv(
        path if path is not None else config.DATA_PATH
    )

    # 2. Vérifier que la colonne cible existe
    if config.TARGET not in df.columns:
        raise ValueError(
            f"Colonne cible absente : {config.TARGET}"
        )

    # 3. Supprimer les lignes sans valeur cible
    df = df.dropna(subset=[config.TARGET])

    # 4. Renvoyer le tableau
    return df

def feature_columns(df: pd.DataFrame) -> list:
    """Les colonnes d'entrée : celles de config.FEATURES, ou toutes sauf la cible et la date. (Fournie.)"""
    if config.FEATURES is not None:
        return list(config.FEATURES)
    excluded = {config.TARGET, config.TIME_COLUMN}
    return [c for c in df.columns if c not in excluded]



def split(df: pd.DataFrame):
    # 1. Séparer les variables explicatives et la cible
    X = df[feature_columns(df)]
    y = df[config.TARGET]

    # 2. Découpage aléatoire
    if config.SPLIT == "random":
        return train_test_split(
            X,
            y,
            test_size=config.TEST_SIZE,
            random_state=config.SEED,
            stratify=y if config.TASK == "classification" else None
        )

    # 3. Découpage chronologique
    elif config.SPLIT == "chronologique":
        if not config.TIME_COLUMN:
            raise ValueError("TIME_COLUMN doit être définie")

        ordre = pd.to_datetime(
            df[config.TIME_COLUMN]
        ).argsort()

        X = X.iloc[ordre]
        y = y.iloc[ordre]

        n_test = max(1, int(round(len(df) * config.TEST_SIZE)))

        X_train = X.iloc[:-n_test]
        X_test = X.iloc[-n_test:]
        y_train = y.iloc[:-n_test]
        y_test = y.iloc[-n_test:]

        return X_train, X_test, y_train, y_test

    else:
        raise ValueError("Méthode de découpage inconnue")


