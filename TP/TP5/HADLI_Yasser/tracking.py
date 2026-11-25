"""Enregistrer et comparer les entraînements de MaintAI avec MLflow.

Chaque appel à :func:`log_run` crée un run contenant ses paramètres, ses
métriques, le modèle et l'empreinte du fichier de données. Par défaut, MLflow
utilise sa configuration locale ; le projet emploie normalement ``mlflow.db``
et ``mlruns/`` dans le dossier courant. La variable ``MLFLOW_TRACKING_URI``
permet de choisir un autre stockage, notamment pour les tests.

Pour consulter les résultats depuis le dossier du projet : ``mlflow ui``.
"""
import hashlib
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # coupe un message d'information de MLflow

import mlflow
import mlflow.sklearn

EXPERIMENT = "maintai"
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]  # nécessaire pour --model forest


def file_md5(path: Path) -> str:
    """Calculer l'empreinte MD5 du contenu brut d'un fichier.

    Args:
        path: Chemin du fichier de données utilisé pour l'entraînement.

    Returns:
        Empreinte hexadécimale, comparable au champ ``md5`` du fichier DVC
        pour ce fichier CSV. MD5 sert ici à identifier la version des données,
        pas à garantir leur authenticité.

    Raises:
        OSError: Si le fichier ne peut pas être ouvert ou lu.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        # Lire par blocs de 1 Mio évite de charger tout le fichier en mémoire.
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def log_run(params: dict, metrics: dict, model, X_example, data_path: Path) -> str:
    """Enregistrer un entraînement dans l'expérience MLflow ``maintai``.

    Args:
        params: Paramètres du run, par exemple le type de modèle, le seuil
            d'alerte et les hyperparamètres. Le champ ``data_md5`` est ajouté
            à partir de ``data_path`` et remplace toute valeur déjà fournie.
        metrics: Métriques numériques calculées lors de l'évaluation, comme
            le rappel, la précision et l'accuracy.
        model: Modèle ou pipeline scikit-learn entraîné à enregistrer.
        X_example: Exemple de données d'entrée utilisé par MLflow pour
            documenter et inférer la signature du modèle.
        data_path: Chemin du fichier utilisé pour calculer ``data_md5``.

    Returns:
        Identifiant du run créé, utilisable pour retrouver ses résultats.

    Le modèle est enregistré sous le nom ``model`` avec son environnement de
    dépendances. Les erreurs de lecture ou d'enregistrement sont propagées à
    l'appelant ; le gestionnaire de contexte MLflow termine alors le run.
    """
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params({**params, "data_md5": file_md5(data_path)})
        mlflow.log_metrics(metrics)
        # MLflow conserve le modèle, son exemple d'entrée et ses dépendances.
        mlflow.sklearn.log_model(
            model, name="model", input_example=X_example,
            skops_trusted_types=TRUSTED_TYPES,
        )
        return run.info.run_id
