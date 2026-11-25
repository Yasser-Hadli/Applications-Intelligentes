# Compte rendu  TP5 : MLflow et DVC

**Membre :** Yasser Hadli  **Date :** 25 novembre 2026 18H00MIN

## 1. Mise en place

Les dépendances ont été installées avec `pip install -r requirements.txt`, puis le package avec `pip install -e .`. Aprés implémentation du suivi MLflow, la suite complète donne :

```text
32 passed, 4 warnings in 68.30s
```

Les avertissements restants concernent des dépendances et le schéma des colonnes entières MLflow.

Le chemin précis du package installé avant `pip install -e .` n'a pas été relevé. Après installation éditable, `maintai` pointe vers le code source du dossier local du TP5 ; les modifications du code sont ainsi prises en compte sans réinstallation.

## 2. Sprint 1 : MLflow

Fonction `log_run` :

```python
def log_run(params: dict, metrics: dict, model, X_example, data_path: Path) -> str:
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params({**params, "data_md5": file_md5(data_path)})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model, name="model", input_example=X_example,
            skops_trusted_types=TRUSTED_TYPES,
        )
        return run.info.run_id
```

Après le sprint : `32 passed, 4 warnings in 68.30s`.

Les quatre essais ci-dessous utilisent les mêmes données (5 000 relevés, dont 4 000 pour l'entraînement et 1 000 pour le test) :

| Commande | Recall | Precision | Accuracy |
|---|---:|---:|---:|
| `python -m maintai.train` | 0,750 | 0,622 | 0,904 |
| `python -m maintai.train --C 0.01` | 0,691 | 0,653 | 0,908 |
| `python -m maintai.train --threshold 0.2` | 0,838 | 0,543 | 0,882 |
| `python -m maintai.train --model forest` | 0,728 | 0,562 | 0,886 |

Je retiens provisoirement la régression logistique avec un seuil de 0,2 : son rappel de 83,8 % est le plus élevé des essais, ce qui réduit le nombre de pannes manquées. En contrepartie, sa précision de 54,3 % implique davantage de fausses alertes. Il faudrait valider ce compromis sur des données réelles et selon le coût des alertes.

La version de scikit-learn utilisée est **1.8.0**. MLflow enregistre les dépendances du modèle, ce qui aide à reproduire l'environnement et évite de charger le modèle avec une version différente sans le savoir, problème rencontré lors de la livraison au TP4.

La base locale est `mlflow.db` et les artefacts sont sous `mlruns/`. Ces fichiers sont ignorés par Git : ils sont générés, potentiellement volumineux et propres à l'environnement local.

## 3. Sprint 2 : DVC

Après `dvc add data/sensors.csv` :

```yaml
outs:
- md5: 73e0b7cb3c6f795c70c3ddf970b99c15
  size: 126815
  hash: md5
  path: sensors.csv
```

Cette empreinte est identique au paramètre `data_md5` des quatre runs. Elle permet d'identifier la version des données utilisée. Le CSV est stocké dans le cache DVC et a été envoyé avec `dvc push` vers `C:/dvc-stockage`. Le petit fichier `.dvc`, lui, décrit les données et peut être versionné dans Git sans y stocker le CSV.

La génération de la v2 avec `--n 8000 --seed 7` a provoqué `dvc status` : `changed outs: modified: data/sensors.csv`. Après `dvc add` et `dvc push`, le pointeur a été enregistré dans Git. Les deux commits ci-dessous permettent de revenir à chaque version avec le pointeur correspondant et `dvc checkout`.

| Version | Commit Git | MD5 (8 caractères) | Relevés |
|---|---|---|---:|
| v1 | `ca2e8ad` | `73e0b7cb` | 5 000 |
| v2 | `0ddeaf7` | `93fdeef2` | 8 000 |

La restauration avec DVC a été vérifiée pour les deux versions : les nombres de lignes et les empreintes correspondent. Pour vérifier `dvc pull` sans supprimer les données de travail, un clone local neuf, sans cache DVC ni CSV, a été configuré avec le même stockage : `dvc pull` a récupéré 8 000 lignes avec le MD5 `93fdeef2cac1893c4f720c16457eea75`.

## 4. Relier modèle et données

| Run | data_md5 | n_train | Recall | Precision |
|---|---|---:|---:|---:|
| v1, seuil 0,2 | `73e0b7cb` | 4 000 | 0,838 | 0,543 |
| v2, seuil 0,2 | `93fdeef2` | 6 400 | 0,824 | 0,567 |

Ces métriques MLflow portent sur des jeux de test différents. Pour une comparaison directe, les deux modèles enregistrés ont aussi été évalués au seuil de 0,2 sur un troisième jeu synthétique indépendant commun de 5 000 relevés, généré avec la graine 2026 :

| Modèle | Rappel | Précision | Accuracy | Fausses alertes | Pannes manquées |
|---|---:|---:|---:|---:|---:|
| v1 | 0,8003 | 0,5341 | 0,8912 | 423 | 121 |
| v2 | 0,7888 | 0,5501 | 0,8962 | 391 | 128 |

La v2 réduit les fausses alertes de 32 mais manque 7 pannes supplémentaires. Elle n'est donc pas meilleure sur tous les critères. Ce jeu reste synthétique et ne garantit pas les performances en situation réelle. Run v2 : `8c8eaa344a6c43f7b3689e6ef6eebc4a`.

### Coût des fausses alertes et choix du seuil

Sur ce jeu commun, pour la v2, le seuil 0,3 donne 239 fausses alertes et 172 pannes manquées, contre respectivement 391 et 128 au seuil 0,2. Le seuil 0,2 évite donc 44 pannes manquées au prix de 152 fausses alertes supplémentaires. Si l'on ne considère que ces deux coûts unitaires, il est moins coûteux lorsque le coût d'une panne manquée dépasse environ 3,45 fois celui d'une fausse alerte (152 / 44). Cette analyse est indicative : aucun coût réel d'intervention ou de panne n'est fourni et les données sont synthétiques. Le seuil reste provisoire en attendant les données métier.

## 5. Décision : ADR 004

Voir `docs/adr/004-choix-du-modele.md`. Le run retenu est `e63993f2e7dd4151a536dad6bcfdcd8a`, avec les données d'empreinte `73e0b7cb3c6f795c70c3ddf970b99c15`.

## 6. Questions de compréhension

1. **Git** suit le code, les documents et les pointeurs `.dvc`. **DVC** gère les versions des données et leur stockage externe ; les pointeurs sont enregistrés dans Git. **MLflow** enregistre les expériences : paramètres, métriques, modèle et empreinte des données.
2. `--local` conserve la configuration du stockage propre à cette machine. Pour partager les données, le groupe doit disposer d'un stockage commun accessible et de ses droits d'accès, puis configurer le remote DVC et utiliser `push`/`pull`.
3. Livrer un modèle nécessite le modèle entraîné, son contrat d'entrée et son environnement de dépendances. Ici, ils se trouvent respectivement dans `models/model.joblib`, `models/model_info.json` et les dépendances enregistrées par MLflow (ainsi que `requirements.txt` pour le projet).

## 7. Le rendu Git
- Le tag `tp5` existe dans le dépôt distant Applications-Intelligentes : oui, il a été récupéré lors de la mise à jour de la branche `TP`. Il ne prouve pas que ce compte rendu local y figure.
- `git check-ignore` confirme que `mlflow.db`, `mlruns/`, `data/sensors.csv` et `.dvc/cache/` sont ignorés. Des commits locaux de suivi DVC ont été créés ; aucun push n'a été effectué.

## 8. Ce qui reste à faire

Obtenir les coûts métier réels et valider les modèles sur des données réelles indépendantes avant de fixer définitivement le seuil.

## 9. Recherche d'hyperparamètres avec GridSearchCV

Cette expérience utilise la v2 des données (`data_md5 = 93fdeef2cac1893c4f720c16457eea75`). Un échantillon stratifié reproductible de **20 relevés**, dont **3 pannes**, est sélectionné avec la graine 42. La grille teste trois valeurs de `C` et deux réglages de `class_weight`, soit six combinaisons. La métrique optimisée est le rappel, avec le seuil de prédiction par défaut de la régression logistique, 0,5.

Code de l'expérience (exécutable avec `python -m maintai.gridsearch`) :

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from maintai.config import FEATURES, SEED, TARGET, get_settings
from maintai.data import load_data

data = load_data(get_settings().data_path)
sample, _ = train_test_split(
    data, train_size=20, stratify=data[TARGET], random_state=SEED
)
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
])
grid = {"clf__C": [0.01, 1.0, 100.0],
        "clf__class_weight": [None, "balanced"]}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
search = GridSearchCV(pipeline, grid, scoring="recall", cv=cv, refit=True)
search.fit(sample[FEATURES], sample[TARGET])
for params, score in zip(search.cv_results_["params"],
                         search.cv_results_["mean_test_score"]):
    print(params, score)
print(search.best_params_)
```

| Essai | `C` | `class_weight` | Rappel moyen en validation croisée |
|---|---:|---|---:|
| 1 | 0,01 | `None` | 0,0000 |
| 2 | 0,01 | `balanced` | 0,6667 |
| 3 | 1,0 | `None` | 0,3333 |
| 4 | 1,0 | `balanced` | 0,6667 |
| 5 | 100,0 | `None` | 0,3333 |
| 6 | 100,0 | `balanced` | 0,6667 |

Les essais 2, 4 et 6 sont ex æquo ; `GridSearchCV` choisit le premier, soit `C=0.01` et `class_weight="balanced"`. Le découpage en trois plis entraîne chaque combinaison trois fois sur une partie des 20 lignes, puis réentraîne le meilleur réglage sur les 20 lignes. Il s'agit donc de six combinaisons, et non de six runs MLflow : cette expérience n'enregistre pas de runs MLflow.

Avec seulement trois pannes, chaque pli de validation ne contient qu'une panne : le score est très instable. Aucun jeu de test indépendant n'a été utilisé ici ; ces résultats servent à illustrer la méthode et ne suffisent pas à choisir un modèle pour la production.
