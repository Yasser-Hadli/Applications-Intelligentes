# Compte rendu · TP5 : suivre les entraînements (MLflow) et versionner les données (DVC)

**Groupe :** 5 IAD G1  ·  **Membres :** Mohamed BENHAYOUN SADAFI  ·  **Date :** 25/11/2026

---

## 1. Mise en place (dans `TP/TP5`)

Dernière ligne de `python -m pytest` après `pip install -r requirements.txt` et `pip install -e .` :

```
5 failed, 27 passed, 2 warnings in 19.36s
```

Quels tests échouent, et pourquoi ?

Les cinq tests de `tests/test_tracking.py` : `test_training_creates_a_run`, `test_run_records_metrics`, `test_run_records_data_version`, `test_run_stores_the_model` et `test_options_change_the_run`. Dans le starter, `log_run` ne fait que `return ""` : l'entraînement se termine, mais rien n'est enregistré dans MLflow. `mlflow.search_runs(experiment_names=["maintai"])` ne trouve aucun run, et chaque test s'arrête sur `aucun run MLflow : log_run n'a rien enregistré`. Les 27 autres tests passent, y compris ceux de `test_package.py`, puisque `pip install -e .` a été fait.

Avant `pip install -e .`, où pointait le package `maintai` installé dans l'environnement ? Et après ?

J'ai créé un environnement propre au TP5 (`TP5\.venv`, Python 3.12) au lieu de réutiliser celui du TP4.

- **Avant**, dans ce nouvel environnement, `maintai` n'était pas installé (`pip show maintai` : `Package(s) not found: maintai`). `import maintai` fonctionnait quand même depuis le dossier du TP5, mais seulement parce que le dossier courant est dans `sys.path`. Dans l'environnement du TP4 (celui que propose le polycopié), `pip show maintai` donne `Version: 0.2.0` et `Editable project location: ...\TP4\maintai-starter-tp4` : le package pointait vers le code du TP4. Sans réinstallation, `maintai-predict` aurait lancé le code du TP4, et `test_installed_version_matches_the_code` aurait échoué (0.2.0 ≠ 0.3.0).
- **Après**, `pip show maintai` donne `Version: 0.3.0` et `Editable project location: C:\Users\user\Desktop\5 IAD G1 Developpement applications intelligentes\TP5\TP5\TP5`. Depuis n'importe quel dossier, `python -c "import maintai; print(maintai.__file__)"` affiche `...\TP5\TP5\TP5\maintai\__init__.py`.

## 2. Sprint 1 : MLflow

Votre fonction `log_run` (`maintai/tracking.py`) :

```python
def log_run(params: dict, metrics: dict, model, X_example, data_path: Path) -> str:
    """Enregistre un entraînement : paramètres, version des données, métriques et modèle. Renvoie l'id du run."""
    # Tous les entraînements de MaintAI sont rangés dans la même expérience
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        # Les réglages de l'entraînement (model, C, threshold, n_train)
        mlflow.log_params(params)
        # La version des données : la même empreinte que celle du reçu DVC (data/sensors.csv.dvc)
        mlflow.log_param("data_md5", file_md5(data_path))
        # Les résultats sur le jeu de test
        mlflow.log_metrics(metrics)
        # Le modèle, avec un exemple d'entrée et les versions exactes des bibliothèques (requirements.txt)
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_example,
            skops_trusted_types=TRUSTED_TYPES,  # sinon --model forest échoue au chargement
        )
    return run.info.run_id
```

Dernière ligne de `python -m pytest` à la fin du sprint 1 :

```
32 passed, 4 warnings in 31.11s
```

Vos runs (au moins quatre), relevés dans `mlflow ui` (mode « Model training ») :

Tous les runs ont été entraînés sur les données v1 (`data_md5 = 73e0b7cb…`, 5 000 relevés) et évalués sur le même jeu de test (1 000 relevés, 136 pannes).

| Commande | Run | recall | precision | accuracy |
|---|---|---|---|---|
| `python -m maintai.train` | `f0a233bc` | 0.750 | 0.622 | 0.904 |
| `python -m maintai.train --C 0.01` | `9a23ae38` | 0.691 | 0.653 | 0.908 |
| `python -m maintai.train --threshold 0.2` | `09570622` | 0.838 | 0.543 | 0.882 |
| `python -m maintai.train --model forest` | `7eb12597` | 0.728 | 0.562 | 0.886 |
| `python -m maintai.train --model forest --threshold 0.2` (essai) | `073572e3` | 0.831 | 0.479 | 0.854 |
| `python -m maintai.train --threshold 0.15` (essai) | `1775edba` | 0.875 | 0.500 | 0.864 |

La baseline (toujours « pas de panne ») obtient une accuracy de 0.864 et un recall de 0.

Quel run choisiriez-vous pour MaintAI, et pourquoi ? (Rappel : une panne manquée coûte plus cher qu'une fausse alerte.)

Je choisis la **régression logistique avec le seuil 0.2** (run `095706227bc048438f7e4d27eae5498a`). Elle détecte 114 pannes sur 136 (recall 0.838), au lieu de 102 au seuil 0.3. Elle donne 34 fausses alertes de plus (96 au lieu de 62), soit environ 2,8 fausses alertes par panne détectée en plus : c'est rentable, puisqu'une panne manquée coûte plus cher qu'une inspection.

- `--C 0.01` est le pire choix : plus de régularisation, et 42 pannes manquées.
- La **forêt aléatoire n'est pas meilleure** : au même seuil, elle a à la fois un recall plus faible et une precision plus faible que la régression logistique (0.728 / 0.562 contre 0.750 / 0.622 à 0.3 ; 0.831 / 0.479 contre 0.838 / 0.543 à 0.2). Elle est aussi plus lourde.
- Le seuil 0.15 détecte encore 5 pannes de plus, mais au prix de 23 fausses alertes de plus. Une alerte sur deux serait fausse, et l'accuracy retombe au niveau de la baseline. Je le garde en réserve si le coût réel d'une panne s'avère très élevé.

Ce choix a été revu ensuite par la grid search de la partie 3 (section 5), qui retient un gradient boosting au seuil 0.15.

Dans `mlflow ui`, ouvrez le modèle d'un run. Quelle version de scikit-learn est indiquée dans son `requirements.txt` ? Quel problème du TP4 cela règle-t-il ?

`scikit-learn==1.8.0`. Le fichier complet, pour le run `09570622` :

```
mlflow==3.18.0
numpy==2.4.4
pandas==3.0.2
psutil==7.2.2
pyarrow==26.0.1
pytest==9.1.1
scikit-learn==1.8.0
scipy==1.18.1
skops==0.16.0
```

Le fichier `MLmodel` indique aussi `sklearn_version: 1.8.0` et `serialization_format: skops`.

Au TP4, le wheel n'exigeait que `scikit-learn>=1.5` : pip avait installé la 1.9.1, et le chargement du modèle entraîné en 1.8.0 déclenchait `InconsistentVersionWarning`. La seule trace de la bonne version était une ligne de `model_info.json`. Maintenant, chaque modèle enregistré **emporte ses versions exactes** (`requirements.txt`, `conda.yaml`, `python_env.yaml`) : on recrée l'environnement d'entraînement avec `pip install -r requirements.txt`, ou automatiquement avec `mlflow models serve`.

Où MLflow enregistre-t-il les runs sur votre disque ? Ces fichiers vont-ils dans Git ? Pourquoi ?

Dans le dossier d'où l'on lance l'entraînement (`TP5\TP5\TP5`) :

- `mlflow.db`, une base SQLite (`sqlite:///…/TP5/TP5/TP5/mlflow.db`) : l'expérience `maintai`, ses runs, paramètres, métriques et tags ;
- `mlruns/1/` : les artefacts. Chaque modèle est dans `mlruns/1/models/m-<id>/artifacts/` (`model.skops`, `MLmodel`, `requirements.txt`, `conda.yaml`, `python_env.yaml`, `input_example.json`).

Ils ne vont pas dans Git : `mlflow.db` et `mlruns/` sont dans le `.gitignore`. Ce sont des fichiers générés, binaires, qui grossissent à chaque entraînement. Ils contiennent des chemins absolus propres à mon PC (`file:C:/Users/user/...`), et une base SQLite ne se fusionne pas entre plusieurs membres. Git garde le code qui permet de refaire les runs. Pour partager les runs dans le groupe, il faudrait un serveur MLflow commun (`mlflow server`) plutôt qu'un fichier local.

## 3. Sprint 2 : DVC

Contenu de `data/sensors.csv.dvc` après `dvc add data/sensors.csv` :

```
outs:
- md5: 73e0b7cb3c6f795c70c3ddf970b99c15
  size: 126815
  hash: md5
  path: sensors.csv
```

Comparez la valeur `md5` de ce fichier au paramètre `data_md5` de votre dernier run MLflow. Que constatez-vous ?

Elles sont **identiques** : `73e0b7cb3c6f795c70c3ddf970b99c15`, pour les six runs entraînés sur la v1. `file_md5` calcule le MD5 des octets du fichier, exactement comme DVC 3 (`hash: md5`). Le run et le reçu Git parlent donc de la même version des données. Depuis un run, on retrouve ses données : on cherche le commit dont le reçu `.dvc` porte ce md5, puis `git checkout <commit> -- data/sensors.csv.dvc` et `dvc checkout`. De même, le run entraîné sur la v2 porte `data_md5 = 93fdeef2…`, le md5 du reçu de la v2.

Pourquoi `data/sensors.csv` ne va pas dans Git, alors que `data/sensors.csv.dvc` y va ?

Le CSV est un gros fichier de données (127 Ko ici, mais des centaines de Mo pour de vraies données) : Git garde chaque version pour toujours dans l'historique, chaque `git clone` les télécharge toutes, et GitHub refuse les fichiers de plus de 100 Mo. Un diff ligne à ligne de données n'a pas de sens, et les données peuvent être confidentielles. DVC range le fichier dans son cache (`.dvc/cache`), puis dans le stockage. Le CSV est ignoré par Git grâce à la ligne `data/*.csv` du `.gitignore` du projet. C'est pourquoi `dvc add` n'a pas eu besoin de créer `data/.gitignore` et ne demande que `git add data\sensors.csv.dvc`.

Le `.dvc`, lui, est un **reçu** de 5 lignes de texte (md5, taille, chemin). Versionné avec le code, il fixe pour chaque commit la version exacte des données qui va avec.

Après `python -m maintai.dataset --n 8000 --seed 7`, que dit `dvc status` ?

```
data\sensors.csv.dvc:
	changed outs:
		modified:           data\sensors.csv
```

Le fichier de travail ne correspond plus au reçu : DVC voit que les données ont changé et qu'il faut refaire `dvc add` pour enregistrer une nouvelle version.

| Version | Commit Git (`git log --oneline`) | md5 (8 premiers caractères) | Nombre de lignes |
|---|---|---|---|
| v1 | `86484ca` Données v1 | `73e0b7cb` | 5000 |
| v2 | `ce83c1c` Données v2 | `93fdeef2` | 8000 |

Sortie de `git log --oneline` (dépôt Git local du dossier TP5, où vit DVC) :

```
ce83c1c Données v2
86484ca Données v1
1bb831b DVC : init
7b38007 TP5 : starter
```

La v1 fait 126 815 octets et contient 13,6 % de pannes ; la v2 fait 202 834 octets et en contient 12,8 %.

Revenez à la v1 (`git checkout <commit> -- data/sensors.csv.dvc`, puis `dvc checkout`). Combien de lignes contient `data/sensors.csv` ? Puis revenez à la v2.

**5000 lignes**, avec `git checkout 86484ca -- data/sensors.csv.dvc`. `dvc checkout` affiche `M data\sensors.csv` : il remplace le fichier de travail par la version du cache qui correspond au reçu v1. `git status` montre alors `M  data/sensors.csv.dvc`, puisque c'est l'ancien reçu : je ne le commite pas. Retour à la v2 avec `git checkout HEAD -- data/sensors.csv.dvc` puis `dvc checkout` : **8000 lignes**.

Supprimez `data/sensors.csv` et le dossier `.dvc/cache`, puis lancez `dvc pull`. Que se passe-t-il ?

```
A       data\sensors.csv
1 file fetched and 1 file added
```

DVC lit le reçu, constate que le fichier n'est ni dans l'espace de travail ni dans le cache, et le **télécharge depuis le stockage** (`%USERPROFILE%\dvc-stockage`, soit `C:\Users\user\dvc-stockage`) vers le cache. Il le remet ensuite dans `data/`. Le fichier revient avec 8000 lignes et le même md5 (`93fdeef2…`). C'est exactement la situation d'un coéquipier qui vient de faire `git clone` : Git lui donne le reçu, `dvc pull` lui donne les données.

## 4. Relier modèle et données

Entraînez un modèle sur la v2. Dans `mlflow ui`, comparez ce run à celui de la v1 avec les mêmes réglages :

Mêmes réglages dans les deux cas : régression logistique, C = 1.0, seuil 0.3.

| Run | data_md5 (8 caractères) | n_train | recall | precision |
|---|---|---|---|---|
| v1 (`f0a233bc`) | `73e0b7cb` | 4000 | 0.750 | 0.622 |
| v2 (`9a14011c`) | `93fdeef2` | 6400 | 0.741 | 0.650 |

(accuracy : 0.904 sur la v1, 0.916 sur la v2)

Peut-on dire que le modèle v2 est « meilleur » ? Pourquoi faut-il être prudent en comparant deux runs entraînés sur des données différentes ?

Non. Le run v2 a un recall un peu plus faible et une precision un peu plus forte, mais **les deux modèles n'ont pas passé le même examen**.

- Le jeu de test de la v1 contient 1 000 relevés et 136 pannes ; celui de la v2, 1 600 relevés et 205 pannes, tirés avec une autre graine.
- Le taux de panne change (13,6 % contre 12,8 %), ce qui suffit à déplacer la precision et l'accuracy sans que le modèle soit meilleur.
- L'écart est minuscule : sur 136 pannes, une panne de plus ou de moins fait varier le recall de 0,7 point. Les 0,9 point d'écart représentent une ou deux pannes, ce qui relève du bruit.

Pour comparer honnêtement deux modèles, il faut les évaluer sur le **même** jeu de test, idéalement des données récentes que ni l'un ni l'autre n'a vues, ou par validation croisée. Une seule chose doit changer entre les deux runs. Grâce à `data_md5`, MLflow montre justement que ces deux runs ne sont pas comparables directement.

## 5. Partie 3 : grid search (30 runs maximum)

**But :** comparer de façon systématique plusieurs entraînements, au lieu de quelques essais à la main, et garder chacun comme un run MLflow.

### Choix des hyperparamètres

- **Learning rate :** la régression logistique et la forêt aléatoire n'en ont pas. J'ai donc ajouté un **gradient boosting** (`GradientBoostingClassifier`), un modèle de ML classique qui a un vrai `learning_rate` : le poids de chaque nouvel arbre ajouté. Trop petit, le modèle apprend trop lentement en 100 arbres ; trop grand, il sur-apprend. Valeurs : **0.05 · 0.1 · 0.2**. Un premier essai avec 0.01, 0.03 et 0.5 a montré que ces valeurs-là étaient moins bonnes.
- **Dropout :** non retenu. C'est la probabilité (entre 0 et 1) d'éteindre un neurone pendant l'entraînement : il n'existe que dans un **réseau de neurones**, et MaintAI n'en utilise pas.
- **Seuil d'alerte :** **10 valeurs de 0.1 à 0.35** (0.1 · 0.125 · 0.15 · 0.175 · 0.2 · 0.225 · 0.25 · 0.275 · 0.3 · 0.35). Au-dessus de 0.35, le recall chute : avec la régression logistique sur la v2, il vaut 0.58 à 0.5 et 0.45 à 0.7. Sous 0.1, la precision passe sous 0.5. La grille est donc resserrée sur la zone utile.
- **Taille :** 3 learning rates × 10 seuils = **30 runs**. Les autres réglages sont fixes : 100 arbres, `random_state = 42`.
- **Données :** v2 (commit `ce83c1c`, `data_md5 = 93fdeef2…`, 6 400 relevés d'entraînement, 1 600 de test dont 205 pannes).
- **Critère de choix :** le plus grand **recall** parmi les runs dont la **precision est ≥ 0.5** (au moins une alerte sur deux est une vraie panne).

### Le code

Dans `maintai/train.py`, un troisième modèle et une option `--lr` :

```python
def build_model(kind: str = "logreg", C: float = 1.0, lr: float = 0.1) -> Pipeline:
    if kind == "forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=SEED)
    elif kind == "gboost":
        # Gradient boosting : 100 petits arbres ajoutés un par un ; lr = poids de chaque nouvel arbre
        clf = GradientBoostingClassifier(learning_rate=lr, n_estimators=100, random_state=SEED)
    else:
        clf = LogisticRegression(C=C, max_iter=1000)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])

# parse_args :  p.add_argument("--lr", type=float, default=0.1, help="learning rate du gradient boosting")
# main :        elif args.model == "gboost": params["learning_rate"] = args.lr
```

Le script de la grille, `maintai/grid_search.py` :

```python
LEARNING_RATES = [0.05, 0.1, 0.2]
THRESHOLDS = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35]
MIN_PRECISION = 0.5  # critère : au moins une alerte sur deux est une vraie panne


def run_grid() -> list[dict]:
    """Lance les 30 entraînements et renvoie, pour chacun, ses hyperparamètres et ses métriques."""
    grid = list(itertools.product(LEARNING_RATES, THRESHOLDS))
    assert len(grid) <= 30, "la grille ne doit pas dépasser 30 runs"
    results = []
    for i, (lr, threshold) in enumerate(grid, start=1):
        print(f"[{i}/{len(grid)}] learning_rate={lr}  threshold={threshold}")
        metrics = main(["--model", "gboost", "--lr", str(lr), "--threshold", str(threshold)])
        run_id = mlflow.last_active_run().info.run_id
        mlflow.MlflowClient().set_tag(run_id, "grid", "tp5")
        results.append({"run_id": run_id, "learning_rate": lr, **metrics["model"]})
    return results


def best_first(results: list[dict]) -> list[dict]:
    """Classe les runs : d'abord ceux qui respectent la precision minimale, puis par recall, puis par precision."""
    return sorted(
        results,
        key=lambda r: (r["precision"] >= MIN_PRECISION, r["recall"], r["precision"]),
        reverse=True,
    )
```

Chaque combinaison appelle `train.main`, donc chaque entraînement passe par `log_run` et devient un run MLflow complet : paramètres, `data_md5`, métriques et modèle. Lancement :

```
python -m maintai.grid_search
```

Dans `mlflow ui` (mode « Model training », expérience `maintai`), le filtre `tags.grid = "tp5"` n'affiche que ces 30 runs.

### Les 30 entraînements

Classés par le critère : precision ≥ 0.5 d'abord, puis par recall décroissant.

| # | Run | learning_rate | threshold | recall | precision | accuracy | precision ≥ 0.5 |
|---|---|---|---|---|---|---|---|
| 1 | `341e1560` | 0.05 | 0.15 | 0.868 | 0.500 | 0.872 | oui |
| 2 | `d25018b1` | 0.1 | 0.15 | 0.854 | 0.515 | 0.878 | oui |
| 3 | `8dcdb6dd` | 0.2 | 0.15 | 0.844 | 0.524 | 0.882 | oui |
| 4 | `4a429e18` | 0.05 | 0.175 | 0.839 | 0.526 | 0.882 | oui |
| 5 | `11dcf0f2` | 0.1 | 0.175 | 0.829 | 0.528 | 0.883 | oui |
| 6 | `9525a103` | 0.05 | 0.2 | 0.815 | 0.559 | 0.894 | oui |
| 7 | `0adaed06` | 0.2 | 0.175 | 0.815 | 0.546 | 0.889 | oui |
| 8 | `6cd93f54` | 0.1 | 0.2 | 0.800 | 0.545 | 0.889 | oui |
| 9 | `c053736c` | 0.05 | 0.225 | 0.795 | 0.588 | 0.902 | oui |
| 10 | `bc6c800b` | 0.2 | 0.2 | 0.795 | 0.566 | 0.896 | oui |
| 11 | `22134b1f` | 0.2 | 0.225 | 0.780 | 0.584 | 0.901 | oui |
| 12 | `15a31e45` | 0.1 | 0.225 | 0.776 | 0.570 | 0.896 | oui |
| 13 | `21bb061f` | 0.2 | 0.25 | 0.771 | 0.605 | 0.906 | oui |
| 14 | `c2859ebd` | 0.1 | 0.25 | 0.766 | 0.599 | 0.904 | oui |
| 15 | `2f649e30` | 0.2 | 0.275 | 0.756 | 0.613 | 0.907 | oui |
| 16 | `bd3f3634` | 0.05 | 0.25 | 0.756 | 0.598 | 0.904 | oui |
| 17 | `c9071863` | 0.1 | 0.275 | 0.751 | 0.611 | 0.907 | oui |
| 18 | `34ef94af` | 0.1 | 0.3 | 0.732 | 0.617 | 0.907 | oui |
| 19 | `1f730234` | 0.05 | 0.275 | 0.722 | 0.609 | 0.905 | oui |
| 20 | `748ae572` | 0.05 | 0.3 | 0.717 | 0.631 | 0.910 | oui |
| 21 | `fcacd87c` | 0.2 | 0.3 | 0.717 | 0.626 | 0.909 | oui |
| 22 | `62854e07` | 0.1 | 0.35 | 0.683 | 0.670 | 0.916 | oui |
| 23 | `57d0ef30` | 0.2 | 0.35 | 0.678 | 0.653 | 0.912 | oui |
| 24 | `415a102e` | 0.05 | 0.35 | 0.644 | 0.673 | 0.914 | oui |
| 25 | `373d0075` | 0.05 | 0.1 | 0.932 | 0.436 | 0.837 | non |
| 26 | `e3eed8c4` | 0.1 | 0.1 | 0.912 | 0.458 | 0.851 | non |
| 27 | `287f85c1` | 0.05 | 0.125 | 0.893 | 0.473 | 0.859 | non |
| 28 | `904b5893` | 0.2 | 0.125 | 0.878 | 0.497 | 0.871 | non |
| 29 | `1ce757ca` | 0.2 | 0.1 | 0.878 | 0.458 | 0.851 | non |
| 30 | `3649b401` | 0.1 | 0.125 | 0.873 | 0.485 | 0.865 | non |

Les 6 derniers ont le meilleur recall brut (jusqu'à 0.932), mais une alerte sur deux ou plus y est fausse : ils sont éliminés par le critère.

### Les 6 meilleurs modèles

Sur 1 600 relevés de test, dont 205 pannes :

| Rang | Run | learning_rate | threshold | recall | precision | accuracy | Pannes détectées | Pannes manquées | Fausses alertes |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **`341e1560`** | **0.05** | **0.15** | **0.868** | **0.500** | **0.872** | **178** | **27** | **178** |
| 2 | `d25018b1` | 0.1 | 0.15 | 0.854 | 0.515 | 0.878 | 175 | 30 | 165 |
| 3 | `8dcdb6dd` | 0.2 | 0.15 | 0.844 | 0.524 | 0.882 | 173 | 32 | 157 |
| 4 | `4a429e18` | 0.05 | 0.175 | 0.839 | 0.526 | 0.882 | 172 | 33 | 155 |
| 5 | `11dcf0f2` | 0.1 | 0.175 | 0.829 | 0.528 | 0.883 | 170 | 35 | 152 |
| 6 | `9525a103` | 0.05 | 0.2 | 0.815 | 0.559 | 0.894 | 167 | 38 | 132 |

### Le modèle choisi, et pourquoi

**Gradient boosting, learning_rate = 0.05, seuil 0.15** (run `341e156009fc48629f5ae70a29c7a912`).

- **C'est le meilleur selon le critère :** le plus grand recall (0.868, soit 178 pannes sur 205) parmi les runs dont la precision est ≥ 0.5. Sa precision exacte vaut 178 / 356 = 0.500 : il respecte la limite, de justesse.
- **Le learning rate le plus petit gagne à chaque seuil bas** (0.15 et 0.175). Avec 100 arbres, un pas de 0.05 corrige les erreurs plus progressivement, et le modèle généralise mieux qu'avec 0.1 ou 0.2.
- **Le seuil fait plus que le learning rate.** À learning rate fixe, passer de 0.35 à 0.15 fait gagner environ 20 points de recall. Entre les trois learning rates, à seuil égal, l'écart n'est que de 1 à 3 points.
- **Face à la régression logistique sur la même v2 et le même jeu de test** : au seuil 0.15, elle détecte 176 pannes (29 manquées) avec 172 fausses alertes, et au seuil 0.2, 169 pannes (36 manquées) avec 129 fausses alertes. Le gradient boosting retenu détecte **2 pannes de plus** que la meilleure régression logistique, pour 6 fausses alertes de plus. Le gain est réel mais **petit**.
- **Prudence :** le seuil est choisi sur le jeu de test lui-même, donc les chiffres sont un peu optimistes. Et le premier run est pile à la limite de precision : sur de nouvelles données, il pourrait passer en dessous. Le run n° 2 (lr 0.1, seuil 0.15 : recall 0.854, precision 0.515) est la solution de repli, avec plus de marge.

## 6. Décision : l'ADR 004

Voir `004-choix-du-modele_BENHAYOUN_SADAFI_Mohamed.md` (dans le projet : `docs/adr/`). La décision a été révisée après la grid search :

- **Modèle :** gradient boosting (`GradientBoostingClassifier`, 100 arbres), learning_rate = 0.05
- **Seuil d'alerte :** 0.15 (recall 0.868, precision 0.500, accuracy 0.872)
- **Run MLflow :** `341e156009fc48629f5ae70a29c7a912`
- **data_md5 :** `93fdeef2cac1893c4f720c16457eea75` (données v2, commit `ce83c1c`, 8 000 relevés)
- **Ce que l'on accepte :** 178 fausses alertes pour 1 600 relevés (une alerte sur deux), pour ne manquer que 27 pannes sur 205.
- **Avant la grid search** (sprint 1, données v1), le choix était la régression logistique au seuil 0.2 (run `095706227bc048438f7e4d27eae5498a`).

## 7. Questions de compréhension

1. Quelle différence entre Git, DVC et MLflow ? Que versionne chacun ?
   **Git** versionne le **code** et les petits fichiers texte : le code Python, les tests, les ADR, et les reçus `.dvc`. **DVC** versionne les **données** (et plus largement les gros fichiers) : il les garde dans un cache et un stockage, identifiées par leur md5, et ne laisse dans Git qu'un reçu. **MLflow** garde la trace des **entraînements** : pour chaque run, les paramètres, l'empreinte des données, les métriques et le modèle avec son environnement exact. Git répond à « quel code ? », DVC à « quelles données ? », MLflow à « quel réglage a donné quel résultat, et avec quel modèle ? ».
2. Pourquoi le stockage DVC est-il ajouté avec `--local` dans ce TP ? Que faudrait-il pour que tout le groupe partage les mêmes données ?
   Le stockage est un dossier de **mon** PC (`%USERPROFILE%\dvc-stockage`) : ce chemin n'a aucun sens sur la machine d'un coéquipier, et un chemin personnel n'a pas à aller dans Git. Avec `--local`, il est écrit dans `.dvc/config.local`, ignoré par Git, et non dans `.dvc/config`. Pour partager les données, il faut un stockage accessible à tous (S3, Google Drive, serveur SSH, partage réseau), ajouté **sans** `--local` (`dvc remote add -d stockage s3://…`) pour que `.dvc/config` soit commité. Les identifiants de chacun restent dans `config.local`. Ensuite, `git pull` puis `dvc pull` donnent à tout le groupe les mêmes données.
3. Livrer un modèle, c'est trois choses (TP4). Lesquelles, et où sont-elles maintenant ?
   Au TP4 : **le code** (le package, le wheel), **le modèle entraîné** (`models/model.joblib`) et **les versions exactes** des bibliothèques (sinon `InconsistentVersionWarning`). Maintenant :
   - le code est dans **Git** (et le wheel du TP4) ;
   - le modèle est dans **MLflow**, retrouvable par l'identifiant de son run (`runs:/341e156009fc48629f5ae70a29c7a912/model`, le modèle retenu après la grid search) ;
   - les versions exactes sont dans le `requirements.txt` du modèle, enregistré par **MLflow**.

   Le TP5 ajoute une quatrième chose : **les données** d'entraînement, avec le reçu dans Git et le fichier dans le stockage **DVC**, reliées au run par `data_md5`.

## 8. Le rendu Git

- Lien de la pull request : pas de pull request. Comme pour le TP4, le rendu (`tracking.py`, `train.py`, `grid_search.py`, ce compte rendu et l'ADR 004) a été poussé directement sur la branche `TP` du dépôt https://github.com/Yasser-Hadli/Applications-Intelligentes, dans `TP/TP5/BENHAYOUN_SADAFI_Mohamed` : https://github.com/Yasser-Hadli/Applications-Intelligentes/tree/TP/TP/TP5/BENHAYOUN_SADAFI_Mohamed
- Le tag `tp5` existe sur GitHub : oui
- `git status` avant le commit : `mlflow.db`, `mlruns/`, `data/sensors.csv` ou `.dvc/cache` apparaissaient-ils ? non. Le `.gitignore` du TP5 exclut `mlflow.db`, `mlruns/` et `data/*.csv`, et le `.dvc/.gitignore` créé par `dvc init` exclut `/cache`, `/tmp` et `/config.local`. Le `.gitignore` du dépôt du groupe exclut aussi `*.db`, `mlruns/` et `data/`. Seuls les fichiers du rendu ont été ajoutés.

## 9. Ce qui reste flou

- Comment partager les runs MLflow dans le groupe : `mlflow.db` reste sur mon PC. Faut-il un serveur MLflow commun, et où l'héberger ?
- Le choix du seuil : je l'ai réglé sur le jeu de test, ce qui rend les chiffres un peu optimistes. Je ne sais pas encore quand utiliser un jeu de validation à part ou la validation croisée pour choisir le seuil sans tricher.
- Comment comparer proprement un modèle v1 et un modèle v2 : faut-il garder un jeu de test fixe, hors de toutes les versions des données ?
- Pour un vrai stockage DVC partagé (Google Drive, S3) : comment gérer l'authentification de chaque membre sans mettre de secret dans Git.
