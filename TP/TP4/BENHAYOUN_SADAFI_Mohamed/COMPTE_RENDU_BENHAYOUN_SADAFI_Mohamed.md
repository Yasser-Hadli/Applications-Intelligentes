# Compte rendu · TP4 : MaintAI devient un package Python

**Groupe :** 5 IAD G1  ·  **Membres :** Mohamed BENHAYOUN SADAFI  ·  **Date :** 21/11/2026

---

## 1. Mise en place (dans `TP/TP4`, avec son propre `.venv`)

Dernière ligne de `python -m pytest` juste après `pip install -r requirements.txt` :

```
2 failed, 25 passed, 1 warning in 30.35s
```

Quels sont les deux tests qui échouent, et pourquoi ?

Les deux tests de `tests/test_package.py` :

- `test_installed_version_matches_the_code` : il appelle `importlib.metadata.version("maintai")`, qui lit les métadonnées d'un package **installé**. Or `maintai` n'est pas encore installé (pytest le trouve seulement parce qu'il est dans le dossier courant), donc il n'y a pas de métadonnées et le test lève `PackageNotFoundError`.
- `test_command_is_declared` : il cherche `maintai-predict` dans les points d'entrée `console_scripts` déclarés par les packages installés. La commande n'est ni déclarée dans `pyproject.toml` (TODO), ni installée, donc l'assertion échoue.

## 2. Sprint 1 : `pyproject.toml` et `pip install -e .`

Vos sections `dependencies` et `[project.scripts]` :

```toml
dependencies = [
    "numpy>=2.0",
    "pandas>=2.2",
    "scikit-learn>=1.5",
    "joblib>=1.4",
    "httpx>=0.27",  # utilisé par maintai/client.py
]

[project.scripts]
maintai-predict = "maintai.cli:main"
```

Pourquoi `pytest` et `fastapi` ne sont-ils pas dans `dependencies` ?

`dependencies` contient seulement ce dont le cœur du package a besoin pour fonctionner (`maintai.predict`, `maintai.cli`, etc.). `pytest` ne sert qu'au développeur pour lancer les tests : il est dans le groupe `dev`. `fastapi` et `uvicorn` ne servent qu'à faire tourner l'API : ils sont dans le groupe `api`. Quelqu'un qui veut seulement utiliser `maintai-predict` n'a pas à les installer. Ceux qui en ont besoin font `pip install maintai[api]` ou `pip install -e .[dev]`.

Sortie de `pip show maintai` (les quatre premières lignes) :

```
Name: maintai
Version: 0.2.0
Summary: Prédiction de panne de machines à partir de relevés de capteurs.
Home-page:
```

Dernière ligne de `python -m pytest` à la fin du sprint 1 :

```
27 passed, 1 warning in 7.79s
```

Placez-vous dans un autre dossier (`cd %TEMP%`) et lancez `maintai-predict 72 6 85 15000`, puis `python -c "import maintai; print(maintai.__file__)"`. Qu'affichent ces deux commandes ? Où Python trouve-t-il le code de `maintai` ?

```
Probabilité de panne : 95 % (ALERTE)
C:\Users\user\Desktop\5 IAD G1 Developpement applications intelligentes\TP4\maintai-starter-tp4\maintai\__init__.py
```

La commande marche depuis n'importe quel dossier, car elle est installée dans `.venv\Scripts`. Python trouve le code dans **le dossier du projet**, pas dans une copie : `pip install -e .` a seulement déposé dans `.venv\Lib\site-packages` un fichier `__editable__.maintai-0.2.0.pth` et un « finder » (`__editable___maintai_0_2_0_finder.py`) qui redirigent `import maintai` vers `maintai-starter-tp4\maintai\`.

Modifiez le message affiché par `maintai/cli.py`, puis relancez `maintai-predict` sans réinstaller. Le changement est-il visible ? Pourquoi ?

Oui. Après avoir remplacé « Probabilité de panne : » par « Probabilité de panne (modifiée) : », la commande affiche directement `Probabilité de panne (modifiée) : 95 % (ALERTE)`. En mode éditable, le package installé pointe vers le code source du projet : toute modification du code est prise en compte au prochain lancement, sans réinstaller. Il faut réinstaller seulement si on change `pyproject.toml` (dépendances, commandes, version…).

## 3. Sprint 2 : construire et installer le wheel

Sortie de `dir dist` après `python -m build` :

```
Name                           Length
----                           ------
maintai-0.2.0-py3-none-any.whl  11623
maintai-0.2.0.tar.gz            12288
```

Contenu du wheel (`tar -tf dist\maintai-0.2.0-py3-none-any.whl`). Le modèle, les tests et `ui/` y sont-ils ? Pourquoi ?

```
maintai/__init__.py
maintai/cli.py
maintai/client.py
maintai/config.py
maintai/data.py
maintai/dataset.py
maintai/predict.py
maintai/train.py
maintai/api/__init__.py
maintai/api/main.py
maintai/api/schemas.py
maintai/api/security.py
maintai-0.2.0.dist-info/METADATA
maintai-0.2.0.dist-info/WHEEL
maintai-0.2.0.dist-info/entry_points.txt
maintai-0.2.0.dist-info/top_level.txt
maintai-0.2.0.dist-info/RECORD
```

Non : ni `models/`, ni `tests/`, ni `ui/`, ni `web/`, ni `data/`. La section `[tool.setuptools.packages.find]` avec `include = ["maintai*"]` ne met dans le package que le dossier `maintai/` et ses sous-packages. Le wheel distribue le **code** ; le modèle est un artefact produit par l'entraînement, il dépend des données et se remplace sans changer le code. On indique donc son emplacement à l'exécution (`MAINTAI_MODELS_DIR`). Le dossier `maintai-0.2.0.dist-info/` contient les métadonnées lues par pip : dépendances (`METADATA`) et commande `maintai-predict` (`entry_points.txt`).

Dans un environnement neuf (`%TEMP%\essai`), installez le wheel et lancez `maintai-predict` :

| Étape | Ce qui s'affiche |
|---|---|
| sans `MAINTAI_MODELS_DIR` | `Modèle introuvable : C:\Users\user\AppData\Local\Temp\essai\.venv\Lib\site-packages\models\model.joblib` puis `Entraînez-le (python -m maintai.train) ou indiquez son dossier dans MAINTAI_MODELS_DIR.` |
| avec `MAINTAI_MODELS_DIR` vers vos `models` | 3 avertissements `InconsistentVersionWarning: Trying to unpickle estimator StandardScaler / LogisticRegression / Pipeline from version 1.8.0 when using version 1.9.1`, puis `Probabilité de panne : 95 % (ALERTE)` |
| après `pip install scikit-learn==1.8.0` | `Probabilité de panne : 95 % (ALERTE)` (plus aucun avertissement) |

Sans la variable, `config.py` cherche le modèle à côté du code installé (`site-packages\models`), où il n'existe pas.

D'où vient l'avertissement `InconsistentVersionWarning` ? Quelle ligne de `models/model_info.json` permet de le comprendre ?

Le wheel installe les dépendances de `dependencies`, qui ne donnent qu'une version **minimale** (`scikit-learn>=1.5`) : pip a donc pris la dernière, scikit-learn 1.9.1. Mais le modèle a été sérialisé (joblib/pickle) avec scikit-learn 1.8.0, la version figée dans `requirements.txt`. En le rechargeant avec une autre version, scikit-learn prévient que les objets internes ont pu changer et que les résultats ne sont pas garantis. La ligne qui l'explique :

```json
"sklearn_version": "1.8.0"
```

Après `pip install scikit-learn==1.8.0`, la version correspond à celle de l'entraînement et l'avertissement disparaît.

## 4. Questions de compréhension

1. Quelle différence entre `requirements.txt` et la liste `dependencies` de `pyproject.toml` ?
   `requirements.txt` fige **toutes** les versions exactes (`==`) de l'environnement de travail : ML, API, interface, tests, outils de build. Il sert à reproduire exactement l'environnement où le modèle a été entraîné. `dependencies` décrit ce dont le **package** a besoin pour fonctionner, avec des versions minimales (`>=`), pour rester compatible avec d'autres packages installés à côté. pip la lit automatiquement quand on installe `maintai` ; `requirements.txt` doit être installé à la main avec `pip install -r`.
2. Quelle différence entre `pip install -e .` et `pip install dist\maintai-0.2.0-py3-none-any.whl` ?
   `pip install -e .` (éditable) ne copie pas le code : il crée un lien vers le dossier du projet, et les modifications du code sont visibles immédiatement. C'est le mode du développeur. `pip install …whl` **copie** les fichiers figés du wheel dans `site-packages` : le code installé ne bouge plus tant qu'on n'installe pas un nouveau wheel, et il n'a pas besoin des sources. C'est le mode de l'utilisateur ou du serveur de production.
3. Qu'est-ce qu'un « point d'entrée » (`[project.scripts]`) ? Où pip crée-t-il le fichier `maintai-predict.exe` ?
   C'est une association « nom de commande = module:fonction » (`maintai-predict = "maintai.cli:main"`). À l'installation, pip génère un petit exécutable qui importe `maintai.cli` et appelle `main()`, ce qui permet de lancer `maintai-predict` comme une commande du système au lieu de `python -m maintai.cli`. Il le crée dans le dossier `Scripts` de l'environnement virtuel : `.venv\Scripts\maintai-predict.exe` (pour l'essai : `C:\Users\user\AppData\Local\Temp\essai\.venv\Scripts\maintai-predict.exe`). Ce dossier est dans le `PATH` quand le venv est activé.

## 5. Le rendu Git

- Lien de la pull request : pas de pull request, le rendu a été poussé directement sur la branche `TP` du dépôt https://github.com/Yasser-Hadli/Applications-Intelligentes (commit https://github.com/Yasser-Hadli/Applications-Intelligentes/commit/ff881b28ed1c430a91e37f61a34889cca09dad49)
- Le tag `tp4` existe sur GitHub : oui
- `git status` avant le commit : `dist/`, `build/` ou `maintai.egg-info/` apparaissaient-ils ? non : ils sont listés dans le `.gitignore` du projet (`dist/`, `build/`, `*.egg-info/`), comme `.venv/` et `models/`. Seuls `pyproject.toml` et le compte rendu ont été ajoutés.

## 6. Ce qui reste flou

- Comment distribuer proprement le modèle avec le package : le wheel ne contient que le code, et il faut indiquer `MAINTAI_MODELS_DIR` à la main. Je ne sais pas encore quelle est la bonne pratique en production (dossier partagé, téléchargement au premier lancement, image Docker…).
- Le choix des versions minimales dans `dependencies` : avec `scikit-learn>=1.5`, pip installe la dernière version (1.9.1) et le modèle entraîné en 1.8.0 affiche un avertissement. Faut-il plutôt borner la version (`scikit-learn>=1.8,<1.9`) pour rester cohérent avec le modèle, au risque de gêner d'autres packages ?
