# Compte rendu · TP4 : MaintAI devient un package Python

**Groupe :** 1 · **Nom :** Yasser Hadli · **Date :** 12/11/2026

## 1. Mise en place

J’utilise un environnement virtuel `.venv` propre au dossier du TP4. Les dépendances du projet sont installées à partir de `requirements.txt`.

La sortie initiale des tests avant l’installation du package reste à ajouter : je n’ai pas conservé cette sortie.

Les deux tests concernés sont `test_installed_version_matches_the_code` et `test_command_is_declared`, dans `tests/test_package.py`. Avant l’installation de MaintAI, le premier ne trouve pas les métadonnées du package installé et le second ne trouve pas la commande `maintai-predict` dans les points d’entrée. Avoir le dossier `maintai/` dans le projet ne suffit pas à déclarer une commande dans l’environnement.

## 2. Sprint 1 : pyproject.toml et installation éditable

J’ai complété la description du package, ses dépendances et le point d’entrée de la commande. La version est récupérée depuis `maintai.__version__`, qui vaut `0.2.0`.

```toml
dependencies = [
    "numpy>=2.4",
    "pandas>=3.0",
    "scikit-learn>=1.8",
    "joblib>=1.5",
    "httpx>=0.28",
]

[project.scripts]
maintai-predict = "maintai.cli:main"
```

`pytest` sert aux tests pendant le développement. Il est donc dans le groupe optionnel `dev`. `fastapi` sert à lancer l’API, mais n’est pas nécessaire pour utiliser la prédiction depuis le terminal. Il est dans le groupe optionnel `api`. J’ai gardé `httpx` dans les dépendances principales, car le client `maintai/client.py` l’importe.

Les quatre premières lignes de `python -m pip show maintai` sont :

```text
Name: maintai
Version: 0.2.0
Summary: Prediction de panne a partir de releves de capteurs.
Home-page:
```

La vérification actuelle avec `python -m pytest` donne :

```text
27 passed, 1 warning in 12.01s
```

L’avertissement concerne la dépréciation de l’utilisation de `httpx` par le client de test Starlette. Il ne fait pas échouer les tests.

La commande `maintai-predict 72 6 85 15000`, exécutée dans le projet, affiche :

```text
Probabilité de panne : 95 % (ALERTE)
```

Avec l’installation éditable, Python utilise le code du dossier source du TP4. Depuis un autre dossier, en gardant le même environnement activé, `python -c "import maintai; print(maintai.__file__)"` doit pointer vers :

```text
C:\Users\user\Desktop\Semestre 9 Anfa\Applications Intelligentes\Tps\TP4\maintai-starter-tp4\maintai\__init__.py
```

Le test depuis un autre dossier reste à effectuer pour confirmer ce chemin dans le terminal.

Si je modifie le message dans `maintai/cli.py`, le changement doit être visible au prochain lancement sans réinstaller le package : `pip install -e .` utilise directement les sources. La manipulation du message reste à vérifier.

## 3. Sprint 2 : construire et installer le wheel

Le dossier `dist/` contient les deux fichiers suivants :

```text
maintai-0.2.0-py3-none-any.whl
maintai-0.2.0.tar.gz
```

Le premier est le wheel installable et le second est l’archive des sources.

Le contenu du wheel a été vérifié :

```text
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

Le modèle, les tests et le dossier `ui/` ne sont pas dedans. La configuration `include = ["maintai*"]` sélectionne uniquement les packages Python de MaintAI. Les fichiers `.dist-info` contiennent les métadonnées nécessaires à l’installation.

L’installation dans un environnement neuf reste à réaliser. Voici les comportements attendus, à comparer aux sorties réelles :

| Étape | Comportement attendu, non encore vérifié dans un environnement neuf |
|---|---|
| Sans `MAINTAI_MODELS_DIR` | La commande indique « Modèle introuvable » si aucun modèle n’est présent au chemin par défaut. |
| Avec `MAINTAI_MODELS_DIR` vers mon dossier `models` | La commande charge le modèle et affiche la prédiction ; un avertissement de version est possible si scikit-learn diffère. |
| Après `pip install scikit-learn==1.8.0` | La version correspond à celle du modèle ; l’avertissement de différence de version ne devrait plus apparaître. |

`InconsistentVersionWarning` signifie que la version de scikit-learn utilisée pour charger le modèle diffère de celle utilisée pour l’enregistrer. Dans mon fichier `models/model_info.json`, la ligne suivante indique la version d’entraînement :

```json
"sklearn_version": "1.8.0"
```

Cet avertissement n’est donc pas systématique : il dépend de la version installée dans le nouvel environnement.

Pour livrer l’application, je dois fournir le modèle séparément du wheel, notamment `model.joblib` et `model_info.json`, puis renseigner `MAINTAI_MODELS_DIR` avec le chemin du dossier qui les contient.

## 4. Questions de compréhension

1. **Quelle différence entre requirements.txt et dependencies ?**

   `requirements.txt` fixe les versions exactes utilisées pour ce TP, avec les outils de test et de construction. La liste `dependencies` déclare les bibliothèques nécessaires au package, avec des versions minimales. Elle est lue automatiquement lors de l’installation de MaintAI. Les fonctions optionnelles ont leurs propres groupes.

2. **Quelle différence entre l’installation éditable et le wheel ?**

   Avec `pip install -e .`, mes modifications du code source sont prises en compte directement : c’est adapté au développement. Avec le wheel, pip installe une version construite du package dans l’environnement. Modifier ensuite le dossier source ne modifie pas cette installation ; il faut reconstruire et réinstaller le wheel pour livrer une nouvelle version.

3. **Qu’est-ce qu’un point d’entrée ? Où est créé maintai-predict.exe ?**

   Un point d’entrée associe un nom de commande à une fonction Python. Ici, `maintai-predict` appelle `main` dans `maintai/cli.py`. Sous Windows, pip crée `maintai-predict.exe` dans le dossier `.venv\Scripts\` de l’environnement utilisé pour l’installation.

4. **Pourquoi la page Streamlit n’est-elle pas bloquée par CORS ?**

   L’appel à l’API est effectué par le code Python du serveur Streamlit. CORS est une restriction appliquée par le navigateur aux requêtes entre origines différentes ; il ne bloque pas cet appel entre serveurs.

## 5. Le rendu Git

Dépôt GitHub : https://github.com/Yasser-Hadli/Applications-Intelligentes

Branche de travail : TP

Le fichier `.gitignore` contient déjà `dist/`, `build/`, `*.egg-info/`, `.venv/`, `models/` et `.env`. Je dois vérifier que les fichiers générés et les secrets ne sont pas suivis avant de faire mon commit.

## 6. Ce qui reste flou

Je comprends la différence entre l’installation éditable et le wheel. Le point que je dois encore approfondir est la livraison du modèle séparément du code, surtout la gestion du chemin `MAINTAI_MODELS_DIR` et la compatibilité des versions de scikit-learn sur une autre machine.
