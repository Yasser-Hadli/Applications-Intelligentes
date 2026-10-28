# Projet à compléter : entraîner et comparer des modèles

Vous écrivez vous-mêmes le chargement des données, le découpage entraînement / test, les modèles à comparer et leur évaluation. La structure du projet et les tests sont fournis : **les tests disent ce qui est attendu**, et vous avez fini quand ils passent tous.

## 1. Installer (une fois)

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows ; macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                    # au départ : 7 failed, 1 passed (c'est normal)
```

## 2. Régler config.py

Votre fichier CSV dans `data/`, puis dans `monprojet/config.py` : `DATA_PATH`, `TARGET`, `TASK`, `MAIN_METRIC` (celle de votre section 5), `SPLIT` et `TIME_COLUMN` pour une série temporelle, `TEXT_COLUMN` pour du texte libre. Pas encore de données ? Gardez l'exemple `data/exemple_pannes.csv`.

## 3. Écrire le code, fonction par fonction

| Ordre | Fichier | Fonction | Tests qui passent ensuite |
|---|---|---|---|
| 1 | `monprojet/data.py` | `load_data()`, `split()` | 3 passed |
| 2 | `monprojet/models.py` | `candidates()` : la baseline d'abord, puis au moins trois modèles | 5 passed |
| 3 | `monprojet/train.py` | `evaluate()`, puis `main()` | 8 passed |

Chaque fonction contient un commentaire `TODO` qui dit ce qu'elle doit faire et quelles fonctions de pandas ou scikit-learn utiliser. `feature_columns()` et `build_preprocessing()` sont fournies.

Après chaque fonction : `python -m pytest`. Une fois tout écrit : `python -m monprojet.train` affiche le tableau de comparaison.

## 4. Choisir

Remplissez [docs/choix-modele_Yasser_Hadi.md](docs/choix-modele_Yasser_Hadi.md) : les résultats sur le jeu de test, le modèle retenu, l'écart avec la baseline, le critère d'acceptation atteint ou non.

## Ce que produit l'entraînement

| Fichier | Contenu |
|---|---|
| `models/model.joblib` | Le meilleur modèle selon la métrique principale |
| `models/metrics.json` | `main_metric`, `best`, et les scores de tous les modèles |

## Organisation du rendu et collaboration

Le dépôt commun contient `TP/TP0/` pour le projet terminé et `TP/TP1/` pour le prochain TP. Toutes les commandes Python ci-dessus sont à exécuter depuis `TP/TP0/`, et les commandes Git depuis la racine du dépôt. La seule branche de travail du groupe est `TP`.

Les documents individuels de Yasser sont [le choix du modèle](docs/choix-modele_Yasser_Hadi.md) et [l'étude de conception](docs/etude-de-conception_Yasser_Hadi.md). Chaque étudiant conserve ses propres documents avec le suffixe `_Prenom_Nom`, sans modifier ceux de son binôme. Les commits doivent être réalisés avec l'identité Git réelle de leur auteur. Le responsable publie le projet commun ; le binôme ajoute ou modifie uniquement ses documents dans des commits distincts.

Les données, les modèles sauvegardés, les environnements virtuels, les caches et les secrets restent locaux et sont exclus par le `.gitignore` à la racine. Après un clonage, créer localement `TP/TP0/data/` et y placer le fichier autorisé `exemple_pannes.csv`, obtenu séparément : Git ne le fournit pas. Conserver tout autre dataset dans un dossier `data/` ou `datasets/` ignoré, même si son extension est inhabituelle. Ne pas contourner ces exclusions avec `git add -f`.

Le modèle et `metrics.json` ne sont pas distribués par Git ; les scores du rendu figurent dans les documents individuels. Les tests qui chargent le CSV exigent sa présence locale. Deux tests de `tests/test_train.py` appellent `main()` : une exécution standard de toute la suite réentraîne les candidats et remplace les artefacts locaux dans `models/`.

L'environnement `.venv` existant est conservé localement. Après le déplacement du projet, sélectionner `TP/TP0/.venv/Scripts/python.exe` dans VS Code et utiliser `python -m pip` plutôt que le lanceur `pip.exe`, qui peut conserver un ancien chemin absolu. Sur un nouveau poste, créer un nouvel environnement avec les commandes d'installation ci-dessus.

Avant chaque commit, contrôler `git status`, `git diff --cached` et `git check-ignore -v TP/TP0/data/exemple_pannes.csv TP/TP0/models/model.joblib TP/TP0/models/metrics.json` depuis la racine. Avant de travailler, synchroniser avec `git pull --ff-only origin TP`. Avant de publier un commit local, récupérer les commits distants avec `git pull --rebase origin TP`, résoudre les éventuels conflits, puis utiliser `git push origin TP`. Ne pas utiliser de push forcé sur cette branche commune.
