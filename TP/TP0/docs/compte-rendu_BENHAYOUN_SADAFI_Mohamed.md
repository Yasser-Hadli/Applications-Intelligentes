# Compte rendu du TP0 — Projet reproductible et comparaison de modèles

Auteur : BENHAYOUN SADAFI Mohamed · Groupe : 5IAD G1 · Coordinateur : Yasser Hadli · Module : Développement d'applications intelligentes · Date : 18/11/2026

## 1. Objectifs

Ce TP correspond à la séance 3. Il poursuit deux objectifs :

1. comprendre ce qui rend un projet reproductible, à partir du notebook de Karim ;
2. compléter une chaîne de Machine Learning pour prédire des pannes (cas MaintAI), comparer plusieurs modèles à une baseline et justifier le choix final.

Ce document est un compte rendu individuel. Le projet commun se trouve dans `TP/TP0/` du dépôt du groupe. L'étude de conception et le choix du modèle sont rédigés par le coordinateur dans `etude-de-conception_Yasser_Hadi.md` et `choix-modele_Yasser_Hadi.md`.

## 2. Pourquoi le projet de Karim n'est pas reproductible

Karim affirme obtenir 14,2 % d'alertes, mais son notebook `analyse.ipynb` ne permet pas de retrouver ce résultat :

| Problème constaté | Conséquence | Bonne pratique |
|---|---|---|
| Chemin absolu `C:/Users/Karim/Desktop/PFE/donnees_capteurs.csv` | Le fichier est introuvable sur une autre machine | Chemin construit dans `config.py` |
| Le fichier fourni s'appelle `data/capteurs.csv` | Même en corrigeant le dossier, le nom ne correspond pas | Un seul chemin de données, documenté |
| Aucune liste de dépendances ni de versions | Bibliothèques absentes ou versions différentes | `.venv` et `requirements.txt` avec versions épinglées |
| `seaborn` est importé sans être utilisé | Dépendance inutile, qui peut bloquer l'exécution | Garder seulement les imports utiles |
| Cellules dans le désordre : le modèle utilise `X_train` avant le découpage | « Tout exécuter » échoue | Script lancé avec `python -m` |
| `colonnes` et `seuil` ne sont définis nulle part | `NameError`, et le résultat dépend d'un seuil inconnu | Réglages écrits dans `config.py` |
| `train_test_split` sans graine | Le résultat change à chaque exécution | Graine fixe (`SEED`) |
| Aucun test | Personne ne peut vérifier que le résultat est bon | Tests avec `pytest` |

## 3. Mise en place du projet

Le projet s'installe dans un environnement virtuel, depuis sa racine :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
python -m monprojet.train
```

La commande `python monprojet/train.py` échoue avec `ModuleNotFoundError: No module named 'monprojet'`. Lancé ainsi, le script ne trouve pas le paquet depuis la racine du projet. Il faut utiliser `python -m monprojet.train`.

## 4. Données

Le fichier `data/exemple_pannes.csv` contient 5 000 relevés synthétiques. Les quatre variables d'entrée sont `temperature_c`, `vibration_mm_s`, `load_pct` et `runtime_hours`. La cible est `failure`.

Il y a 878 pannes (17,56 %) et 4 122 relevés sans panne (82,44 %). Les classes sont donc déséquilibrées. Le fichier ne contient ni valeur manquante ni ligne dupliquée.

## 5. Chaîne de traitement

| Fichier | Fonction | Rôle |
|---|---|---|
| `monprojet/data.py` | `load_data()` | Lit le CSV, vérifie que la cible existe, retire les lignes sans cible |
| `monprojet/data.py` | `split()` | Sépare les variables et la cible, puis découpe les données |
| `monprojet/models.py` | `candidates()` | Définit la baseline et les modèles à comparer |
| `monprojet/train.py` | `evaluate()` | Calcule accuracy, rappel, précision et F1 sur le test |
| `monprojet/train.py` | `main()` | Entraîne, compare, puis sauvegarde le meilleur modèle et les scores |

Le découpage est aléatoire et stratifié, avec 20 % des données en test et la graine 42. On obtient 4 000 lignes d'entraînement (702 pannes) et 1 000 lignes de test (176 pannes).

Les variables numériques passent par une imputation par la médiane, puis par `StandardScaler`. Ces étapes sont placées dans un `Pipeline` avec le modèle : elles apprennent seulement sur l'entraînement, ce qui évite une fuite d'information venant du test.

Le projet commun compare quatre candidats :

- **Baseline** : `DummyClassifier(strategy="most_frequent")` ;
- **Régression logistique** : `max_iter=1000` ;
- **Random Forest** : `n_estimators=100` ;
- **Gradient Boosting** : paramètres par défaut.

## 6. Résultats du projet commun

| Modèle | Accuracy | Rappel | Précision | F1 | Pannes détectées | Pannes manquées | Fausses alertes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0,824 | 0,000 | 0,000 | 0,000 | 0 | 176 | 0 |
| Régression logistique | 0,889 | 0,574 | 0,737 | 0,645 | 101 | 75 | 36 |
| Random Forest | 0,873 | 0,545 | 0,671 | 0,602 | 96 | 80 | 47 |
| **Gradient Boosting** | 0,888 | **0,608** | 0,713 | 0,656 | 107 | 69 | 43 |

À retenir de ce tableau :

- **L'accuracy trompe ici.** La baseline atteint 82,4 % sans détecter une seule panne, simplement parce que la plupart des relevés sont sans panne.
- **Le rappel décide.** Gradient Boosting détecte le plus de pannes : 107 sur 176, soit 60,8 %.
- **Le compromis est favorable.** Par rapport à la régression logistique, il détecte 6 pannes de plus pour 13 inspections de plus. Une panne manquée coûte une journée d'arrêt et une fausse alerte une heure d'inspection.

## 7. Essai complémentaire

La version locale du projet garde les mêmes données, le même découpage et les mêmes métriques, avec deux variantes testées :

| Modèle testé localement | Accuracy | Rappel | Précision | F1 | Pannes détectées | Fausses alertes |
|---|---:|---:|---:|---:|---:|---:|
| Arbre de décision (`max_depth=5`) | 0,873 | 0,540 | 0,674 | 0,599 | 95 | 46 |
| Forêt aléatoire (`n_estimators=200`) | 0,881 | 0,557 | 0,705 | 0,622 | 98 | 41 |

Passer de 100 à 200 arbres améliore un peu la forêt : 2 pannes détectées en plus et 6 fausses alertes en moins. Cela ne change pas la conclusion, car Gradient Boosting garde le meilleur rappel.

## 8. Le modèle est-il suffisant ?

Le dépôt commun ne fixe pas de seuil chiffré d'acceptation. Pour juger le résultat, ce compte rendu reprend l'exemple du polycopié pour MaintAI : rappel ≥ 0,75 et précision ≥ 0,50 sur le jeu de test.

| Critère sur le jeu de test | Résultat de Gradient Boosting | Atteint ? |
|---|---|---|
| Rappel ≥ 0,75 | 0,608 | Non |
| Précision ≥ 0,50 | 0,713 | Oui |
| Rappel supérieur à la baseline | 0,608 contre 0,000 | Oui |

Pour atteindre 0,75, il faudrait détecter au moins 132 pannes sur 176, soit 25 de plus. Le modèle apporte donc une vraie amélioration, mais il ne suffit pas pour un usage réel.

## 9. Vérification

La version locale passe les tests : **8 passed**, le 18/11/2026, sur une copie du projet avec Python 3.12.6 et les versions de `requirements.txt`. L'entraînement relancé dans cette copie a redonné le même `metrics.json`.

La Random Forest à 100 arbres du projet commun a aussi été reproduite avec le code local : elle donne les mêmes scores que ceux publiés par le coordinateur. Le code du dépôt commun n'a pas été exécuté ici ; son résultat de tests (8 passed) est celui indiqué dans le compte rendu du coordinateur.

Ces tests vérifient que le code se comporte comme prévu. Ils ne prouvent pas que le modèle est assez bon pour une usine.

## 10. Limites et améliorations

- 39,2 % des pannes du test restent non détectées.
- L'évaluation repose sur un seul découpage, avec seulement 176 pannes en test.
- Le jeu de test sert aussi à choisir le modèle, ce qui peut rendre le score un peu optimiste.
- Les données sont synthétiques et n'ont ni date ni identifiant de machine.

Pour la suite, il faudrait créer un jeu de validation, y régler le seuil de décision, tester `class_weight="balanced"` et ajuster les hyperparamètres. Le jeu de test serait alors gardé pour la mesure finale.

## 11. Conclusion

Ce TP montre qu'un résultat n'a de valeur que si quelqu'un d'autre peut le reproduire : environnement, versions, chemins, graine et tests comptent autant que le modèle. Il montre aussi que la métrique doit venir du coût des erreurs. Gradient Boosting est le meilleur candidat avec un rappel de 60,8 %, mais il reste sous l'objectif de 75 % : il faut encore l'améliorer avant de s'en servir pour décider des inspections.
