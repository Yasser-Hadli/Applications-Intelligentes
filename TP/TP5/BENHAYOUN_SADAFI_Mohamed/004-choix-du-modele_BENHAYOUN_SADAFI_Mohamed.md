# ADR 004 : choix du modèle et du seuil d'alerte

Date : 25/11/2026 · Auteurs : Mohamed BENHAYOUN SADAFI

## Contexte

MaintAI doit lever une alerte avant qu'une machine tombe en panne. Une panne manquée (faux négatif) coûte plus cher qu'une fausse alerte (faux positif) : la première arrête la production et impose une réparation, la seconde coûte une inspection inutile. Depuis le TP5, chaque entraînement est un run MLflow, et les données sont versionnées par DVC : le choix peut être justifié par des runs identifiés, entraînés sur une version connue des données.

Une première comparaison (sprint 1, données v1) avait retenu la régression logistique au seuil 0.2. Une grid search de **30 runs** a ensuite été menée sur les données **v2** (commit `ce83c1c`, `data_md5 = 93fdeef2cac1893c4f720c16457eea75`, 8 000 relevés). Tous ses modèles sont évalués sur le même jeu de test : 1 600 relevés, dont 205 pannes.

**Critère de choix :** le plus grand recall parmi les modèles dont la precision est ≥ 0.5, c'est-à-dire qu'au moins une alerte sur deux est une vraie panne.

## Options étudiées

1. **Régression logistique (C = 1.0)** : simple, rapide et lisible. Son seul réglage utile est `C` ; elle n'a pas de learning rate. Sur la v2 : seuil 0.15, recall 0.859 et precision 0.506 (176 pannes détectées, 172 fausses alertes) ; seuil 0.2, recall 0.824 et precision 0.567 (169 détectées, 129 fausses alertes).
2. **Forêt aléatoire** : écartée dès le sprint 1. Au même seuil, elle était moins bonne que la régression logistique en recall comme en precision.
3. **Réseau de neurones (learning rate + dropout)** : non retenu. MaintAI n'utilise pas de réseau de neurones, et le dropout n'existe que dans ce type de modèle. Il aurait fallu installer PyTorch et réécrire l'entraînement.
4. **Gradient boosting (100 arbres) : grid search sur learning rate × seuil**, soit 3 learning rates (0.05, 0.1, 0.2) × 10 seuils (0.1 à 0.35) = 30 runs, marqués `grid = tp5` dans MLflow. Les 6 meilleurs :

| Rang | Run | learning_rate | Seuil | recall | precision | accuracy | Pannes manquées | Fausses alertes |
|---|---|---|---|---|---|---|---|---|
| **1** | **`341e1560`** | **0.05** | **0.15** | **0.868** | **0.500** | **0.872** | **27** | **178** |
| 2 | `d25018b1` | 0.1 | 0.15 | 0.854 | 0.515 | 0.878 | 30 | 165 |
| 3 | `8dcdb6dd` | 0.2 | 0.15 | 0.844 | 0.524 | 0.882 | 32 | 157 |
| 4 | `4a429e18` | 0.05 | 0.175 | 0.839 | 0.526 | 0.882 | 33 | 155 |
| 5 | `11dcf0f2` | 0.1 | 0.175 | 0.829 | 0.528 | 0.883 | 35 | 152 |
| 6 | `9525a103` | 0.05 | 0.2 | 0.815 | 0.559 | 0.894 | 38 | 132 |

Les seuils 0.1 et 0.125 donnent un recall plus élevé (jusqu'à 0.932), mais leur precision passe sous 0.5 : ils sont éliminés par le critère.

## Décision

**Gradient boosting (`GradientBoostingClassifier`, 100 arbres), learning_rate = 0.05, seuil d'alerte 0.15.**

- Run MLflow : `341e156009fc48629f5ae70a29c7a912` (expérience `maintai`, tag `grid = tp5`)
- Données : v2 sous DVC, `data_md5 = 93fdeef2cac1893c4f720c16457eea75` (commit `ce83c1c`, `n_train = 6400`)
- Métriques : recall 0.868 (178 pannes sur 205), precision 0.500, accuracy 0.872
- Versions : scikit-learn 1.8.0, enregistrée avec le modèle dans son `requirements.txt` MLflow

**Pourquoi ce modèle :**
- C'est le meilleur recall de toute la grille parmi les modèles qui respectent la precision ≥ 0.5 : il ne manque que 27 pannes sur 205.
- Le learning rate le plus petit (0.05) gagne à chaque seuil bas. Avec 100 arbres, il corrige les erreurs plus progressivement et généralise mieux.
- Il fait mieux que la meilleure régression logistique sur les mêmes données et le même jeu de test : 2 pannes détectées de plus qu'elle au seuil 0.15, et 9 de plus qu'au seuil 0.2, l'ancien choix.
- Le seuil reste le réglage qui compte le plus. À learning rate fixe, passer de 0.35 à 0.15 fait gagner environ 20 points de recall, alors que les learning rates ne diffèrent que de 1 à 3 points à seuil égal.

## Conséquences

- **Ce que nous acceptons :** beaucoup de fausses alertes. 178 inspections inutiles pour 1 600 relevés : une alerte sur deux est fausse. C'est le prix pour ne manquer que 13 % des pannes.
- **Un gain petit et fragile :** l'avantage sur la régression logistique est de 2 pannes sur 205. La precision du run retenu est pile à la limite (178 / 356 = 0.500) : sur de nouvelles données, elle peut passer en dessous. **Solution de repli :** le run n° 2 (`d25018b1`, learning_rate 0.1, seuil 0.15 : recall 0.854, precision 0.515), qui garde plus de marge.
- **Le modèle est moins lisible :** 100 arbres au lieu de 4 coefficients. On ne peut plus expliquer simplement le poids de chaque capteur.
- **Mise en service :** `predict.py` lit le seuil dans `models/model_info.json`. Il faut donc entraîner le modèle déployé avec `python -m maintai.train --model gboost --lr 0.05 --threshold 0.15`, ou recharger le modèle exact depuis MLflow : `mlflow.sklearn.load_model("runs:/341e156009fc48629f5ae70a29c7a912/model")`.
- **Reproductibilité :** on retrouve les données du run par son `data_md5` : `git checkout ce83c1c -- data/sensors.csv.dvc` puis `dvc checkout` (ou `dvc pull`).
- **À surveiller :** le seuil a été choisi sur le jeu de test lui-même, donc les chiffres sont optimistes. Il faudrait le régler sur un jeu de validation, puis le confirmer sur des données nouvelles. En production, il faut suivre le recall réel (les pannes non signalées), la part de fausses alertes et la dérive des données. Il faudra ré-entraîner sur une version DVC identifiée si l'une d'elles se dégrade.
