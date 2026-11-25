# ADR 004 : choix du modèle et du seuil d'alerte

Date : 25/11/2026 · Auteurs : Mohamed BENHAYOUN SADAFI

## Contexte

MaintAI doit lever une alerte avant qu'une machine tombe en panne. Une panne manquée (faux négatif) coûte plus cher qu'une fausse alerte (faux positif) : la première arrête la production et impose une réparation, la seconde coûte une inspection inutile. Jusqu'au TP4, le modèle et le seuil étaient choisis sans trace ; depuis le TP5, chaque entraînement est un run MLflow, et les données sont versionnées par DVC.

Les six runs comparés ont été entraînés sur la **même version des données** : la v1 suivie par DVC (`data_md5 = 73e0b7cb3c6f795c70c3ddf970b99c15`, 5 000 relevés). Ils sont évalués sur le même jeu de test : 1 000 relevés, dont 136 pannes.

## Options étudiées

| Run MLflow | Modèle | C | Seuil | recall | precision | accuracy | Pannes manquées | Fausses alertes |
|---|---|---|---|---|---|---|---|---|
| `f0a233bc` | régression logistique | 1.0 | 0.3 | 0.750 | 0.622 | 0.904 | 34 | 62 |
| `9a23ae38` | régression logistique | 0.01 | 0.3 | 0.691 | 0.653 | 0.908 | 42 | 50 |
| **`09570622`** | **régression logistique** | **1.0** | **0.2** | **0.838** | **0.543** | **0.882** | **22** | **96** |
| `1775edba` | régression logistique | 1.0 | 0.15 | 0.875 | 0.500 | 0.864 | 17 | 119 |
| `7eb12597` | forêt aléatoire (200 arbres) | – | 0.3 | 0.728 | 0.562 | 0.886 | 37 | 77 |
| `073572e3` | forêt aléatoire (200 arbres) | – | 0.2 | 0.831 | 0.479 | 0.854 | 23 | 123 |

La baseline (toujours « pas de panne ») a une accuracy de 0.864 et un recall de 0 : elle ne détecte aucune panne.

1. **Régression logistique, seuil 0.3 (réglage du TP4)** : la meilleure accuracy, des alertes plutôt fiables (62 % sont de vraies pannes), mais une panne sur quatre est manquée (34 sur 136).
2. **Régression logistique, C = 0.01** : plus de régularisation. Un peu moins de fausses alertes, mais 42 pannes manquées : c'est le plus mauvais recall, donc le pire choix pour MaintAI.
3. **Régression logistique, seuil 0.2** : 12 pannes de plus détectées que l'option 1 (22 manquées au lieu de 34), en échange de 34 fausses alertes de plus (96 au lieu de 62).
4. **Régression logistique, seuil 0.15** : encore 5 pannes de plus détectées (17 manquées), mais 23 fausses alertes de plus (119). La precision tombe à 0.50 (une alerte sur deux est fausse) et l'accuracy (0.864) n'est plus meilleure que celle de la baseline.
5. **Forêt aléatoire** : moins bonne que la régression logistique au même seuil, sur les deux tableaux à la fois. Au seuil 0.2, elle manque 23 pannes (contre 22) et donne 123 fausses alertes (contre 96). Elle est aussi plus lourde (200 arbres) et moins lisible.

## Décision

**Régression logistique (C = 1.0) avec un seuil d'alerte de 0.2.**

- Run MLflow : `095706227bc048438f7e4d27eae5498a` (expérience `maintai`, nom `amusing-panda-382`)
- Données : v1 sous DVC, `data_md5 = 73e0b7cb3c6f795c70c3ddf970b99c15` (5 000 relevés, `n_train = 4000`)
- Versions : scikit-learn 1.8.0, enregistrée avec le modèle dans son `requirements.txt` MLflow

La contrainte qui justifie le choix est le coût d'une panne manquée. Passer de 0.3 à 0.2 coûte environ 2,8 fausses alertes par panne détectée en plus (34 / 12) : c'est rentable dès qu'une panne coûte plus que trois inspections, ce qui est le cas pour un arrêt de machine. Descendre à 0.15 coûterait 4,6 fausses alertes par panne détectée en plus (23 / 5), et une alerte sur deux serait fausse. Les opérateurs risqueraient alors de ne plus croire les alertes. Le seuil 0.2 est le compromis retenu. La forêt aléatoire est écartée : elle n'est meilleure sur aucune métrique.

## Conséquences

- **Ce que nous acceptons :** davantage de fausses alertes. Sur 1 000 relevés, 96 inspections inutiles au lieu de 62, et 46 % des alertes sont fausses (precision 0.543). En échange, 84 % des pannes sont détectées au lieu de 75 %.
- **Mise en service :** `predict.py` lit le seuil dans `models/model_info.json`. Le modèle déployé doit donc être entraîné avec `python -m maintai.train --threshold 0.2` (ou `MAINTAI_THRESHOLD=0.2`), sinon l'API garde le seuil par défaut de 0.3. Le modèle exact peut aussi être rechargé depuis MLflow : `mlflow.sklearn.load_model("runs:/095706227bc048438f7e4d27eae5498a/model")`.
- **Reproductibilité :** les données du run se retrouvent par leur empreinte. On repère le commit « Données v1 » dont le reçu `data/sensors.csv.dvc` contient `md5: 73e0b7cb…`, puis `git checkout <commit> -- data/sensors.csv.dvc` et `dvc checkout` (ou `dvc pull`).
- **À surveiller :** le seuil a été choisi sur le jeu de test lui-même, donc les chiffres sont un peu optimistes. Il faudrait le régler sur un jeu de validation, puis le confirmer sur des données nouvelles. Une v2 des données existe (`93fdeef2…`, 8 000 relevés) : avant de changer de modèle, il faut refaire cette comparaison sur un même jeu de test. En production, il faut suivre le recall réel (les pannes que le modèle n'a pas signalées) et la dérive des données, puis ré-entraîner sur une version DVC identifiée.
- **À revoir si** le coût réel d'une panne et d'une inspection est connu : si une panne coûte plus d'environ cinq inspections, le seuil 0.15 devient préférable.
