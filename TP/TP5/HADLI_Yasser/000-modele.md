# ADR 000 : choix d'un modèle de prédiction des pannes

Date : 25/11/2026 · Auteur : Yasser Hadli vers 18H00MIN

## Contexte
MaintAI doit estimer le risque de panne à partir de quatre relevés de capteurs : température, vibration, charge et durée de fonctionnement. Les pannes sont moins fréquentes que les cas sans panne ; l'accuracy seule ne suffit donc pas à juger le modèle. Une panne manquée peut coûter plus cher qu'une fausse alerte.

Les données disponibles sont synthétiques. Le choix reste provisoire tant qu'il n'a pas été validé sur des données réelles et selon les coûts métier.

## Options étudiées
1. **Classe majoritaire** : toujours prédire l'absence de panne. Cette baseline est simple à comprendre, mais ne détecte aucune panne ; son rappel est nul.
2. **Régression logistique avec standardisation** : rapide à entraîner, relativement simple à interpréter et capable de fournir une probabilité pour régler le seuil d'alerte. Elle suppose une relation linéaire entre les variables standardisées et le log-rapport des probabilités ; des relations plus complexes peuvent lui échapper.
3. **Forêt aléatoire** : peut représenter des relations non linéaires, mais est plus complexe à interpréter et coûte davantage à entraîner et à stocker.

## Décision
Retenir provisoirement une régression logistique scikit-learn dans un pipeline avec `StandardScaler`. Le prétraitement et le classifieur sont ainsi sauvegardés ensemble. Le seuil de décision peut être ajusté pour privilégier le rappel plutôt que la seule accuracy.

Sur les quatre essais comparés sur la première version synthétique des données, la régression logistique avec `C=1.0` et un seuil de 0,2 obtient un rappel de 0,838 et une précision de 0,543. Ce réglage, ses limites et les résultats complémentaires sont documentés dans [ADR 004](004-choix-du-modele.md).

## Conséquences
- Le pipeline facilite la reproduction de la prédiction et évite d'oublier la standardisation lors de l'utilisation du modèle.
- Un seuil plus bas permet de détecter davantage de pannes, mais augmente le nombre de fausses alertes. Il faut suivre le rappel et la précision, puis mesurer les coûts réels avant de fixer définitivement le seuil.
- Le modèle, son contrat d'entrée et les versions des dépendances doivent être conservés ensemble. MLflow suit les paramètres, métriques et artefacts ; DVC permet d'identifier les données utilisées.
- Les résultats sur données synthétiques ne démontrent pas la performance en production. Une évaluation sur des données réelles indépendantes et une surveillance des performances restent nécessaires.
