# ADR 004 — Choix du modèle et du seuil



## Contexte

Une panne non détectée coûte plus cher qu'une fausse alerte. Les quatre modèles ont été évalués sur la même version synthétique de 5 000 relevés.

## Décision

Retenir provisoirement la régression logistique avec `C=1.0` et un seuil d'alerte de `0.2`.

- Run MLflow : `e63993f2e7dd4151a536dad6bcfdcd8a`.
- Empreinte des données : `73e0b7cb3c6f795c70c3ddf970b99c15`.
- Rappel : 0,838 ; précision : 0,543 ; accuracy : 0,882.

## Conséquences

Ce réglage a le meilleur rappel des quatre essais et manque moins de pannes, au prix de plus nombreuses fausses alertes. Les données sont synthétiques : il faut confirmer le choix sur des données réelles et selon les coûts métier.

Une comparaison complémentaire sur 5 000 relevés synthétiques indépendants donne, à ce seuil, 423 fausses alertes et 121 pannes manquées pour ce modèle v1. Le modèle v2 (`8c8eaa344a6c43f7b3689e6ef6eebc4a`, données `93fdeef2cac1893c4f720c16457eea75`) produit 391 fausses alertes et manque 128 pannes. Le choix de la v1 favorise donc le rappel, mais reste provisoire.

Pour la v2, passer du seuil 0,3 à 0,2 évite 44 pannes manquées et ajoute 152 fausses alertes sur ce jeu. En supposant des coûts unitaires constants et en ignorant les autres coûts, le seuil 0,2 est préférable si une panne manquée coûte plus de 3,45 fois une fausse alerte. Les coûts réels ne sont pas connus ; ils doivent être mesurés avant une décision définitive.
