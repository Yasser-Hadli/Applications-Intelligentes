# Choix du modèle

Date : [28/10/2026] · Groupe : [Yasser Hadli]

## Contexte

Le projet vise à classifier des observations de fonctionnement industriel selon la présence (`failure=1`) ou l'absence (`failure=0`) d'une panne, afin d'éclairer la priorisation des contrôles de maintenance. Conformément à la section 5 de l'étude de conception et à `MAIN_METRIC="recall"`, le rappel constitue la métrique principale, car un faux négatif correspond à une panne réelle non détectée, potentiellement coûteuse pour l'exploitation.

L'expérimentation reste une classification hors ligne : le fichier ne définit pas d'horizon temporel d'anticipation et le prototype ne déclenche pas d'intervention sur un équipement.

## Données et découpage

Le fichier `data/exemple_pannes.csv`, fourni comme exemple pédagogique, contient 5 000 lignes. Les variables explicatives sont `temperature_c`, `vibration_mm_s`, `load_pct` et `runtime_hours` ; la cible est `failure`. La classe positive comprend 878 observations (17,56 %), contre 4 122 observations sans panne (82,44 %). Aucune valeur manquante ni ligne entièrement dupliquée n'a été relevée dans ce fichier. La provenance détaillée et la licence restent à préciser.

Le partage est aléatoire et stratifié sur la cible, avec `TEST_SIZE=0.2` et `SEED=42`. Il réserve 4 000 lignes à l'entraînement, dont 702 pannes, et 1 000 lignes au test, dont 176 pannes et 824 observations sans panne. Aucune colonne de date n'est configurée (`TIME_COLUMN=None`) ; il ne s'agit donc pas d'une validation chronologique.

Chaque candidat reçoit les mêmes partitions et un pipeline de prétraitement numérique commun : imputation par la médiane, puis centrage et réduction. Ces transformations sont ajustées sur l'entraînement uniquement. L'imputation reste prévue pour des valeurs explicatives manquantes éventuelles, bien que le fichier actuel soit complet. Aucun rééquilibrage des classes ni réglage du seuil de décision n'est implémenté.

## Résultats sur le jeu de test

| Modèle | Métrique principale | Autres métriques |
|---|---|---|
| Baseline (`baseline`) | Recall : 0,000 | Accuracy : 0,824 ; Precision : 0,000 ; F1 : 0,000 |
| Logistic Regression (`logistic_regression`) | Recall : 0,574 | Accuracy : 0,889 ; Precision : 0,737 ; F1 : 0,645 |
| Random Forest (`random_forest`) | Recall : 0,545 | Accuracy : 0,873 ; Precision : 0,671 ; F1 : 0,602 |
| Gradient Boosting (`gradient_boosting`) | Recall : 0,608 | Accuracy : 0,888 ; Precision : 0,713 ; F1 : 0,656 |

Les valeurs du tableau proviennent exclusivement de `models/metrics.json`, où `main_metric` vaut `recall` et `best` vaut `gradient_boosting`. Les scores y sont arrondis à trois décimales. La précision, le rappel et le F1 concernent la classe positive `1` ; l'accuracy concerne l'ensemble du test. Aucun nouvel entraînement n'a été réalisé pour cette rédaction.

La baseline est un `DummyClassifier(strategy="most_frequent")`. Elle prédit toujours l'absence de panne et classe donc correctement les 824 observations négatives du test, sans détecter les 176 positives. Son accuracy de 82,4 % montre pourquoi cette métrique seule serait trompeuse sur un jeu déséquilibré. Sa précision est fixée à zéro par `zero_division=0` en l'absence de prédictions positives ; ce score traduit ici l'absence d'alertes.

## Décision

Le modèle retenu est **Gradient Boosting**, car son rappel de **0,608, soit 60,8 %**, est le plus élevé des quatre candidats. Cette décision correspond au classement décroissant selon `MAIN_METRIC` dans `train.py` et au nom enregistré dans `metrics.json`. Le code prévoit la sauvegarde du pipeline gagnant dans `models/model.joblib`.

Sur la métrique principale, l'écart atteint 60,8 points de pourcentage par rapport à la baseline, 3,4 points par rapport à la régression logistique et 6,3 points par rapport à Random Forest. Ces écarts sont calculés à partir des scores enregistrés ; le gain relatif sur la baseline n'est pas défini puisque son rappel est nul.

La sélection implique un compromis. La régression logistique présente la meilleure accuracy (0,889 contre 0,888) et la meilleure précision (0,737 contre 0,713) : une plus grande proportion de ses alertes correspond à des pannes réelles, mais elle détecte une proportion plus faible des pannes existantes. Gradient Boosting gagne 3,4 points de rappel en cédant 2,4 points de précision et 0,1 point d'accuracy. Son F1 de 0,656 dépasse celui de la régression logistique (0,645), ce qui traduit un meilleur compromis harmonique précision-rappel sur ce test, sans constituer un calcul de coût industriel.

Random Forest obtient des scores inférieurs à ceux de Gradient Boosting sur les quatre métriques enregistrées. La baseline remplit son rôle de référence minimale ; son accuracy élevée ne compense pas son rappel nul. Pour l'objectif choisi, le rappel supérieur de Gradient Boosting justifie donc sa sélection, même s'il n'est pas premier sur chaque indicateur.

Concernant l'acceptation, les documents originaux demandent d'en discuter mais ne donnent aucun seuil numérique obligatoire. La règle de sélection implémentée est satisfaite : Gradient Boosting maximise le rappel parmi les candidats et dépasse la baseline. En revanche, aucun critère chiffré d'acceptation industrielle n'ayant été défini, il est impossible d'affirmer qu'un tel critère est atteint. Les huit tests Pytest annoncés réussis attestent les comportements logiciels vérifiés, sans valider cette acceptabilité métier.

## Limites

Un rappel de 60,8 % laisse environ **39,2 % des pannes non détectées**, d'après le score arrondi. Il est supérieur à celui des autres modèles testés, mais reste limité lorsque certaines pannes ne doivent pas être manquées. La précision de 71,3 % indique aussi qu'une partie des alertes est erronée. Sans connaître les conséquences des types de panne et les coûts d'inspection, ces scores ne suffisent pas pour autoriser une décision automatique ou garantir la sécurité des équipements.

Le test comprend 1 000 observations, dont seulement 176 pannes. Un unique découpage ne mesure pas la variabilité des performances ; aucun intervalle de confiance ni test de significativité n'est fourni. Les écarts observés, notamment celui entre les deux meilleurs rappels, doivent donc être interprétés avec prudence. De plus, ce même test sert à sélectionner le modèle : une évaluation finale indépendante serait nécessaire pour réduire le biais lié à cette sélection.

L'absence de valeurs manquantes dans le fichier ne garantit pas que les capteurs réels fourniront des mesures complètes ou comparables. Les quatre variables ne renseignent ni le type d'équipement, ni le mode de panne, ni l'environnement d'exploitation. Sans identifiant machine, date et documentation de collecte, il est impossible de vérifier les dépendances entre lignes, la couverture des cas rares et la généralisation à d'autres installations. La durée de fonctionnement n'est pas un horodatage permettant à elle seule une validation temporelle.

Les améliorations suivantes sont proposées pour une étude ultérieure ; elles ne sont pas implémentées et aucun gain n'est revendiqué :

- **Renforcer le protocole d'évaluation** : réserver un test final indépendant et comparer les candidats par validation croisée stratifiée sur les données de développement. Si des identifiants ou dates deviennent disponibles, privilégier une séparation par machine ou une validation temporelle selon l'usage visé, afin de limiter les fuites entre observations liées.
- **Étudier le seuil de décision** : analyser les probabilités prédites et la courbe précision-rappel sur des données de validation, puis choisir un seuil en fonction du coût des pannes manquées et des fausses alertes. Un seuil plus bas peut augmenter le rappel au prix d'une baisse de précision ; l'effet doit être mesuré avant toute conclusion.
- **Prendre en compte le déséquilibre** : évaluer des pondérations de classes pour les estimateurs compatibles ou un rééchantillonnage limité aux plis d'entraînement. Les partitions de validation et de test devraient conserver une distribution représentative pour ne pas fausser l'évaluation.
- **Ajuster les hyperparamètres** : rechercher notamment la profondeur et le nombre d'arbres ainsi que le taux d'apprentissage du Gradient Boosting, en utilisant exclusivement les données de développement. Une validation imbriquée peut être envisagée pour évaluer l'ensemble de cette procédure de sélection.
- **Documenter et enrichir les données** : préciser la source, la licence, l'étiquetage et l'horizon de panne, recueillir des cas rares et analyser les faux négatifs avec les spécialistes de maintenance. Des variables temporelles ne devraient être construites qu'à partir d'informations effectivement disponibles au moment de la prédiction.
- **Définir l'acceptation avec le métier** : établir un rappel minimal et une charge de fausses alertes supportable selon la criticité des équipements, puis vérifier ces conditions sur un jeu représentatif indépendant. Ces seuils restent à déterminer et ne constituent pas des exigences chiffrées du professeur.
