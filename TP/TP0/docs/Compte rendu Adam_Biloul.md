# Compte rendu du TP0 — Classification des pannes industrielles

Étudiant : Adam Biloul  
Groupe : Yasser Hadli et Adam Biloul · Version finale · Date : 28/10/2026

## 1. Contexte et objectifs du travail

Réalisé dans le module Applications Intelligentes, ce TP0 vise à construire une chaîne de Machine Learning capable de classer des observations industrielles selon la présence ou l'absence d'une panne. Le projet pédagogique sert de base pour compléter les étapes de lecture des données, de séparation des ensembles, de définition des algorithmes et d'évaluation des performances. Le choix final doit ensuite être expliqué à partir des résultats obtenus.

L'étude confronte un modèle de référence à trois méthodes de classification supervisée, en appliquant un protocole commun et reproductible. Le rappel est le critère prioritaire : une panne qui échappe à la détection peut perturber la maintenance et la production. Cette réalisation constitue un prototype évalué hors ligne. Elle ne pilote pas de machine et ne précise pas de délai d'anticipation avant une panne.

## 2. Description du jeu de données

Les données sont stockées dans le fichier local `data/exemple_pannes.csv`. Celui-ci rassemble 5 000 observations réparties sur cinq colonnes :

| Colonne | Nature | Interprétation du nom |
|---|---|---|
| `temperature_c` | Variable explicative numérique | Température exprimée en degrés Celsius |
| `vibration_mm_s` | Variable explicative numérique | Niveau de vibration en millimètres par seconde |
| `load_pct` | Variable explicative numérique | Charge exprimée en pourcentage |
| `runtime_hours` | Variable explicative numérique | Nombre d'heures de fonctionnement |
| `failure` | Variable cible binaire | `0` pour une absence de panne, `1` pour une panne |

La classe sans panne regroupe 4 122 lignes, soit 82,44 % du jeu, contre 878 lignes avec panne, soit 17,56 %. Ce déséquilibre rend l'accuracy insuffisante pour juger seule la qualité de la détection. Selon l'inspection rapportée, le fichier ne comporte ni valeur manquante ni ligne entièrement dupliquée. Les éléments disponibles ne précisent toutefois pas l'origine détaillée des données, leur licence ou la méthode utilisée pour attribuer les étiquettes.

## 3. Construction de la chaîne de traitement

L'implémentation s'appuie sur quatre modules. Les réglages sont regroupés dans `config.py` : emplacement du CSV, cible, type de tâche, métrique principale et paramètres de séparation. Le module `data.py` prend en charge la lecture et le découpage, tandis que `models.py` déclare les modèles à comparer. Le module `train.py` assemble le prétraitement et les estimateurs, mesure leurs performances et détermine le candidat retenu.

La lecture du CSV est effectuée avec pandas dans `load_data()`. Cette fonction contrôle la présence de la cible et écarte les lignes dont cette dernière serait absente. Comme `FEATURES=None` et qu'aucune colonne temporelle n'est définie, les quatre variables explicatives sont utilisées.

Les données sont séparées aléatoirement en conservant la proportion des classes grâce à une stratification sur `failure`. Le test représente 20 % du jeu et la graine aléatoire est fixée à 42. L'ensemble d'entraînement contient ainsi 4 000 observations, dont 702 pannes, et le test en contient 1 000, dont 176 pannes.

Le traitement des variables numériques combine le remplacement des valeurs manquantes par la médiane et une standardisation avec `StandardScaler`. Bien que les données actuelles soient complètes, l'imputation permet de prendre en charge d'éventuelles valeurs absentes. Ces opérations sont intégrées à un `Pipeline` et ajustées uniquement à partir de l'entraînement. Le test reçoit ensuite les transformations apprises, ce qui évite que ses statistiques interviennent dans l'apprentissage du prétraitement.

## 4. Algorithmes étudiés et méthode d'évaluation

Les quatre approches suivantes sont comparées sur un ensemble de test identique :

- **Modèle de référence (baseline)** : `DummyClassifier(strategy="most_frequent")`, qui renvoie toujours la classe la plus représentée.
- **Régression logistique** : modèle linéaire configuré avec `max_iter=1000` et `random_state=42`.
- **Random Forest** : ensemble de 100 arbres, avec `random_state=42`, capable de modéliser des relations non linéaires.
- **Gradient Boosting** : méthode qui construit successivement des modèles pour corriger les erreurs précédentes ; `random_state=42` est fixé et les autres paramètres conservent leurs valeurs par défaut.

L'évaluation repose sur la fonction `evaluate()`, qui fournit l'accuracy, le rappel, la précision et le F1-score. Le rappel, la précision et le F1-score concernent la classe positive `failure=1`. Les valeurs sont arrondies à trois décimales et `zero_division=0` permet de gérer les divisions indéfinies. Le protocole ne comprend ni recherche d'hyperparamètres, ni pondération des classes, ni ajustement du seuil de classification.

## 5. Présentation et interprétation des résultats

Les scores ci-dessous sont repris du fichier `models/metrics.json`. Aucune nouvelle expérimentation n'a été menée pour produire cette reformulation.

| Modèle | Accuracy | Rappel | Précision | F1-score |
|---|---|---|---|---|
| Baseline | 0,824 | 0,000 | 0,000 | 0,000 |
| Régression logistique | 0,889 | 0,574 | 0,737 | 0,645 |
| Random Forest | 0,873 | 0,545 | 0,671 | 0,602 |
| Gradient Boosting | 0,888 | 0,608 | 0,713 | 0,656 |

En prédisant systématiquement une absence de panne, la baseline atteint 82,4 % d'accuracy. Son rappel de zéro indique cependant qu'elle ne détecte aucune panne. Ce résultat illustre l'effet du déséquilibre des classes et sert de point de comparaison pour mesurer l'intérêt des autres approches.

Avec un rappel de 60,8 %, Gradient Boosting se classe en tête pour le critère principal. Son avance est de 3,4 points de pourcentage sur la régression logistique et de 6,3 points sur Random Forest. Il est donc sélectionné par le protocole. Le pipeline retenu est enregistré par le code dans `models/model.joblib`, tandis que les scores comparatifs sont sauvegardés dans `models/metrics.json`.

La régression logistique conserve les meilleurs résultats en précision, avec 73,7 %, et en accuracy, avec 88,9 %. Gradient Boosting détecte une plus grande part des pannes, avec une précision de 71,3 % et une accuracy de 88,8 %. Il obtient aussi le F1-score le plus élevé, à 0,656. Cette dernière mesure décrit un équilibre entre précision et rappel, sans traduire directement le coût des erreurs dans un contexte industriel.

## 6. Vérification du logiciel et organisation des livrables

Le compte rendu source rapporte qu'après la réorganisation du projet, les huit tests Pytest existants ont réussi : **8 passed en 36,85 secondes**. Ils vérifient notamment le chargement et la séparation des données, la disponibilité de la baseline et des modèles candidats, le calcul de la métrique principale, la sauvegarde des résultats et la comparaison à la référence.

Deux de ces tests exécutent la fonction d'entraînement `main()`. Lors de cette vérification, les sorties ont été temporairement dirigées vers un autre emplacement afin de conserver les modèles et résultats déjà présents. Aucun fichier Python ni test n'a été modifié. La vérification des empreintes a confirmé que le dataset, les modèles et les résultats locaux avaient été préservés. Ces contrôles attestent du fonctionnement des comportements testés ; ils ne constituent pas une validation du modèle pour une utilisation industrielle.

Le travail du TP0 est regroupé dans `TP/TP0/`. Le répertoire `TP/TP1/`, prévu pour le TP suivant, contient un `.gitkeep`. À la racine, le `.gitignore` exclut les données, les modèles entraînés, les environnements virtuels, les caches et les fichiers contenant des secrets. Les sources `monprojet/data.py` et `monprojet/models.py` restent prévues pour le suivi Git.

L'organisation collaborative repose sur un dépôt partagé et la branche `TP`, avec des documents personnels et des commits permettant d'identifier les contributions de chaque étudiant. Les documents de Yasser déjà présents ont été renommés avec le suffixe `_Yasser_Hadi` selon la convention adoptée. Cette version reformulée est enregistrée dans `Compte rendu Adam_Biloul.md`.

## 7. Limites de l'étude et améliorations envisagées

Un rappel de 60,8 % implique qu'environ 39,2 % des pannes du test restent non détectées, sur la base du score arrondi. La première place dans cette comparaison ne suffit donc pas à établir que le système répond aux exigences de sécurité d'une installation. Les consignes initiales ne fixent aucun seuil chiffré d'acceptation. Pour un usage industriel, ces exigences devraient être définies avec les responsables métier en tenant compte de la gravité des pannes et du coût des fausses alertes.

Les résultats proviennent d'un unique découpage aléatoire. De plus, le même ensemble de test intervient dans la comparaison et dans le choix du modèle, et aucun intervalle de confiance n'est fourni. Faute de dates et d'identifiants de machines, il n'est pas possible de vérifier la capacité du modèle à se généraliser à d'autres équipements ou à des périodes ultérieures.

Plusieurs prolongements peuvent être envisagés : appliquer une validation croisée aux données de développement, réserver un test final indépendant, examiner le seuil de décision au moyen d'une courbe précision-rappel et rechercher de meilleurs hyperparamètres pour Gradient Boosting. L'analyse des faux négatifs avec des spécialistes de maintenance et une documentation plus complète de la collecte aideraient également à interpréter les performances. Ces pistes restent des propositions ; elles n'ont pas été mises en œuvre et aucun progrès chiffré n'est annoncé.

## 8. Bilan

Ce TP0 a permis de mettre en place une chaîne de classification supervisée et d'en vérifier le fonctionnement logiciel, puis de confronter trois algorithmes à une méthode de référence. Le choix de Gradient Boosting repose sur son rappel de 60,8 %, conformément à l'objectif de détection des pannes. Les résultats montrent aussi que la réussite des tests, le classement des modèles et l'aptitude à un déploiement industriel doivent être évalués séparément.
