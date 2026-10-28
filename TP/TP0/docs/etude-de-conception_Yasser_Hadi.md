

Groupe : [Yasser Hadli - Adam biloul] · Version : [finale] · Date : [28/10/2026]

## 1. Contexte et utilisateurs

- Utilisateur principal (un rôle précis) : responsable de maintenance industrielle, envisagé comme destinataire de l'aide à la décision.

Le projet S3 du module Applications Intelligentes porte sur la détection de situations associées à une panne à partir de mesures de fonctionnement. Son objectif académique est de construire une chaîne reproductible de classification supervisée, de comparer une baseline à au moins trois modèles et de sélectionner le meilleur selon une métrique explicitement choisie.

Le prototype vise à aider le responsable de maintenance à prioriser les équipements à contrôler. Cet usage constitue un scénario métier : le code actuel réalise un apprentissage et une évaluation hors ligne, sans interface ni connexion à des machines. Les données ne documentent ni l'instant de la panne ni un horizon de prévision ; la classification de l'étiquette `failure` ne démontre donc pas une capacité à anticiper une panne plusieurs heures à l'avance.

## 2. Problème et décision

- Décision que l'utilisateur prend avec le système : déterminer quelles observations justifient une inspection prioritaire. La sortie binaire indique une panne prédite (`1`) ou une absence de panne prédite (`0`) ; elle doit éclairer une décision humaine.
- Fréquence : à chaque exécution manuelle du pipeline pour l'expérimentation. La fréquence d'analyse des équipements dans un futur usage industriel reste à définir, car aucune acquisition périodique n'est implémentée.
- Coût d'une erreur, et pour qui : un faux négatif correspond à une panne réelle non détectée, susceptible d'entraîner un arrêt non planifié, une dégradation du matériel ou un risque pour les opérateurs. Un faux positif peut provoquer une inspection inutile, une mobilisation de techniciens ou une interruption préventive injustifiée. Ces coûts concernent l'exploitant et les équipes de maintenance ; ils ne sont pas chiffrés dans les données disponibles.

La problématique est une classification binaire supervisée : apprendre la relation entre quatre variables explicatives et la cible `failure`. La priorité méthodologique est de détecter les observations positives, ce qui motive le choix du rappel plutôt que de la seule proportion globale de prédictions correctes.

## 3. Approche par sous-problème

| Sous-problème | Approche (règles / ML / DL / LLM / RAG) | Pourquoi pas une approche plus simple |
|---|---|---|
| Contrôler et préparer les données | Règles déterministes avec pandas et transformations scikit-learn | Le contrôle de la cible, le retrait des cibles manquantes et l'imputation répondent à des règles explicites ; un modèle supplémentaire serait inutile. |
| Établir une référence minimale | Baseline : classe majoritaire avec `DummyClassifier` | Cette approche simple est conservée pour vérifier qu'un modèle apprend davantage que la fréquence des classes. |
| Relier les mesures à la panne | ML : régression logistique | Elle fournit une première classification fondée sur les variables, avec une frontière linéaire dans l'espace transformé ; aucune règle métier chiffrée n'est fournie pour s'y substituer. |
| Représenter des relations non linéaires | ML : Random Forest et Gradient Boosting | Les arbres peuvent représenter des interactions entre température, vibration, charge et durée de fonctionnement que la frontière linéaire peut manquer. Leur intérêt est évalué sur le même jeu de test. |
| Évaluer et sélectionner | Règles : calcul des scores et classement par rappel | Le critère de sélection est explicite et ne nécessite pas d'apprentissage supplémentaire. |

La régression logistique utilise `max_iter=1000` et `random_state=42`. La forêt aléatoire agrège les décisions de 100 arbres (`n_estimators=100`, `random_state=42`). Le Gradient Boosting construit successivement des arbres qui corrigent les erreurs de l'ensemble ; seul `random_state=42` est explicitement réglé dans sa construction, les autres paramètres étant ceux de scikit-learn. Aucune recherche d'hyperparamètres n'est implémentée. Le recours au deep learning, aux LLM ou au RAG n'est pas justifié par ces données tabulaires de faible dimension, sans texte libre ni base documentaire.

## 4. Données

| Jeu de données | Source et licence | Volume | Données personnelles ? |
|---|---|---|---|
| `data/exemple_pannes.csv` | Fichier d'exemple fourni avec le projet pédagogique. La provenance détaillée, le mode de collecte et la licence restent à documenter. | 5 000 lignes, 4 variables explicatives numériques et 1 cible binaire | Aucun identifiant personnel ni attribut directement personnel observé dans les colonnes ; la provenance reste à vérifier. |

Les noms de colonnes indiquent les grandeurs suivantes : `temperature_c` (température en °C, de 35,8 à 92,6), `vibration_mm_s` (vibration en mm/s, de 0,02 à 22,23), `load_pct` (charge en %, de 20 à 100) et `runtime_hours` (durée de fonctionnement en heures, de 5 à 29 999). La cible `failure` vaut `0` ou `1`. Les conventions de mesure et la définition opérationnelle d'une panne ne sont pas précisées par le CSV.

Le fichier contient 4 122 observations de classe `0` (82,44 %) et 878 de classe `1` (17,56 %). L'inspection ne relève aucune valeur manquante ni ligne entièrement dupliquée. Ces constats ne prouvent pas l'absence d'erreurs de mesure, de biais d'échantillonnage ou de dépendances entre observations. Aucun horodatage ni identifiant d'équipement n'est disponible.

Dans `config.py`, `FEATURES=None` conduit à retenir toutes les colonnes hors cible et éventuelle date : les quatre variables numériques sont donc utilisées. `load_data()` lit le CSV, vérifie la présence de `failure` et supprime les lignes sans cible ; sur ce fichier complet, aucune ligne n'est supprimée pour cette raison. Les valeurs explicatives manquantes éventuelles seraient imputées par la médiane, puis chaque variable numérique serait centrée et réduite avec `StandardScaler`. Cette mise à l'échelle est particulièrement pertinente pour la régression logistique ; elle est conservée dans le pipeline commun aux modèles, même si les arbres n'en ont généralement pas besoin. Aucun filtrage des valeurs extrêmes n'est implémenté.

Le découpage configuré est aléatoire et stratifié sur la cible (`SPLIT="random"`, `TEST_SIZE=0.2`, `SEED=42`). Il produit 4 000 observations d'entraînement, dont 702 pannes, et 1 000 observations de test, dont 176 pannes. La stratification préserve approximativement la proportion des classes, sans rééquilibrage. La graine fixe rend le partage reproductible dans un même environnement. L'imputation et la standardisation sont ajustées uniquement sur l'entraînement grâce au `Pipeline`, puis appliquées au test : les statistiques du test ne servent pas au prétraitement appris.

## 5. Métriques, baseline et critères d'acceptation

- Baseline et son score : `DummyClassifier(strategy="most_frequent")` prédit toujours `0`. Sur le test, son accuracy vaut 0,824 et son rappel, sa précision et son F1 valent tous 0,000 selon `models/metrics.json`. Son accuracy élevée reflète surtout la majorité des observations sans panne ; elle ne détecte aucune panne.
- Métrique principale, et pourquoi : le rappel (`MAIN_METRIC="recall"`), défini par `VP / (VP + FN)`, mesure la proportion des pannes réelles détectées. Il répond à la priorité donnée à la réduction des faux négatifs. La classe positive est `failure=1` et l'évaluation est binaire.
- Critères d'acceptation (métrique, nombre, jeu de données, condition) : les consignes demandent de discuter l'acceptation, mais ne fixent aucun seuil numérique. La règle implémentée consiste à retenir le rappel maximal parmi les candidats sur les 1 000 observations de test. Gradient Boosting satisfait cette règle avec 0,608 et dépasse la baseline de 0,608, soit 60,8 points de pourcentage. L'acceptabilité industrielle reste indéterminée : un seuil devrait être défini avec les responsables métier, pour des données représentatives et des coûts d'erreur documentés.

L'accuracy est la proportion globale de prédictions correctes. La précision, `VP / (VP + FP)`, indique la proportion de pannes réelles parmi les alertes. Le F1 est la moyenne harmonique de la précision et du rappel : il renseigne sur leur compromis, sans représenter explicitement les coûts métier. `evaluate()` arrondit chaque score à trois décimales et utilise `zero_division=0`, ce qui explique la précision nulle de la baseline sans prédiction positive.

Gradient Boosting obtient une accuracy de 0,888, une précision de 0,713 et un F1 de 0,656. Son rappel de 60,8 % implique qu'environ 39,2 % des pannes du test ne sont pas détectées, d'après le score arrondi. Ce niveau est le meilleur de la comparaison, mais ne constitue pas une garantie de sécurité. Les résultats détaillés et les compromis sont présentés dans `choix-modele_Yasser_Hadi.md`.

## 6. Architecture prévue

L'architecture effectivement implémentée est une chaîne locale d'apprentissage par lots, organisée en modules :

1. `monprojet/config.py` centralise le chemin des données, la cible, le type de tâche, la métrique, le découpage et la graine.
2. `monprojet/data.py` charge les données, sélectionne les variables et réalise le partage entraînement/test.
3. `monprojet/models.py` fournit un dictionnaire de candidats contenant d'abord la baseline, puis les trois classifieurs.
4. `monprojet/train.py` construit pour chaque candidat un `Pipeline` associant le prétraitement à l'estimateur, ajuste ce pipeline sur l'entraînement et calcule les scores sur le même test.
5. Les scores sont classés par rappel décroissant. Le programme affiche le tableau, sauvegarde le pipeline retenu dans `models/model.joblib` et écrit la métrique principale, le nom du meilleur modèle et tous les scores dans `models/metrics.json`.

Le `ColumnTransformer` produit une représentation dense (`sparse_threshold=0`). Il prévoit aussi l'imputation des catégories par la modalité la plus fréquente et leur encodage one-hot avec gestion des catégories inconnues ; une branche TF-IDF est prévue pour une colonne textuelle configurée. Ces branches ne sont pas actives ici : les entrées sont numériques et `TEXT_COLUMN=None`. De même, le code permet un partage chronologique et des modèles de régression, mais la configuration présente ne les utilise pas.

La sauvegarde du pipeline associe les transformations apprises au classifieur pour permettre leur réutilisation cohérente. Elle ne constitue pas à elle seule une application de production : aucun service de prédiction, tableau de bord, ordonnanceur ou mécanisme de surveillance continue n'est implémenté.

## 7. Choix technologiques

Python sert à organiser le traitement en fonctions et modules testables. pandas assure la lecture du CSV et la manipulation des tableaux ; NumPy intervient notamment dans l'identification des classes pour l'évaluation. scikit-learn fournit le découpage stratifié, les transformations, les pipelines, les classifieurs et les métriques, ce qui permet de comparer les candidats selon un protocole commun. joblib assure la sérialisation du pipeline retenu et le module standard `json` l'enregistrement lisible des scores.

Les dépendances sont fixées dans `requirements.txt` : NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, joblib 1.5.3 et pytest 9.1.1. Ce verrouillage et la graine 42 contribuent à la reproductibilité, sans remplacer la traçabilité des données et de l'environnement. Le volume de 5 000 lignes ne justifie pas une infrastructure distribuée de Big Data dans ce prototype.

Pytest vérifie les comportements attendus du chargement, du découpage, des candidats et de l'évaluation. Les huit tests sont annoncés réussis dans l'état du projet communiqué ; ils ne démontrent ni la représentativité industrielle des données ni la qualité des prédictions en conditions réelles.

## 8. Risques, éthique, confidentialité

Le risque principal est une confiance excessive dans une prédiction négative : le modèle retenu manque encore une part importante des pannes. Toute utilisation opérationnelle nécessiterait une supervision humaine et une articulation avec les procédures de maintenance et de sécurité existantes. Inversement, les fausses alertes peuvent augmenter la charge de maintenance ; le choix d'un futur seuil devrait intégrer ces deux types de coûts.

Le déséquilibre des classes peut masquer les erreurs sur les pannes si l'on privilégie l'accuracy. Le partage aléatoire suppose une indépendance suffisante entre observations ; si plusieurs lignes proviennent d'une même machine ou d'une même période, les scores pourraient être optimistes. L'absence d'identifiants et de dates empêche de vérifier ce point. Les changements de machines, de capteurs ou de régimes de fonctionnement peuvent également dégrader les performances après l'apprentissage.

Les colonnes inspectées ne contiennent pas de données directement personnelles. Des mesures industrielles peuvent néanmoins être confidentielles, car elles renseignent sur l'activité des équipements. Avant tout usage externe, il conviendrait de confirmer les droits d'utilisation du jeu fourni et de prévoir les accès aux données et aux artefacts. Aucun dispositif de contrôle d'accès n'est décrit dans le prototype. Enfin, un fichier joblib ne devrait être chargé que s'il provient d'une source de confiance, sa désérialisation pouvant exécuter du code.

## 9. Limites et hors périmètre

L'évaluation repose sur un seul partage de données. Le même test sert à comparer les candidats et à sélectionner le meilleur : le score retenu peut bénéficier de cette sélection et ne remplace pas une évaluation finale indépendante. Aucune validation croisée, estimation d'incertitude, recherche d'hyperparamètres, pondération des classes ou optimisation du seuil de décision n'est implémentée. Les scores enregistrés sont arrondis et ne permettent pas de conclure à la significativité des écarts.

La source détaillée, la licence, la représentativité du fichier et le protocole d'étiquetage restent à documenter. Sans chronologie, horizon cible ni identifiant machine, le projet n'évalue ni une anticipation temporelle des pannes ni la généralisation à des équipements inconnus. Il n'estime pas la durée de vie restante et ne diagnostique pas la cause d'une panne.

Le déploiement industriel, l'acquisition en temps réel, les décisions automatiques d'arrêt et la surveillance de dérive sont hors du périmètre réalisé. Des prolongements pertinents seraient une validation adaptée aux machines et aux périodes, un réglage sur données de validation distinctes et une analyse des coûts d'erreur avec les utilisateurs. Il s'agit de pistes futures, sans résultat expérimental revendiqué.
