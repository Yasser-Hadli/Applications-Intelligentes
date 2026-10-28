# Compte rendu du TP0 — Prédiction des pannes industrielles

Groupe : [Yasser Hadli - Adam biloul] · Version : [finale] · Date : [28/10/2026]

## 1. Introduction et objectifs

Ce TP0, réalisé dans le cadre du module Applications Intelligentes, porte sur la mise en œuvre d'une chaîne de Machine Learning pour la classification de pannes industrielles. À partir du projet pédagogique fourni, le travail consiste à compléter le chargement des données, leur découpage, la définition des modèles candidats et leur évaluation, puis à justifier le choix du modèle retenu.

L'objectif est de comparer une méthode de référence à trois algorithmes de classification supervisée dans des conditions communes et reproductibles. Le rappel constitue la métrique principale, car une panne réelle non détectée peut avoir des conséquences importantes pour la maintenance et la continuité de production. Le prototype reste une expérimentation hors ligne ; il ne commande aucun équipement et ne définit pas d'horizon temporel d'anticipation des pannes.

## 2. Présentation des données

Le fichier local `data/exemple_pannes.csv` contient 5 000 observations et cinq colonnes :

| Colonne | Rôle | Signification indiquée par le nom |
|---|---|---|
| `temperature_c` | Variable explicative numérique | Température en degrés Celsius |
| `vibration_mm_s` | Variable explicative numérique | Vibration en millimètres par seconde |
| `load_pct` | Variable explicative numérique | Charge en pourcentage |
| `runtime_hours` | Variable explicative numérique | Durée de fonctionnement en heures |
| `failure` | Cible binaire | Absence de panne (`0`) ou panne (`1`) |

Le jeu comprend 4 122 observations sans panne et 878 observations avec panne, soit respectivement 82,44 % et 17,56 %. Cette répartition déséquilibrée impose de ne pas interpréter l'accuracy isolément. L'inspection du fichier n'a révélé aucune valeur manquante ni ligne entièrement dupliquée. La provenance détaillée, la licence et le protocole d'étiquetage ne sont pas documentés dans les éléments disponibles.

## 3. Réalisation du pipeline

Le projet est organisé en quatre modules principaux. `config.py` centralise le chemin du CSV, la cible, la tâche de classification, la métrique principale et les paramètres de découpage. `data.py` assure le chargement et la séparation des données. `models.py` définit les candidats. Enfin, `train.py` associe le prétraitement aux estimateurs, calcule les métriques et sélectionne le modèle retenu.

La fonction `load_data()` lit le CSV avec pandas, vérifie que la colonne cible existe et retire les observations dont la cible serait manquante. Les quatre variables explicatives sont sélectionnées, puisque `FEATURES=None` et qu'aucune colonne temporelle n'est configurée.

Le découpage est aléatoire et stratifié sur `failure`, avec une proportion de test de 20 % et une graine fixée à 42. Il produit 4 000 observations d'entraînement, dont 702 pannes, et 1 000 observations de test, dont 176 pannes. La stratification maintient approximativement la répartition des classes dans les deux ensembles.

Le prétraitement numérique associe une imputation par la médiane à un centrage et une réduction avec `StandardScaler`. L'imputation est prévue pour d'éventuelles valeurs explicatives manquantes, même si le CSV actuel est complet. Les transformations sont ajustées uniquement sur l'entraînement, au sein d'un `Pipeline`, puis appliquées au test. Cette organisation évite d'utiliser les statistiques du test pour apprendre le prétraitement.

## 4. Modèles comparés et protocole d'évaluation

Quatre candidats sont évalués sur le même jeu de test :

- **Baseline** : `DummyClassifier(strategy="most_frequent")`, qui prédit systématiquement la classe majoritaire.
- **Régression logistique** : un classifieur linéaire avec `max_iter=1000` et `random_state=42`.
- **Random Forest** : un ensemble de 100 arbres avec `random_state=42`, permettant de représenter des relations non linéaires.
- **Gradient Boosting** : un ensemble construit séquentiellement pour corriger les erreurs précédentes, avec `random_state=42` et les autres paramètres laissés à leurs valeurs par défaut.

La fonction `evaluate()` calcule l'accuracy, le rappel, la précision et le F1-score. Les trois dernières métriques portent sur la classe positive `failure=1`. Les scores sont arrondis à trois décimales, et les cas de division indéfinie sont traités avec `zero_division=0`. Aucune recherche d'hyperparamètres, pondération des classes ou optimisation du seuil de décision n'est implémentée.

## 5. Résultats obtenus

Les résultats suivants proviennent de `models/metrics.json` ; ils ne correspondent pas à une nouvelle expérimentation réalisée pour ce compte rendu.

| Modèle | Accuracy | Recall | Precision | F1-score |
|---|---|---|---|---|
| Baseline | 0,824 | 0,000 | 0,000 | 0,000 |
| Logistic Regression | 0,889 | 0,574 | 0,737 | 0,645 |
| Random Forest | 0,873 | 0,545 | 0,671 | 0,602 |
| Gradient Boosting | 0,888 | 0,608 | 0,713 | 0,656 |

La baseline obtient une accuracy de 82,4 % uniquement en prédisant l'absence de panne. Son rappel nul montre qu'elle ne répond pas à l'objectif de détection. Elle fournit ainsi une référence essentielle pour apprécier l'apport des modèles supervisés.

Gradient Boosting présente le rappel le plus élevé, à 60,8 %. Il dépasse la régression logistique de 3,4 points de pourcentage et Random Forest de 6,3 points. Il est donc retenu conformément à la métrique principale configurée. Le code sauvegarde le pipeline gagnant dans `models/model.joblib` et le rapport de comparaison dans `models/metrics.json`.

La régression logistique obtient néanmoins la meilleure précision, à 73,7 %, et la meilleure accuracy, à 88,9 %. Gradient Boosting privilégie davantage la détection des pannes, avec une précision de 71,3 % et une accuracy de 88,8 %. Son F1-score de 0,656 est également le meilleur de la comparaison, mais ne constitue pas une mesure directe des coûts industriels.

## 6. Validation logicielle et organisation du rendu

Après la réorganisation du projet, les huit tests Pytest existants ont été exécutés avec succès : **8 passed en 36,85 secondes**. Les tests couvrent notamment le chargement, le découpage, la présence de la baseline et des modèles candidats, le calcul de la métrique principale, l'écriture des résultats et la comparaison avec la baseline.

Deux tests appellent la fonction d'entraînement `main()`. Pour préserver les modèles et résultats déjà enregistrés, leur destination de sauvegarde a été redirigée temporairement pendant cette vérification. Les fichiers Python et les tests n'ont pas été modifiés. La comparaison des empreintes des fichiers a confirmé la conservation du dataset, des modèles et des résultats locaux. La réussite des tests valide les comportements logiciels vérifiés, sans démontrer l'acceptabilité industrielle du modèle.

Le projet terminé est organisé dans `TP/TP0/`, tandis que `TP/TP1/` est réservé au TP suivant et contient un `.gitkeep`. Un `.gitignore` placé à la racine exclut les données, les artefacts entraînés, les environnements virtuels, les caches et les fichiers de secrets. Les fichiers source `monprojet/data.py` et `monprojet/models.py` restent destinés au suivi Git.

La collaboration est prévue dans un dépôt commun et sur la branche `TP`, avec des documents individuels et des commits identifiables pour chaque étudiant. Les documents existants ont été renommés avec le suffixe `_Yasser_Hadi`, conformément à la convention demandée lors de l'organisation du dépôt. Le présent compte rendu conserve le nom de fichier explicitement demandé : `Compte rendu Yasser Hadli.md`.

## 7. Limites et perspectives

Le rappel de 60,8 % signifie qu'environ 39,2 % des pannes du test restent non détectées, d'après le score arrondi. Le meilleur résultat parmi les candidats ne suffit donc pas à garantir la sécurité d'une installation. Aucun seuil numérique d'acceptation n'est imposé dans les consignes originales ; une éventuelle validation industrielle devrait être définie avec les responsables métier selon la criticité des pannes et le coût des fausses alertes.

L'évaluation repose sur un seul partage aléatoire, et le même test sert à comparer les modèles puis à sélectionner le meilleur. Aucun intervalle de confiance n'est disponible. L'absence de dates et d'identifiants d'équipements empêche également de vérifier la généralisation à de nouvelles machines ou à des périodes futures.

Pour prolonger ce travail, il serait pertinent de mettre en place une validation croisée sur les données de développement, de conserver un test final indépendant, d'étudier le seuil de décision à l'aide d'une courbe précision-rappel et de rechercher les hyperparamètres du Gradient Boosting. Une analyse des faux négatifs avec des spécialistes de maintenance et une documentation plus précise de la collecte renforceraient aussi l'interprétation des résultats. Ces propositions n'ont pas été implémentées et aucun gain de performance n'est revendiqué.

## 8. Conclusion

Ce TP0 a permis de compléter et de valider une chaîne de classification supervisée, puis de comparer une baseline à trois modèles de Machine Learning. Gradient Boosting est retenu pour son rappel de 60,8 %, en cohérence avec la priorité donnée à la détection des pannes. L'analyse souligne cependant la nécessité de distinguer réussite des tests logiciels, supériorité sur les modèles comparés et validation pour un usage industriel.
