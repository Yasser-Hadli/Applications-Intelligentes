# Compte rendu · TP1 : une API de prédiction pour MaintAI

Groupe : [Yasser Hadli - Adam biloul] · Version : [finale] · Date : [28/10/2026]

**Module :** Développement et déploiement d'applications intelligentes — EMSI.

**TP :** TP1 — Une API de prédiction pour MaintAI.

**Dépôt du projet :** https://github.com/Yasser-Hadli/Applications-Intelligentes · Branche commune : `TP`.

Ce TP porte sur l'exposition d'un modèle de prédiction de panne au moyen d'une API FastAPI. Le travail consiste à définir un contrat d'entrée, à implémenter la route de prédiction et à vérifier le comportement du service pour des requêtes valides, invalides et en cas d'indisponibilité du modèle. Les résultats présentés proviennent des exécutions locales consignées dans les journaux de vérification.

---

## 1. Mise en place

L'exécution initiale de `python -m pytest`, avant l'implémentation du schéma d'entrée et de la route de prédiction, donne le résultat suivant :

```
=================== 3 failed, 10 passed, 1 warning in 9.60s ===================
```

L'analyse des tests initiaux permet de distinguer les comportements déjà opérationnels des validations qui réussissent sans encore contrôler les contraintes attendues.

L'environnement de travail repose sur Python 3.11.0 et un environnement virtuel contenant les dépendances du projet. L'installation à partir de `requirements.txt` a confirmé leur présence. Les versions fixées comprennent numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, joblib 1.5.3, FastAPI 0.141.1, Uvicorn 0.54.0, httpx 0.28.1 et pytest 9.1.1 ; Pydantic installé est en version 2.14.1.

Avant toute modification du code de ce TP, les quatre tests API déjà réussis sont `test_health`, `test_out_of_range_value_is_rejected`, `test_missing_field_is_rejected` et `test_openapi_documents_predict`. Les six tests de données et d'entraînement passent aussi. Les trois échecs concernent `test_predict_returns_probability` (422 au lieu de 200), `test_high_risk_reading_raises_alert` (`KeyError: 'alert'`, car la réponse contient des erreurs de validation) et `test_503_when_model_missing` (422 au lieu de 503).

Avec un `SensorReading` vide et `extra="forbid"`, chaque clé reçue est inconnue : Pydantic rejette donc les corps de requête avant l'appel à la route. Les deux tests 422 passent pour une mauvaise raison : ils ne prouvent encore ni la détection d'une température excessive, ni le caractère obligatoire de `load_pct`. La comparaison du seul code HTTP masque la nature réelle de l'erreur.

L'implémentation porte sur les quatre champs du schéma et le corps de `predict()`. La génération synthétique, le découpage stratifié, le pipeline `StandardScaler` puis `LogisticRegression`, la sauvegarde des artefacts et les fixtures sont fournis dans le projet. Les fixtures entraînent sur 2 000 relevés avec une graine 0 dans un dossier temporaire ; l'exécution manuelle utilise 5 000 relevés avec la graine 42. Les tests et le code ML sont conservés.

## 2. Le schéma d'entrée (`maintai/api/schemas.py`)

| Variable | Type | Minimum (`ge`) | Maximum (`le`) | Justification physique (pas le min / max des données) |
|---|---|---|---|---|
| `temperature_c` | float | -40 | 150 | Plage de fonctionnement retenue pour une machine industrielle ; accepte le froid et un échauffement important, rejette 500 °C. |
| `vibration_mm_s` | float | 0 | 50 | Amplitude de vitesse vibratoire positive ou nulle ; plafond de plausibilité choisi pour la démo, autorisant des vibrations sévères. |
| `load_pct` | float | 0 | 100 | Pourcentage de capacité nominale ; ce contrat suppose l'absence de surcharge supérieure à 100 %. |
| `runtime_hours` | float | 0 | 100000 | Compteur cumulatif non négatif ; plafond d'environ 11,4 années de marche continue retenu pour le parc simulé. |

Après l'implémentation des quatre champs de `schemas.py`, une nouvelle exécution de la suite produit :

```
================== 3 failed, 10 passed, 1 warning in 13.66s ===================
```

Les trois mêmes tests échouent désormais avec `NotImplementedError`, levée dans `maintai/api/main.py`. C'est une progression cohérente : `VALID` et `HIGH_RISK` passent la validation et atteignent la fonction encore vide. Les tests 422 passent maintenant pour les raisons attendues : violation de la borne supérieure pour 500 °C et absence de `load_pct`. Le nombre de tests réussis ne change donc pas, mais la cause des échecs établit que la première étape est correcte. Les bornes inclusives constituent le contrat physique retenu pour cette démo ; elles doivent être adaptées aux spécifications du capteur et de la machine avant un usage industriel, et ne sont pas déduites des extrêmes du jeu synthétique.

## 3. La fonction `predict()` (`maintai/api/main.py`)

La route de prédiction récupère le prédicteur chargé au démarrage, contrôle sa disponibilité et transmet les mesures validées au modèle :

```python
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle de prédiction indisponible")
    return PredictionOut(**predictor.predict(reading.model_dump()))
```

**Gestion de l'indisponibilité du modèle.**

Le code 503 (`Service Unavailable`) décrit une indisponibilité du service de prédiction : les artefacts nécessaires ne sont pas disponibles, alors que l'API peut répondre à `/health`. Une erreur 500 décrit une erreur interne non traitée ; ici, l'absence du modèle est un état prévu et traité explicitement. Le test dédié vérifie effectivement ce comportement.

**Chargement du modèle au démarrage.**

Le lifespan charge le pipeline et ses métadonnées une fois par processus serveur. Les requêtes réutilisent ensuite le prédicteur en mémoire, ce qui évite de relire et désérialiser les fichiers à chaque appel et réduit la latence. L'ordre des variables et le seuil restent ceux de l'artefact chargé. Avec plusieurs workers, chaque processus dispose de sa propre instance ; un redémarrage ou un rechargement crée une nouvelle instance.

## 4. Les tests

L'exécution finale de `python -m pytest` confirme la réussite des treize tests :

```
======================== 13 passed, 1 warning in 6.60s ========================
```

Les sept tests API, les trois tests de données et les trois tests d'entraînement réussissent sans modification des tests fournis. Un avertissement de dépréciation concerne l'utilisation de `httpx` avec `starlette.testclient`. Il provient des dépendances installées et ne fait échouer aucun test. Les versions définies dans le projet sont conservées.

## 5. L'API en marche

La vérification du service suit trois étapes : génération des données avec `python -m maintai.dataset`, entraînement avec `python -m maintai.train`, puis lancement du serveur avec `uvicorn maintai.api.main:app --reload`. Les requêtes HTTP donnent les réponses suivantes :

| Requête | Code obtenu | Réponse (le corps JSON) |
|---|---|---|
| `GET /health` | 200 | `{"status": "ok", "model_loaded": true}` |
| `POST /v1/predict` avec `{"temperature_c": 72.5, "vibration_mm_s": 6.1, "load_pct": 85, "runtime_hours": 15000}` | 200 | `{"failure_probability": 0.9553, "alert": true, "threshold": 0.3, "model_version": "2026-10-28T22:06:47Z"}` |
| `POST /v1/predict` avec une température de 500 | 422 | `{"detail": [{"type": "less_than_equal", "loc": ["body", "temperature_c"], "msg": "Input should be less than or equal to 150", "input": 500, "ctx": {"le": 150.0}}]}` |
| `POST /v1/predicte` (faute de frappe dans l'adresse) | 404 | `{"detail": "Not Found"}` |

Vérifications effectuées avec `.venv\Scripts\python.exe -m maintai.dataset`, puis `.venv\Scripts\python.exe -m maintai.train`. La génération produit 5 000 relevés avec un taux de panne affiché de 13,6 %. Le serveur est lancé par `.venv\Scripts\python.exe -m uvicorn maintai.api.main:app --reload`, puis arrêté après la vérification. Les quatre réponses ci-dessus sont issues de requêtes HTTP réelles vers `http://127.0.0.1:8000`, sans recours au TestClient pour cette étape. La température 500 remplace uniquement 72,5 dans le deuxième corps ; les trois autres valeurs sont conservées. La requête mal orthographiée reprend le corps valide.

Métriques réellement produites par l'entraînement manuel (distinct du modèle temporaire des tests) :

```json
{
  "model": {
    "accuracy": 0.904,
    "recall": 0.75,
    "precision": 0.622,
    "threshold": 0.3
  },
  "baseline": {
    "accuracy": 0.864,
    "recall": 0.0
  },
  "n_train": 4000,
  "n_test": 1000
}
```

L'accuracy de la baseline illustre le déséquilibre des classes : elle ne détecte aucune panne (rappel nul). Au seuil 0,3, le modèle détecte 75 % des pannes du lot de test synthétique ; ces résultats ne démontrent pas une performance sur des machines réelles. La valeur `model_version` est l'horodatage UTC enregistré par le programme d'entraînement et reproduit sans modification.

**Documentation du contrat d'entrée.**

La documentation interactive `/docs` présente le schéma `SensorReading` dans la partie `Schemas`. Les propriétés des capteurs y décrivent les bornes, les unités et les champs obligatoires issus du contrat OpenAPI. Le corps de `POST /v1/predict` référence ce schéma et propose un exemple de relevé. La vérification HTTP a obtenu 200 pour `/docs` et identifié le composant Swagger UI dans sa réponse HTML ; `/openapi.json` confirme les quatre champs requis, leurs minima et maxima ainsi que `additionalProperties: false`.

## 6. Questions de compréhension

1. REST et FastAPI : quelle est la différence ?
   REST est un style architectural d'API organisé autour de ressources et de représentations, exploitant notamment les méthodes et codes HTTP et des échanges sans état de session côté serveur. FastAPI est un framework Python qui permet d'implémenter une API HTTP, avec routage, validation et documentation. Utiliser FastAPI ne garantit pas à lui seul le respect de tous les principes REST.
2. Le schéma Pydantic sert à trois choses. Lesquelles ?
   Le schéma Pydantic définit le contrat typé des données et leurs contraintes ; il valide et convertit les entrées compatibles, en produisant des erreurs structurées ; il fournit la représentation sérialisable et le schéma JSON utilisés par FastAPI pour les échanges et la documentation OpenAPI. `extra="forbid"` rejette les champs inconnus et `model_dump()` transmet au prédicteur un dictionnaire des quatre valeurs validées.
3. Dans `http://localhost:8000/v1/predict`, que désignent `localhost`, `8000` et `/v1/predict` ?
   `localhost` désigne la machine sur laquelle s'exécute le client (boucle locale ; ici le serveur est joint via 127.0.0.1). `8000` est le port TCP écouté par Uvicorn. `/v1/predict` est le chemin de la route : `v1` indique la version du contrat et `predict` l'opération exposée. La méthode POST accompagne ce chemin d'un corps JSON ; l'URL seule ne précise pas cette méthode.
4. Pourquoi `/v1` dans l'adresse ?
   Le préfixe `/v1` rend explicite la version majeure du contrat HTTP. Une future évolution incompatible peut être exposée sous `/v2` tout en conservant temporairement `/v1` pour les clients existants. Cette version d'API est distincte de `model_version`, qui identifie l'artefact ML chargé.
5. Dans la commande `uvicorn maintai.api.main:app`, que désignent `maintai.api.main` et `app` ?
   Uvicorn est le serveur ASGI qui écoute les connexions HTTP et exécute l'application. `maintai.api.main` est le chemin d'import du module Python (fichier `maintai/api/main.py`) ; `app`, après les deux-points, est l'objet FastAPI exporté par ce module. `--reload` surveille les changements et redémarre le processus serveur pendant le développement.

## 7. Le rendu Git

Le TP1 est intégré à l'arborescence du dépôt commun `Applications-Intelligentes`, dans le dossier `TP/TP1/`. Le travail collaboratif utilise la branche `TP`, associée au dépôt distant `origin`. Le présent document constitue le compte rendu individuel de Yasser Hadli au sein du groupe Yasser Hadli – Adam biloul.

Les sources de l'API, les tests, les dépendances et la documentation sont séparés des fichiers d'exécution locaux. Les règles d'exclusion Git couvrent les données, les modèles entraînés, les environnements virtuels, les caches et les journaux de vérification. Cette organisation permet de partager le code et le rendu académique sans inclure les artefacts générés lors des essais.

## 8. Ce qui reste flou

La calibration des probabilités et le choix du seuil constituent les principales limites méthodologiques de cette expérimentation. Le seuil 0,3 est fourni par la configuration. Il permet de transformer une probabilité en alerte, mais sa pertinence dépend du coût des pannes manquées et des interventions inutiles. Une évaluation industrielle nécessiterait des observations représentatives des équipements et une analyse de ces coûts.

Le rappel de 75 % signifie qu'environ un quart des pannes du jeu de test synthétique ne sont pas détectées. La précision de 62,2 % indique également la présence de fausses alertes. Ces résultats illustrent le compromis entre détection et charge de maintenance ; ils ne garantissent pas les mêmes performances sur des machines réelles.

Le TP aboutit à une API fonctionnelle qui valide les relevés, fournit une probabilité de panne et une alerte, documente son contrat et traite explicitement l'indisponibilité du modèle. La réussite des treize tests confirme les comportements vérifiés. Le passage à un usage opérationnel demanderait une validation sur données réelles et une étude du seuil adaptée au contexte de maintenance.
