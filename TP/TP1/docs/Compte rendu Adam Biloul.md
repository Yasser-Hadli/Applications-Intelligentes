# Compte rendu du TP1 — API de prédiction MaintAI

**Étudiant :** Adam Biloul  
**Groupe :** Yasser Hadli et Adam Biloul · Version finale · Date : 28/10/2026  
**Module :** Développement et déploiement d'applications intelligentes — EMSI  
**Sujet :** TP1 — Une API de prédiction pour MaintAI  
**Dépôt :** https://github.com/Yasser-Hadli/Applications-Intelligentes · Branche partagée : `TP`

Ce travail consiste à rendre un modèle de prédiction de panne accessible à travers une API développée avec FastAPI. Il comprend la définition des données acceptées, l'implémentation du point d'accès de prédiction et la vérification des réponses du service face à des entrées correctes, incorrectes ou à un modèle indisponible. Les observations et résultats repris dans ce document proviennent des vérifications locales consignées dans le compte rendu source ; cette reformulation ne correspond pas à une nouvelle exécution.

## 1. Préparation et état initial du projet

Avant de compléter le schéma d'entrée et la route de prédiction, la commande `python -m pytest` produit le bilan suivant :

```text
3 failed, 10 passed, 1 warning in 9.60s
```

L'environnement utilisé comprend Python 3.11.0 et un environnement virtuel dédié. L'installation depuis `requirements.txt` confirme la présence des dépendances. Les versions fixées sont numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, joblib 1.5.3, FastAPI 0.141.1, Uvicorn 0.54.0, httpx 0.28.1 et pytest 9.1.1. La version installée de Pydantic est 2.14.1.

À ce stade, quatre tests de l'API réussissent déjà : `test_health`, `test_out_of_range_value_is_rejected`, `test_missing_field_is_rejected` et `test_openapi_documents_predict`. Les six tests portant sur les données et l'entraînement passent également. Les échecs concernent les cas suivants :

- `test_predict_returns_probability` reçoit un code 422 au lieu de 200.
- `test_high_risk_reading_raises_alert` déclenche un `KeyError: 'alert'`, car le corps de réponse contient des erreurs de validation.
- `test_503_when_model_missing` obtient 422 au lieu du code 503 attendu.

Cette situation s'explique par un schéma `SensorReading` encore vide, associé à `extra="forbid"`. Tous les champs transmis sont alors considérés comme inconnus, et les requêtes sont rejetées avant d'atteindre la route. Les deux tests attendant un code 422 réussissent donc sans vérifier les contraintes recherchées : ils ne démontrent encore ni le rejet d'une température excessive, ni l'obligation de fournir `load_pct`. Le code HTTP seul ne permet pas d'identifier la véritable cause du rejet.

Les modifications demandées concernent les quatre propriétés du schéma et le contenu de `predict()`. Le reste est fourni : génération de données synthétiques, séparation stratifiée, pipeline `StandardScaler` suivi de `LogisticRegression`, sauvegarde des artefacts et fixtures de test. Ces fixtures utilisent 2 000 relevés et une graine de 0 dans un répertoire temporaire, tandis que l'essai manuel repose sur 5 000 relevés avec une graine de 42. Les tests existants et le code de Machine Learning sont conservés.

## 2. Définition du contrat d'entrée

Le fichier `maintai/api/schemas.py` décrit les mesures attendues. Chaque champ est de type `float` et possède des limites inclusives définies avec `ge` et `le`.

| Champ | Type | Minimum | Maximum | Motif des limites retenues |
|---|---|---|---|---|
| `temperature_c` | float | -40 | 150 | Intervalle de fonctionnement choisi pour la démonstration industrielle, couvrant le froid et un fort échauffement, mais excluant 500 °C. |
| `vibration_mm_s` | float | 0 | 50 | Amplitude vibratoire positive ou nulle ; limite haute de plausibilité permettant de représenter des vibrations sévères. |
| `load_pct` | float | 0 | 100 | Part de la capacité nominale, avec l'hypothèse contractuelle qu'une surcharge au-delà de 100 % n'est pas acceptée. |
| `runtime_hours` | float | 0 | 100000 | Durée cumulée non négative, plafonnée à environ 11,4 années de fonctionnement continu pour le parc simulé. |

Après l'ajout de ces champs, la suite de tests affiche :

```text
3 failed, 10 passed, 1 warning in 13.66s
```

Le bilan numérique reste identique, mais les trois échecs proviennent maintenant d'un `NotImplementedError` dans `maintai/api/main.py`. Les entrées `VALID` et `HIGH_RISK` franchissent donc la validation et atteignent bien la fonction à compléter. Les rejets 422 correspondent désormais aux erreurs prévues : une température de 500 dépasse la borne de 150, et l'absence de `load_pct` est signalée comme un champ obligatoire manquant.

Ces limites expriment les hypothèses physiques adoptées pour la démonstration. Elles ne sont pas calculées à partir des valeurs extrêmes du jeu synthétique. Leur utilisation sur un équipement réel demanderait une adaptation aux caractéristiques des capteurs et de la machine.

## 3. Implémentation de la prédiction

Dans `maintai/api/main.py`, la route récupère le prédicteur conservé dans l'état de l'application. Si celui-ci est disponible, elle lui transmet les mesures validées et construit une réponse conforme à `PredictionOut` :

```python
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle de prédiction indisponible")
    return PredictionOut(**predictor.predict(reading.model_dump()))
```

### Traitement de l'absence du modèle

Le statut 503, `Service Unavailable`, indique que la prédiction ne peut pas être fournie parce que les artefacts nécessaires sont indisponibles. L'API peut néanmoins continuer à répondre sur `/health`. Cet état étant prévu et géré explicitement, il est distingué d'une erreur 500, qui correspondrait à une erreur interne non traitée. Un test spécifique contrôle le retour du code 503.

### Réutilisation du prédicteur en mémoire

Le mécanisme de lifespan charge le pipeline et ses métadonnées au démarrage de chaque processus serveur. Les appels suivants réutilisent cette instance, ce qui évite de lire et de désérialiser les fichiers pour chaque requête et réduit la latence. L'ordre des variables et le seuil de décision sont ceux associés à l'artefact chargé. En présence de plusieurs workers, chacun possède son propre prédicteur ; un redémarrage ou un rechargement entraîne la création d'une nouvelle instance.

## 4. Validation par les tests

Une fois la route complétée, l'exécution finale de `python -m pytest` aboutit au résultat suivant :

```text
13 passed, 1 warning in 6.60s
```

Les sept tests de l'API, les trois tests des données et les trois tests d'entraînement réussissent, sans modification des tests fournis. L'avertissement restant concerne une dépréciation liée à l'utilisation de `httpx` avec `starlette.testclient`. Il provient des dépendances et n'empêche pas la réussite de la suite. Les versions prévues par le projet sont maintenues.

## 5. Vérification du service en fonctionnement

L'essai manuel suit trois étapes : création des données avec `python -m maintai.dataset`, entraînement avec `python -m maintai.train`, puis démarrage de l'API avec `uvicorn maintai.api.main:app --reload`.

Dans l'environnement virtuel, les commandes effectivement utilisées sont :

```powershell
.venv\Scripts\python.exe -m maintai.dataset
.venv\Scripts\python.exe -m maintai.train
.venv\Scripts\python.exe -m uvicorn maintai.api.main:app --reload
```

La génération crée 5 000 relevés et affiche un taux de panne de 13,6 %. Le serveur est ensuite interrogé sur `http://127.0.0.1:8000`, puis arrêté à la fin des vérifications. Cette étape utilise de véritables requêtes HTTP, sans passer par le TestClient.

Le corps valide envoyé à la route de prédiction est le suivant :

```json
{
  "temperature_c": 72.5,
  "vibration_mm_s": 6.1,
  "load_pct": 85,
  "runtime_hours": 15000
}
```

| Requête | Statut observé | Corps de réponse |
|---|---|---|
| `GET /health` | 200 | `{"status": "ok", "model_loaded": true}` |
| `POST /v1/predict` avec le corps valide | 200 | `{"failure_probability": 0.9553, "alert": true, "threshold": 0.3, "model_version": "2026-10-28T22:06:47Z"}` |
| `POST /v1/predict` avec `temperature_c` remplacée par 500 | 422 | `{"detail": [{"type": "less_than_equal", "loc": ["body", "temperature_c"], "msg": "Input should be less than or equal to 150", "input": 500, "ctx": {"le": 150.0}}]}` |
| `POST /v1/predicte` avec le corps valide | 404 | `{"detail": "Not Found"}` |

Pour tester la valeur hors limites, seule la température est modifiée ; les trois autres mesures restent identiques. L'adresse mal orthographiée permet de vérifier la réponse à une route inexistante.

L'entraînement manuel, distinct de celui effectué temporairement par les tests, produit les métriques suivantes :

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

La baseline atteint 86,4 % d'accuracy tout en ayant un rappel nul, ce qui met en évidence le déséquilibre des classes. Avec un seuil fixé à 0,3, le modèle identifie 75 % des pannes présentes dans le test synthétique. Ces mesures ne permettent pas de conclure à une efficacité équivalente sur des équipements réels. Le champ `model_version` reprend sans changement l'horodatage UTC enregistré lors de l'entraînement.

### Consultation de la documentation

L'interface interactive `/docs` affiche `SensorReading` dans la section `Schemas`. Elle présente les unités, les contraintes et les champs obligatoires issus du contrat OpenAPI. Le corps attendu par `POST /v1/predict` référence ce schéma et fournit un exemple de mesures.

La consultation HTTP de `/docs` renvoie un statut 200 et son HTML contient le composant Swagger UI. Le document `/openapi.json` confirme les quatre propriétés requises, leurs bornes minimales et maximales, ainsi que `additionalProperties: false`, qui traduit le refus des champs supplémentaires.

## 6. Réponses aux questions de compréhension

### Quelle différence existe entre REST et FastAPI ?

REST est un style d'architecture qui organise les échanges autour de ressources et de leurs représentations. Il s'appuie notamment sur les méthodes et les statuts HTTP, avec des échanges sans état de session côté serveur. FastAPI est un framework Python servant à construire des API HTTP : il fournit le routage, la validation et la documentation. Le choix de FastAPI ne garantit donc pas, à lui seul, le respect de l'ensemble des principes REST.

### Quels sont les trois rôles du schéma Pydantic ?

Le schéma définit d'abord les types et les contraintes du contrat de données. Il valide ensuite les entrées, convertit les valeurs compatibles et décrit les erreurs de manière structurée. Enfin, il fournit une représentation sérialisable et un schéma JSON que FastAPI exploite pour les échanges et la documentation OpenAPI. Dans ce projet, `extra="forbid"` bloque les propriétés inconnues et `model_dump()` fournit au prédicteur un dictionnaire contenant les quatre mesures validées.

### Comment interpréter `http://localhost:8000/v1/predict` ?

`localhost` désigne la machine sur laquelle le client s'exécute, accessible par la boucle locale ; les essais utilisent ici `127.0.0.1`. Le nombre `8000` correspond au port TCP sur lequel Uvicorn écoute. Le chemin `/v1/predict` identifie la route : `v1` indique la version du contrat et `predict` désigne l'opération proposée. Une requête POST vers cette route transporte un corps JSON ; la méthode HTTP n'est pas définie par l'URL seule.

### À quoi sert le préfixe `/v1` ?

Ce préfixe indique explicitement la version majeure de l'API. Une évolution incompatible pourrait être proposée sous `/v2`, tout en maintenant temporairement `/v1` pour les applications clientes existantes. Cette version du contrat HTTP est indépendante de `model_version`, qui identifie l'artefact de Machine Learning utilisé.

### Que signifie `uvicorn maintai.api.main:app` ?

Uvicorn est le serveur ASGI chargé de recevoir les connexions HTTP et d'exécuter l'application. `maintai.api.main` indique le module Python à importer, correspondant au fichier `maintai/api/main.py`. Le nom `app`, placé après les deux-points, désigne l'objet FastAPI exporté par ce module. En développement, l'option `--reload` surveille les modifications des fichiers et relance le processus serveur lorsqu'un changement est détecté.

## 7. Organisation du rendu dans Git

Le TP1 est placé dans `TP/TP1/`, au sein du dépôt commun `Applications-Intelligentes`. La collaboration utilise la branche `TP` et le dépôt distant `origin`. Ce fichier constitue le compte rendu individuel d'Adam Biloul dans le groupe formé avec Yasser Hadli.

Le code de l'API, les tests, les dépendances et les documents sont séparés des fichiers générés pendant l'exécution. Les exclusions Git portent sur les données, les modèles entraînés, les environnements virtuels, les caches et les journaux de vérification. Le dépôt peut ainsi accueillir les sources et les livrables académiques sans inclure les artefacts locaux des essais.

## 8. Points à approfondir et bilan

Les principales questions méthodologiques concernent la calibration des probabilités et la détermination du seuil d'alerte. La configuration fournit ici un seuil de 0,3 pour convertir une probabilité en décision. Son intérêt opérationnel dépend toutefois du coût des pannes non détectées et de celui des interventions inutiles. Une validation industrielle devrait s'appuyer sur des données représentatives du parc et sur une étude de ces coûts.

Le rappel de 75 % laisse environ un quart des pannes du test synthétique sans détection. La précision de 62,2 % montre également que certaines alertes sont fausses. Ces résultats traduisent un compromis entre la couverture des pannes et la charge de maintenance, sans garantir une performance identique dans un environnement réel.

Le travail aboutit à une API capable de contrôler les mesures reçues, de retourner une probabilité de panne et une alerte, de documenter son contrat et de signaler explicitement l'absence du modèle. Les treize tests réussis confirment les comportements couverts. Une utilisation opérationnelle nécessiterait encore une évaluation sur des observations réelles et un choix du seuil adapté aux besoins de maintenance.
