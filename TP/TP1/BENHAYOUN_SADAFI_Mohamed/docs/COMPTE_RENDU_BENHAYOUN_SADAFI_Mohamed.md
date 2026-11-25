# Compte rendu · TP1 : une API de prédiction pour MaintAI

**Groupe :** 5IAD G1  ·  **Membre :** BENHAYOUN SADAFI Mohamed  ·  **Coordinateur :** Yasser Hadli  ·  **Date :** 18/11/2026

---

## 1. Mise en place

Dernière ligne de `python -m pytest`, juste après la mise en place (rien n'est encore écrit) :

```
3 failed, 10 passed, 1 warning in 7.22s
```

Cette sortie et celle de la partie 2 ont été obtenues en rejouant ces états du code sur une copie temporaire du projet.

Dans `tests/test_api.py`, 4 tests sur 7 passent déjà alors que rien n'est écrit :

- `test_health`
- `test_out_of_range_value_is_rejected`
- `test_missing_field_is_rejected`
- `test_openapi_documents_predict`

Les deux tests qui attendent un code 422 passent pour une mauvaise raison. Le schéma est vide et `extra="forbid"` refuse tout champ inconnu : les 4 champs envoyés sont donc tous considérés comme inconnus. L'API renvoie 422 pour n'importe quel relevé, même valide. Ces tests ne prouvent donc ni que 500 °C est hors bornes, ni qu'un champ manquant est détecté.

## 2. Le schéma d'entrée (`maintai/api/schemas.py`)

| Variable | Type | Minimum (`ge`) | Maximum (`le`) | Justification physique (pas le min / max des données) |
|---|---|---|---|---|
| `temperature_c` | `float` | -40 | 150 | -40 °C couvre un démarrage à froid dans un site très froid. Au-delà de 150 °C, on atteint l'ordre de grandeur des limites d'isolation d'un moteur courant : la valeur indique plutôt un capteur défaillant. |
| `vibration_mm_s` | `float` | 0 | 50 | Une vitesse de vibration mesurée en valeur efficace ne peut pas être négative. 50 mm/s est déjà très au-delà d'un fonctionnement normal, mais la borne laisse passer les vibrations extrêmes, celles qu'il faut justement détecter. |
| `load_pct` | `float` | 0 | 100 | 0 % correspond à une machine sans charge et 100 % à sa capacité nominale. Limite de ce choix : une surcharge réelle au-delà de 100 % serait refusée. |
| `runtime_hours` | `float` | 0 | 100 000 | Un compteur d'heures ne peut pas être négatif. 100 000 h représentent environ 11,4 ans de marche continue, une durée large pour une machine industrielle. |

Après avoir écrit `schemas.py` seul, `python -m pytest` donne encore 3 échecs, avec cette erreur :

```
FAILED tests/test_api.py::test_predict_returns_probability - NotImplementedError
FAILED tests/test_api.py::test_high_risk_reading_raises_alert - NotImplementedError
FAILED tests/test_api.py::test_503_when_model_missing - NotImplementedError
3 failed, 10 passed, 1 warning in 12.44s
```

C'est un bon signe : les relevés valides ne sont plus refusés par la validation et arrivent jusqu'à `predict()`. Il ne reste plus qu'à écrire cette fonction.

## 3. La fonction `predict()` (`maintai/api/main.py`)

Votre code :

```python
predictor = state.get("predictor")
if predictor is None:
    raise HTTPException(
        status_code=503,
        detail="Modèle non chargé : lancez python -m maintai.train puis redémarrez l'API",
    )
return PredictionOut(**predictor.predict(reading.model_dump()))
```

Pourquoi répondre **503** et pas 500 quand le modèle manque ?

Un code 500 signale une erreur imprévue dans le code. Ici, la situation est connue : le service tourne, mais il ne peut pas encore répondre car le modèle n'est pas disponible. Le code 503 le dit clairement au client, et le message indique comment corriger le problème.

> Note : le code cité ici est celui de la version locale. Le code commun de `TP/TP1` renvoie le message plus court « Modèle de prédiction indisponible ». La remarque sur le message qui indique la correction vaut donc pour la version locale.

Pourquoi le modèle est-il chargé **une seule fois**, au démarrage, et pas à chaque requête ?

Lire le modèle sur le disque est lent par rapport à une prédiction. Le charger une seule fois évite de répéter ce coût à chaque requête et garde des temps de réponse courts. Toutes les requêtes utilisent aussi la même version du modèle, et un modèle absent est repéré dès le démarrage, via `/health`.

## 4. Les tests

Dernière ligne de `python -m pytest` à la fin du TP :

```
13 passed, 1 warning in 6.04s
```

## 5. L'API en marche

Requêtes HTTP envoyées à une instance locale de l'API, sur `127.0.0.1` et le port 8765, car le port 8000 était déjà utilisé par une autre application.

| Requête | Code obtenu | Réponse (le corps JSON) |
|---|---|---|
| `GET /health` | 200 | `{"status":"ok","model_loaded":true}` |
| `POST /v1/predict` avec `{"temperature_c": 72.5, "vibration_mm_s": 6.1, "load_pct": 85, "runtime_hours": 15000}` | 200 | `{"failure_probability":0.9553,"alert":true,"threshold":0.3,"model_version":"2026-11-18T21:07:23Z"}` |
| `POST /v1/predict` avec une température de 500 | 422 | `{"detail":[{"type":"less_than_equal","loc":["body","temperature_c"],"msg":"Input should be less than or equal to 150","input":500,"ctx":{"le":150.0}}]}` |
| `POST /v1/predicte` (faute de frappe dans l'adresse) | 404 | `{"detail":"Not Found"}` |

Dans `/docs`, où voit-on les bornes que vous avez choisies ?

Dans le schéma `SensorReading`, visible dans l'onglet **Schema** du corps de `POST /v1/predict` et dans la section **Schemas** en bas de la page. Chaque champ y affiche son type, son minimum, son maximum et sa description.

## 6. Questions de compréhension

1. REST et FastAPI : quelle est la différence ?
   REST est un ensemble de règles pour concevoir une API web : URL claires, méthodes HTTP, codes de statut, JSON. FastAPI est une bibliothèque Python qui permet d'écrire une API suivant ces règles.
2. Le schéma Pydantic sert à trois choses. Lesquelles ?
   Il valide les données reçues, il alimente la documentation `/docs` et il fournit à la fonction un objet Python typé.
3. Dans `http://localhost:8000/v1/predict`, que désignent `localhost`, `8000` et `/v1/predict` ?
   `localhost` désigne la machine elle-même, `8000` le port sur lequel le serveur écoute, et `/v1/predict` la route définie dans le code.
4. Pourquoi `/v1` dans l'adresse ?
   C'est la version de l'API. Le contrat peut évoluer, par exemple avec un `/v2`, sans casser les clients qui utilisent encore `/v1`.
5. Dans la commande `uvicorn maintai.api.main:app`, que désignent `maintai.api.main` et `app` ?
   `maintai.api.main` désigne le fichier `maintai/api/main.py`, écrit avec des points. `app` est la variable de ce fichier qui contient l'application FastAPI.

## 7. Le rendu Git

- Dépôt du groupe : https://github.com/Yasser-Hadli/Applications-Intelligentes, branche `TP`, dossier `TP/TP1`.
- Lien de la pull request : aucune, le groupe dépose directement sur la branche `TP`.
- Le tag `tp1` existe sur GitHub : non.
- Qui a relu la pull request, et quel commentaire a-t-il laissé ?
  Sans objet, car il n'y a pas eu de pull request.

## 8. Ce qui reste flou

**Les tests qui passent ou échouent.** Au départ, il n'était pas évident de comprendre pourquoi le projet affichait « 3 failed » alors que le TP avançait. Ces échecs étaient attendus aux premières étapes, et leur cause changeait entre elles. Le point encore difficile est qu'un test puisse passer pour une mauvaise raison : il reste à apprendre à vérifier ce qu'un test vert prouve réellement.

**`localhost` et les ports.** Au lancement de l'API, une autre application utilisait déjà le port 8000. Il faut donc vérifier quelle application répond, et il est possible de choisir un autre port. Reste à mieux comprendre pourquoi `localhost` et `127.0.0.1` ont pu mener à deux applications différentes.

**Le rôle de l'ADR.** La notion d'ADR et sa différence avec le compte rendu n'étaient pas claires au départ. Un ADR sert à justifier une décision. Reste à mieux savoir quelles décisions méritent d'en avoir un dans un projet.
