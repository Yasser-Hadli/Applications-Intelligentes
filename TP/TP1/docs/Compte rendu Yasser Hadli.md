# Compte rendu · TP1 : une API de prédiction pour MaintAI

**Groupe :** [Yasser Hadli - Adam Biloul
]  ·  **Membres :** Yasser Hadli (rendu individuel)  ·  **Date :** 28 octobre 2026

**Module :** Développement et déploiement d'applications intelligentes — EMSI.

**TP :** TP1 — Une API de prédiction pour MaintAI.

**Dépôt cible :** https://github.com/Yasser-Hadli/Applications-Intelligentes ; branche commune prévue : `TP`.

> Compte rendu établi à partir des exécutions locales réelles. Les journaux complets sont conservés dans `verification/`, exclu de Git. Le modèle du professeur et l'ADR sont préservés.

---

## 1. Mise en place

Dernière ligne de `python -m pytest`, juste après la mise en place (rien n'est encore écrit) :

```
=================== 3 failed, 10 passed, 1 warning in 9.60s ===================
```

Dans `tests/test_api.py`, 4 tests sur 7 passent déjà alors que rien n'est écrit. Lesquels ? Pour les deux tests qui attendent un code 422, expliquez pourquoi ils passent « pour une mauvaise raison » (indice : regardez `extra="forbid"` dans `schemas.py`).

L'interpréteur système est Python 3.11.0 et ne disposait pas de FastAPI lors de la vérification antérieure de `demo_app.py`. Le TP utilise donc explicitement `.venv\Scripts\python.exe`, également en version 3.11.0. La commande `.venv\Scripts\python.exe -m pip install -r requirements.txt` a confirmé que toutes les dépendances étaient déjà installées. Les versions fixées comprennent numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, joblib 1.5.3, FastAPI 0.141.1, Uvicorn 0.54.0, httpx 0.28.1 et pytest 9.1.1 ; Pydantic installé est en version 2.14.1.

Avant toute modification du code de ce TP, les quatre tests API déjà réussis sont `test_health`, `test_out_of_range_value_is_rejected`, `test_missing_field_is_rejected` et `test_openapi_documents_predict`. Les six tests de données et d'entraînement passent aussi. Les trois échecs concernent `test_predict_returns_probability` (422 au lieu de 200), `test_high_risk_reading_raises_alert` (`KeyError: 'alert'`, car la réponse contient des erreurs de validation) et `test_503_when_model_missing` (422 au lieu de 503).

Avec un `SensorReading` vide et `extra="forbid"`, chaque clé reçue est inconnue : Pydantic rejette donc les corps de requête avant l'appel à la route. Les deux tests 422 passent pour une mauvaise raison : ils ne prouvent encore ni la détection d'une température excessive, ni le caractère obligatoire de `load_pct`. La comparaison du seul code HTTP masque la nature réelle de l'erreur.

L'analyse du projet identifie deux TODO fonctionnels : les quatre champs du schéma et le corps de `predict()`. La génération synthétique, le découpage stratifié, le pipeline `StandardScaler` puis `LogisticRegression`, les artefacts et les fixtures sont déjà implémentés. Les fixtures entraînent sur 2 000 relevés avec une graine 0 dans un dossier temporaire ; l'exécution manuelle utilise 5 000 relevés avec la graine 42. Les tests et le code ML restent inchangés.

## 2. Le schéma d'entrée (`maintai/api/schemas.py`)

| Variable | Type | Minimum (`ge`) | Maximum (`le`) | Justification physique (pas le min / max des données) |
|---|---|---|---|---|
| `temperature_c` | float | -40 | 150 | Plage de fonctionnement retenue pour une machine industrielle ; accepte le froid et un échauffement important, rejette 500 °C. |
| `vibration_mm_s` | float | 0 | 50 | Amplitude de vitesse vibratoire positive ou nulle ; plafond de plausibilité choisi pour la démo, autorisant des vibrations sévères. |
| `load_pct` | float | 0 | 100 | Pourcentage de capacité nominale ; ce contrat suppose l'absence de surcharge supérieure à 100 %. |
| `runtime_hours` | float | 0 | 100000 | Compteur cumulatif non négatif ; plafond d'environ 11,4 années de marche continue retenu pour le parc simulé. |

Après avoir écrit `schemas.py` seul, que donne `python -m pytest`, et quel est le message d'erreur des tests qui échouent ? Pourquoi est-ce un bon signe ?

```
================== 3 failed, 10 passed, 1 warning in 13.66s ===================
```

Les trois mêmes tests échouent désormais avec `NotImplementedError`, levée dans `maintai/api/main.py`. C'est une progression cohérente : `VALID` et `HIGH_RISK` passent la validation et atteignent la fonction encore vide. Les tests 422 passent maintenant pour les raisons attendues : violation de la borne supérieure pour 500 °C et absence de `load_pct`. Le nombre de tests réussis ne change donc pas, mais la cause des échecs établit que la première étape est correcte. Les bornes inclusives constituent le contrat physique retenu pour cette démo ; elles doivent être adaptées aux spécifications du capteur et de la machine avant un usage industriel, et ne sont pas déduites des extrêmes du jeu synthétique.

## 3. La fonction `predict()` (`maintai/api/main.py`)

Votre code :

```python
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle de prédiction indisponible")
    return PredictionOut(**predictor.predict(reading.model_dump()))
```

Pourquoi répondre **503** et pas 500 quand le modèle manque ?

Le code 503 (`Service Unavailable`) décrit une indisponibilité du service de prédiction : les artefacts nécessaires ne sont pas disponibles, alors que l'API peut répondre à `/health`. Une erreur 500 décrit une erreur interne non traitée ; ici, l'absence du modèle est un état prévu et traité explicitement. Le test dédié vérifie effectivement ce comportement.

Pourquoi le modèle est-il chargé **une seule fois**, au démarrage, et pas à chaque requête ?

Le lifespan charge le pipeline et ses métadonnées une fois par processus serveur. Les requêtes réutilisent ensuite le prédicteur en mémoire, ce qui évite de relire et désérialiser les fichiers à chaque appel et réduit la latence. L'ordre des variables et le seuil restent ceux de l'artefact chargé. Avec plusieurs workers, chaque processus dispose de sa propre instance ; un redémarrage ou un rechargement crée une nouvelle instance.

## 4. Les tests

Dernière ligne de `python -m pytest` à la fin du TP :

```
======================== 13 passed, 1 warning in 6.60s ========================
```

Les sept tests API, les trois tests de données et les trois tests d'entraînement réussissent sans modification des tests fournis. Un avertissement demeure : ``StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.`` Il provient des dépendances installées et ne fait échouer aucun test. Les versions du professeur sont conservées.

## 5. L'API en marche

Après `python -m maintai.dataset` et `python -m maintai.train`, lancez `uvicorn maintai.api.main:app --reload`, puis envoyez ces requêtes depuis `/docs` (ou avec `curl.exe`).

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

Dans `/docs`, où voit-on les bornes que vous avez choisies ?

Dans `/docs`, ouvrir `Schemas`, puis `SensorReading` et les propriétés des capteurs : les contraintes `minimum` et `maximum`, les unités dans les descriptions et les champs obligatoires proviennent du schéma OpenAPI. Le corps de `POST /v1/predict` référence ce schéma et propose l'exemple conservé. La vérification HTTP a obtenu 200 pour `/docs` et détecté `SwaggerUIBundle` dans son HTML ; `/openapi.json` confirme les quatre champs requis, leurs minima/maxima et `additionalProperties: false`. Aucun clic dans un navigateur n'a été simulé : les requêtes ont été envoyées par httpx au serveur réel.

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

- Lien de la pull request : [À COMPLÉTER après publication sur GitHub]. Aucune pull request créée pendant cette intervention.
- Le tag `tp1` existe sur GitHub : [À VÉRIFIER après publication]. Aucun tag créé ; état distant non consulté.
- Qui a relu la pull request, et quel commentaire a-t-il laissé ?
  [À COMPLÉTER après publication] : nom du relecteur et commentaire réel ; aucune relecture inventée.

## 8. Ce qui reste flou

Point à approfondir : la calibration de la probabilité et le choix du seuil d'alerte en fonction du coût des faux négatifs et des faux positifs. Le seuil 0,3 est fourni par la configuration ; sa pertinence industrielle demanderait des données représentatives et une analyse métier. Il s'agit d'un axe d'étude, sans prétendre qu'un incident de calibration a été observé pendant ce TP.

Le dossier de travail ne contient pas de dépôt Git (`git status` renvoie `not a git repository`). L'intégration ultérieure au dépôt existant et à la branche `TP` reste à réaliser après validation ; aucune branche, aucun commit, aucun push et aucune publication n'ont été effectués.

Protection locale : `.gitignore` exclut intégralement `data/`, `models/`, `.venv/`, les caches Python/pytest, `.env`, `.env.*`, les fichiers `.pem` et `.key`, ainsi que les journaux locaux `verification/`. Les artefacts utiles restent sur disque. Ces règles préviennent l'ajout ordinaire de fichiers non suivis ; elles ne retirent pas de fichiers déjà suivis dans un autre dépôt et ne bloquent pas `git add -f`. Lors de l'intégration, il faudra contrôler les fichiers suivis et la zone de staging du dépôt cible avant publication.

L'ADR `docs/adr/000-modele.md` conserve exactement son empreinte SHA-256 initiale : `47D775980E9F28EC06CF4CD43E7FA2E6531F717261F479D6AC61C7C91DEFD11C`. Le modèle `docs/COMPTE_RENDU.md`, les tests et les algorithmes ML n'ont pas été modifiés. `demo_app.py`, modifié lors de la demande précédente, n'est pas modifié pendant cette réalisation du TP.
