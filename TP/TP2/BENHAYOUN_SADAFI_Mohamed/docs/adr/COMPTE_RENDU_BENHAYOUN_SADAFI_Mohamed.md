# Compte rendu · TP2 : tester, sécuriser et appeler l'API MaintAI

**Groupe :** 5IAD G1  ·  **Membre :** BENHAYOUN SADAFI Mohamed  ·  **Coordinateur :** Yasser Hadli  ·  **Date :** 18/11/2026

---

## 1. Activité 1 : casser son API (dans `TP/TP1`)

Requêtes envoyées avec `curl.exe` à l'API locale du TP1, sur le port 8766, car le port 8000 est déjà utilisé par Docker sur ce poste.

| Requête | Ce qui ne va pas | Code prévu | Code obtenu |
|---|---|---|---|
| 1 | champ manquant : `load_pct` est absent | 422 | 422 |
| 2 | valeur absurde : `load_pct` vaut 250 | 422 | 422 |
| 3 | texte au lieu d'un nombre : `temperature_c` vaut `"chaud"` | 422 | 422 |
| 4 | champ en trop : `couleur` vaut `"rouge"` | 422 | 422 |
| 5 | mauvaise route : `POST /v1/predire` | 404 | 404 |

Une de vos prévisions était-elle fausse ? Laquelle, et pourquoi ?

Non, les cinq codes obtenus correspondent aux codes prévus. Un essai complémentaire montre toutefois une limite : `temperature_c` envoyé sous la forme `"60"` est accepté avec un code 200, car Pydantic sait convertir ce texte en nombre. Seul un texte non convertible, comme `"chaud"`, est refusé.

Dernière ligne de `python -m pytest` avec `tests/test_casser.py` (dans `TP/TP1`) :

```
18 passed, 1 warning in 11.83s
```

À quoi sert `@pytest.mark.parametrize` ? Combien de tests pytest compte-t-il pour votre fonction, et pourquoi ?

Ce décorateur lance la même fonction de test avec plusieurs jeux de données, sans recopier le code. pytest compte 5 tests pour cette fonction, car la liste `CAS_REFUSES` contient 5 lignes. Chaque cas apparaît séparément dans le rapport, donc on voit lequel échoue.

## 2. ruff

Sortie de `ruff check .` dans `TP/TP1` (ruff 0.17.5) :

```
UP017 [*] Use `datetime.UTC` alias
  --> maintai\train.py:58:36
   |
56 |         "target": TARGET,
57 |         "threshold": settings.threshold,
58 |         "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
   |                                    ^^^^^^^^^^^^
59 |         "sklearn_version": sklearn.__version__,
60 |     }
   |
help: Convert to `datetime.UTC` alias
   |
5  | import json
   - from datetime import datetime, timezone
6  + from datetime import datetime, timezone, UTC
7  |
--------------------------------------------------------------------------------
57 |         "threshold": settings.threshold,
   -         "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
58 +         "trained_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
59 |         "sklearn_version": sklearn.__version__,
   |

Found 1 error.
[*] 1 fixable with `--fix`.
```

Quelle différence entre `ruff check .` et `python -m pytest` ? Que vérifie chacun ?

`ruff check .` lit le code sans l'exécuter et signale ce qui est mal écrit ou peut être simplifié : imports, variables inutiles, écritures dépassées. `python -m pytest` exécute le code et vérifie qu'il donne les résultats attendus. Les deux sont complémentaires : ici, les 18 tests passent alors que ruff signale une écriture à moderniser.

Dans le TP2, `ruff check . --fix` a été lancé seulement après le sprint 1. Il a regroupé les imports de `fastapi` et remplacé `timezone.utc` par `UTC`. `ruff check .` y donne ensuite `All checks passed!`.

## 3. Mise en place du TP2 (dans `TP/TP2`)

Dernière ligne de `python -m pytest` juste après la mise en place :

```
2 failed, 16 passed, 1 warning in 12.40s
```

Quels sont les deux tests qui échouent, et que reçoivent-ils au lieu de 401 ?

Ce sont `test_missing_key_is_rejected` et `test_wrong_key_is_rejected`. Ils reçoivent 200 au lieu de 401, car l'API ne contrôle pas encore la clé.

## 4. La clé API (sprint 1)

Votre fonction `require_api_key()` (`maintai/api/security.py`) :

```python
def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Refuse la requête si la clé est absente ou fausse."""
    expected = get_settings().api_key
    if expected is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="La clé API du serveur n'est pas configurée",
        )
    # Comparaison en octets : compare_digest refuse les textes non ASCII
    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API absente ou invalide",
        )
```

Votre ligne `router = ...` dans `maintai/api/main.py` :

```python
router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
```

Après avoir écrit `security.py` seul (sans toucher au router), combien de tests échouent ? Pourquoi ?

Deux tests échouent encore (`2 failed, 16 passed, 1 warning in 13.94s`). La fonction existe, mais aucune route ne l'appelle : les requêtes sans clé ou avec une fausse clé reçoivent toujours 200.

Dernière ligne de `python -m pytest` à la fin du sprint 1 :

```
18 passed, 1 warning in 5.94s
```

Requêtes envoyées avec `curl.exe` à uvicorn lancé avec `--env-file .env`, sur le port 8765 et sans `--reload`. Le relevé normal est `{"temperature_c": 60.0, "vibration_mm_s": 2.0, "load_pct": 50.0, "runtime_hours": 5000}`.

| Requête | Code obtenu | Réponse (le corps JSON) |
|---|---|---|
| `GET /health`, sans clé | 200 | `{"status":"ok","model_loaded":true}` |
| `POST /v1/predict`, relevé normal, sans clé | 401 | `{"detail":"Clé API absente ou invalide"}` |
| `POST /v1/predict`, relevé normal, fausse clé | 401 | `{"detail":"Clé API absente ou invalide"}` |
| `POST /v1/predict`, relevé normal, bonne clé | 200 | `{"failure_probability":0.0027,"alert":false,"threshold":0.3,"model_version":"2026-11-18T23:36:06Z"}` |
| `POST /v1/predict`, 500 °C, sans clé | 401 | `{"detail":"Clé API absente ou invalide"}` |
| `POST /v1/predict`, 500 °C, bonne clé | 422 | `{"detail":[{"type":"less_than_equal","loc":["body","temperature_c"],"msg":"Input should be less than or equal to 150","input":500,"ctx":{"le":150.0}}]}` |

500 °C sans clé et 500 °C avec la clé ne donnent pas le même code. Pourquoi ?

La clé est contrôlée avant les données du relevé, et le premier contrôle qui échoue donne la réponse. Sans clé, la requête s'arrête à 401 et la température n'est pas évaluée. Avec la bonne clé, elle passe ce contrôle et la validation refuse 500 °C avec un 422.

Relancez uvicorn **sans** `--env-file .env` et envoyez un relevé avec votre clé. Quel code obtenez-vous, et pourquoi ce n'est pas 401 ?

Le code obtenu est 500, avec `{"detail":"La clé API du serveur n'est pas configurée"}` (essai sur le port 8767). Le serveur n'a aucune clé à laquelle comparer celle du client : le problème vient de sa configuration. Un 401 accuserait à tort le client d'avoir envoyé une mauvaise clé.

## 5. La page web (sprint 2)

Votre code `fetch` (le contenu du `addEventListener` dans `web/index.html`) :

```javascript
const reading = Object.fromEntries(FEATURES.map(f => [f, Number(document.getElementById(f).value)]));
const url = document.getElementById("url").value.trim().replace(/\/+$/, "");
const cle = document.getElementById("key").value;
try {
  const resp = await fetch(`${url}/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-API-Key": cle },
    body: JSON.stringify(reading),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map(e => `${e.loc.at(-1)} : ${e.msg}`).join(" ; ")
      : data.detail ?? "réponse inattendue";
    show("err", `Erreur ${resp.status} : ${detail}`);
    return;
  }
  const pct = (data.failure_probability * 100).toFixed(1);
  const seuil = (data.threshold * 100).toFixed(0);
  const titre = data.alert ? "Alerte" : "Pas d'alerte";
  show(data.alert ? "alert" : "ok", `${titre} : probabilité de panne ${pct} % (seuil ${seuil} %)`);
} catch (e) {
  show("err", `API injoignable ou bloquée par CORS : ${e.message}`);
}
```

Ce qu'affiche la page (servie sur le port 5500, avec `http://localhost:8765` comme URL de l'API) :

| Essai | Ce qu'affiche la page |
|---|---|
| relevé normal, bonne clé | `Pas d'alerte : probabilité de panne 0.3 % (seuil 30 %)` |
| 500 °C, bonne clé | `Erreur 422 : temperature_c : Input should be less than or equal to 150` |
| relevé normal, fausse clé | `Erreur 401 : Clé API absente ou invalide` |

**Expérience CORS.** Servez la page sur le port 5501. Que dit la page, et que dit la console (F12) ?

La page affiche `API injoignable ou bloquée par CORS : Failed to fetch`. La console indique :

```
Access to fetch at 'http://localhost:8765/v1/predict' from origin 'http://localhost:5501' has been blocked by CORS policy: Response to preflight request doesn't pass access control check: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

Dans l'onglet réseau, la requête `OPTIONS` reçoit 400 et le `POST` échoue avec `net::ERR_FAILED`.

Pourquoi `curl.exe` obtient-il une réponse alors que la page du port 5501 est bloquée ?

CORS est appliqué par le navigateur, pour protéger l'utilisateur d'une page web. `curl.exe` n'envoie pas de requête préalable `OPTIONS` et ne vérifie pas les en-têtes CORS. Avec l'en-tête `Origin: http://localhost:5501` et la bonne clé, il reçoit bien 200, sans en-tête `Access-Control-Allow-Origin`. CORS ne remplace donc pas la clé API.

## 6. Les tests de l'activité 1 dans le TP2

Copiez `test_casser.py` dans `TP/TP2/tests`. Sans `headers=HEADERS`, combien de tests échouent, et avec quel code ? Lequel des cinq cas passe quand même, et pourquoi ?

Quatre tests échouent (`4 failed, 19 passed`) : ils attendent 422 et reçoivent 401, car la clé est contrôlée avant les données. Le cas « mauvaise route » passe quand même : `/v1/predire` n'existe pas, donc aucune dépendance du router n'est exécutée et l'API répond 404.

Dernière ligne de `python -m pytest` avec `headers=HEADERS` :

```
23 passed, 1 warning in 4.74s
```

## 7. Questions de compréhension

1. Quelle différence entre 401 et 422 ? Et entre 401 et 403 ?
   401 signifie que l'appelant n'est pas authentifié : sa clé est absente ou fausse. 422 signifie que l'appelant a passé ce contrôle, mais que les données envoyées sont invalides. 403 signifie que l'appelant est identifié, mais qu'il n'a pas le droit de faire cette action.
2. Pourquoi la clé n'est-elle ni dans le code ni dans Git ? Que contiennent `.env` et `.env.example`, et lequel est commité ?
   Une clé commitée reste dans l'historique de Git, même si le fichier est supprimé ensuite, et toute personne ayant accès au dépôt peut la lire. `.env` contient la vraie clé et les origines autorisées, et il reste sur la machine. `.env.example` contient les mêmes variables avec une fausse valeur : c'est lui qui est commité.
3. Pourquoi la clé voyage-t-elle dans un en-tête et pas dans l'URL ?
   Une URL est enregistrée à de nombreux endroits, comme les journaux du serveur et l'historique du navigateur. Une clé placée dans l'URL y serait donc conservée et pourrait fuiter.
4. Que fait `secrets.compare_digest` de plus qu'un simple `==` ?
   Sa durée ne dépend pas de la position du premier caractère différent. `==` peut s'arrêter dès la première différence, et le temps de réponse pourrait alors aider un attaquant à deviner la clé caractère par caractère.
5. Qu'est-ce qu'une « origine » ? `http://localhost:5500` et `http://localhost:8000` sont-elles la même origine ?
   Une origine est l'ensemble formé par le schéma, le domaine et le port. Ces deux adresses ne sont pas la même origine, car leurs ports sont différents.

## 8. Le rendu Git

- Dépôt du groupe : https://github.com/Yasser-Hadli/Applications-Intelligentes, branche `TP`, dossier `TP/TP2_BENHAYOUN_SADAFI_Mohamed`.
- Lien de la pull request : aucune, le groupe dépose directement sur la branche `TP`.
- Le tag `tp2` existe sur GitHub : non.
- `git status` avant le commit : le fichier `.env` apparaissait-il ? non
- Qui a relu la pull request, et quel commentaire a-t-il laissé ?
  Sans objet, car il n'y a pas eu de pull request.

## 9. Ce qui reste flou

Le point bloquant est le port de l'API. Sur ce poste, un conteneur Docker d'un autre projet publie déjà le port 8000. La page appelait `http://localhost:8000` par défaut : elle contactait donc ce service et non l'API du TP2. Comme il répond 404 à la requête `OPTIONS`, le navigateur bloquait l'appel et la page affichait `API injoignable ou bloquée par CORS : Failed to fetch`. La liste `MAINTAI_CORS_ORIGINS` était correcte : il ne fallait pas la modifier.

Un port ouvert par Docker peut donc rester actif sans qu'on y pense et avoir la priorité sur l'application. Sur ce poste, un serveur de test démarre sans erreur sur `127.0.0.1:8000`, mais `localhost` désigne d'abord `::1` (IPv6), une adresse où Docker écoute : c'est donc Docker qui répond à `http://localhost:8000`. Solution retenue : lancer l'API sur le port 8765 et utiliser `http://localhost:8765` dans la page.
