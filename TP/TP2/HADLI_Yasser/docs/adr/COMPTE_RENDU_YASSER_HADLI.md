# Compte rendu · TP2 : tester, sécuriser et appeler l’API MaintAI

**Groupe :**YASSER HADLI - ADAM BILOUL  **Date :** 4 novembre 2026

Dans ce TP, j’ai complété le starter MaintAI pour protéger les prédictions avec une clé API et appeler le modèle depuis une page web. J’ai gardé l’architecture et les tests existants. Je précise les étapes non vérifiées pour ne pas confondre les résultats attendus avec les résultats obtenus.

## 1. Activité 1 : casser son API (dans TP1)

Ce tableau reprend les résultats déjà renseignés dans mon compte rendu. Les essais TP1 n’ont pas été relancés pendant la finalisation du TP2.

| Requête | Problème | Code prévu | Code déjà renseigné |
|---|---|---|---|
| `POST /v1/predict` sans `load_pct` | Champ obligatoire manquant | 422 | 422 |
| `POST /v1/predict` avec `temperature_c: 500` | Valeur hors limites | 422 | 422 |
| `POST /v1/predict` avec `vibration_mm_s: "bonjour"` | Mauvais type | 422 | 422 |
| `POST /v1/predict` avec `pression: 10` en plus | Champ inconnu | 422 | 422 |
| `POST /v1/prediction` | Mauvaise route | 404 | 404 |

D’après ce tableau, mes prévisions étaient correctes. Pydantic refuse les données invalides avec 422. Le champ supplémentaire est interdit grâce à `extra="forbid"`. La mauvaise route donne 404 car elle n’existe pas.

**Dernière ligne Pytest dans TP1 avec `test_casser.py` :** sortie non fournie, et TP1 non exécuté pendant ce travail. Je ne peux donc pas donner de sortie réelle.

`@pytest.mark.parametrize` permet de lancer une même fonction avec plusieurs jeux de données. Avec cinq cas, Pytest compte cinq tests, même si on écrit une seule fonction.

## 2. Ruff

La sortie TP1 déjà présente dans le document signalait les erreurs suivantes :

```text
I001 : imports à organiser dans demo_app.py
UP017 : utiliser datetime.UTC dans maintai/train.py
C408 : remplacer dict() par un dictionnaire littéral
       dans verification/verifier_http.py, lignes 11, 44 et 48
Found 5 errors.
```

Ce bloc résume la sortie précédente ; ce n’est pas une nouvelle exécution dans TP1.

Ruff vérifie la qualité du code : imports, variables inutilisées et écritures à simplifier. Pytest exécute des tests pour vérifier le comportement du programme. Un code peut donc passer Ruff et avoir quand même un bug.

Dans TP2, j’ai regroupé les imports de `main.py` et remplacé `timezone.utc` par `UTC` dans `train.py`. Résultat réel après correction :

```text
ruff check .
All checks passed!
```

## 3. Mise en place du TP2

Je n’ai pas de sortie enregistrée avant les modifications. La première exécution enregistrée pendant ce travail a été faite après la correction de la sécurité :

```text
18 passed, 2 warnings in 18.69s
```

Dans le starter non protégé, les deux tests concernés sont `test_missing_key_is_rejected` et `test_wrong_key_is_rejected`. Ils attendent 401, mais la route accepte les requêtes et renvoie 200 si le modèle est disponible. Cette explication vient du code initial, pas d’une exécution mesurée avant correction.

## 4. La clé API (sprint 1)

Voici la fonction de `maintai/api/security.py` :

```python
def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Refuse la requête si la clé est absente ou fausse."""
    expected_key = get_settings().api_key
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Clé API non configurée sur le serveur",
        )
    if x_api_key is None or not secrets.compare_digest(
        x_api_key.encode("utf-8"), expected_key.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API absente ou incorrecte",
        )
```

FastAPI récupère `X-API-Key` avec `Header`. La fonction lit la clé attendue dans la configuration. Si le serveur n’a aucune clé, elle renvoie 500. Si le client n’envoie pas la bonne clé, elle renvoie 401. Sinon, la requête continue. La comparaison utilise `secrets.compare_digest` sur les clés encodées en UTF-8.

Dans `main.py`, j’ai protégé toutes les routes `/v1` :

```python
router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
```

Si je complète seulement `security.py`, les deux tests de clé doivent encore échouer : sans dépendance dans le routeur, FastAPI n’appelle pas la fonction. Cette étape intermédiaire n’a pas été testée séparément.

Après la correction de sécurité, les 18 tests passent. La dernière sortie réelle, après les corrections Ruff, est :

```text
18 passed, 2 warnings in 5.79s
```

### Requêtes HTTP vérifiées

Le relevé normal utilisé est :

```json
{"temperature_c":60,"vibration_mm_s":2,"load_pct":50,"runtime_hours":5000}
```

Les requêtes suivantes ont été envoyées directement à l’API locale, sans passer par Swagger :

| Requête | Code obtenu | Corps JSON |
|---|---|---|
| `GET /health`, sans clé | 200 | `{"status":"ok","model_loaded":true}` |
| Prédiction normale, sans clé | 401 | `{"detail":"Clé API absente ou incorrecte"}` |
| Prédiction normale, fausse clé | 401 | `{"detail":"Clé API absente ou incorrecte"}` |
| Prédiction normale, bonne clé | 200 | `{"failure_probability":0.0027,"alert":false,"threshold":0.3,"model_version":"2026-11-04T17:44:02Z"}` |
| 500 °C, sans clé | 401 | `{"detail":"Clé API absente ou incorrecte"}` |
| 500 °C, bonne clé | 422 | `{"detail":[{"type":"less_than_equal","loc":["body","temperature_c"],"msg":"Input should be less than or equal to 150","input":500,"ctx":{"le":150.0}}]}` |

Sans clé, la dépendance bloque la requête. Avec la bonne clé, la validation indique que 500 °C dépasse la limite de 150 °C.

**Sans clé configurée sur le serveur :** le code 500 a été vérifié avec le client de test en retirant `MAINTAI_API_KEY` de l’environnement. C’est une erreur de configuration du serveur, pas une erreur du client. Un lancement sans `--env-file .env` ne donne cette erreur que si la variable n’existe pas déjà dans l’environnement du terminal. Ce lancement n’a pas été testé séparément.

Après le changement de clé, d’anciens processus utilisaient encore l’ancienne valeur. Après leur arrêt et le redémarrage de l’API, la nouvelle clé donne 200 et l’ancienne donne 401. J’ai compris qu’il faut redémarrer le serveur après une modification de `.env`.

La clé reste dans `.env` et n’est pas recopiée dans ce compte rendu. `.env.example` garde une valeur de démonstration.

## 5. La page web (sprint 2)

J’ai complété le gestionnaire du bouton « Prédire » sans changer l’interface :

```javascript
document.getElementById("go").addEventListener("click", async () => {
  const reading = Object.fromEntries(FEATURES.map(f => [f, Number(document.getElementById(f).value)]));
  const url = document.getElementById("url").value.trim().replace(/\/+$/, "");
  const key = document.getElementById("key").value;
  try {
    const resp = await fetch(`${url}/v1/predict`, {
      method: "POST",
      headers: {"Content-Type": "application/json", "X-API-Key": key},
      body: JSON.stringify(reading),
    });
    const data = await resp.json();
    if (!resp.ok) {
      const message = Array.isArray(data.detail)
        ? data.detail.map(error => `${error.loc.join(".")} : ${error.msg}`).join(" ; ")
        : (data.detail || "Erreur de l'API");
      show("err", `Erreur ${resp.status} : ${message}`);
      return;
    }
    show(data.alert ? "alert" : "ok",
      `Probabilité de panne : ${(data.failure_probability * 100).toFixed(2)} % — Alerte : ${data.alert ? "oui" : "non"}`);
  } catch (error) {
    show("err", "Communication impossible : vérifiez que l'API est démarrée, que son URL est correcte et que CORS autorise l'origine de cette page.");
  }
});
```

Le code récupère l’URL, la clé et les quatre valeurs. Il envoie le relevé en JSON avec `fetch`, puis lit la réponse. Il affiche la probabilité et l’alerte en cas de succès, ou le code et le message d’erreur. Le `catch` gère les problèmes de communication.

Le site a répondu 200 sur le port 5500. Les affichages après clic n’ont pas été vérifiés dans un navigateur pendant ce travail. Pour le relevé de la section 4, voici ce que le code doit afficher :

| Essai | Affichage attendu, non relevé dans le navigateur |
|---|---|
| Relevé normal, bonne clé | `Probabilité de panne : 0.27 % — Alerte : non` |
| 500 °C, bonne clé | `Erreur 422 : body.temperature_c : Input should be less than or equal to 150` |
| Relevé normal, fausse clé | `Erreur 401 : Clé API absente ou incorrecte` |

### Expérience CORS

La configuration autorise uniquement `http://localhost:5500` et `http://127.0.0.1:5500`, les méthodes GET et POST et les en-têtes `Content-Type` et `X-API-Key`.

Les prérequêtes OPTIONS ont été vérifiées avec le client de test :

| Origine | Code obtenu | `Access-Control-Allow-Origin` |
|---|---|---|
| `http://localhost:5500` | 200 | `http://localhost:5500` |
| `http://127.0.0.1:5500` | 200 | `http://127.0.0.1:5500` |
| `http://localhost:5501` | 400 | Absent |

Sur le port 5501, le navigateur devrait donc bloquer l’appel. La page afficherait « Communication impossible… » et la console signalerait une erreur CORS. Je n’ai pas de capture ni de message F12 réellement relevé pour cet essai.

`curl.exe` peut lire la réponse car CORS est appliqué par le navigateur. CORS ne remplace pas la clé API : même avec curl, la route protégée demande la bonne clé.

### Démarrage dans deux terminaux

Depuis la racine du projet, dans le sous-dossier `maintai-starter-tp2` :

Terminal 1 :

```powershell
uvicorn maintai.api.main:app --reload --host 127.0.0.1 --port 8000 --env-file .env
```

Terminal 2 :

```powershell
python -m http.server 5500 -d web
```

La page est sur `http://localhost:5500` et Swagger sur `http://localhost:8000/docs`. Ces deux adresses ont répondu 200 lors des vérifications. Pour saisir la clé, j’ouvre `.env` et je copie la valeur après `MAINTAI_API_KEY=` dans le champ de la page.

## 6. Les tests de l’activité 1 dans le TP2

Le fichier `tests/test_casser.py` n’est pas présent. Il n’a pas été copié depuis TP1 et aucun résultat à 23 tests n’a été obtenu.

Pour les cinq cas de la section 1, sans `headers=HEADERS`, les quatre tests sur `/v1/predict` doivent échouer : ils reçoivent 401 au lieu de 422. Le test de mauvaise route doit toujours recevoir 404 et passer, car la route n’existe pas. Ce sont les résultats attendus d’après le code.

Avec la bonne clé, les quatre cas de données invalides doivent recevoir 422. Ajouter ces cinq tests au starter donnerait un total attendu de 23 tests. Cela reste à vérifier après l’ajout réel du fichier.

La dernière ligne réelle pour les tests présents est :

```text
18 passed, 2 warnings in 5.79s
```

Les tests existants n’ont pas été modifiés. Ils vérifient aussi le code 503 quand le modèle est absent. Les données et le modèle local ont été préparés avec :

```powershell
python -m maintai.dataset
python -m maintai.train
```

## 7. Questions de compréhension

1. **Quelle différence entre 401 et 422 ? Et entre 401 et 403 ?**

   401 indique une authentification absente ou incorrecte. 422 indique que les données ne respectent pas le schéma. 403 correspond à un accès refusé à un utilisateur identifié qui n’a pas les droits nécessaires. Dans ce TP, une clé absente ou fausse donne 401.

2. **Pourquoi la clé n’est-elle ni dans le code ni dans Git ?**

   Si on publie la clé, une autre personne peut appeler l’API. `.env` contient la configuration locale, dont la clé utilisée par le serveur. `.env.example` montre les variables à remplir avec des valeurs de démonstration. On peut commiter `.env.example`, mais `.env` est exclu par `.gitignore`.

3. **Pourquoi la clé voyage-t-elle dans un en-tête et pas dans l’URL ?**

   Une URL peut rester dans l’historique ou apparaître dans des journaux. L’en-tête évite de mettre la clé dans l’adresse. Cela ne chiffre pas la clé : pour un déploiement, il faudrait utiliser HTTPS.

4. **Que fait `secrets.compare_digest` de plus qu’un simple `==` ?**

   Cette fonction compare les secrets en limitant les informations révélées par le temps de comparaison. Une comparaison classique peut s’arrêter dès qu’une différence est trouvée.

5. **Qu’est-ce qu’une origine ?**

   Une origine correspond au protocole, au nom d’hôte et au port. `http://localhost:5500` et `http://localhost:8000` sont deux origines différentes car le port change. C’est pour cela qu’on configure CORS entre la page et l’API.

## 8. Le rendu Git

- **Lien de la pull request :** non fourni ; à renseigner après sa création réelle.
- **Tag `tp2` sur GitHub :** non vérifié, aucun tag créé pendant ce travail.
- **Présence de `.env` dans `git status` :** impossible à confirmer ici. La commande indique `fatal: not a git repository (or any of the parent directories): .git`. La ligne `.env` est bien présente dans `.gitignore`.
- **Relecture de la PR :** aucun nom ni commentaire de relecture fourni pour le moment.

Il reste à faire le rendu depuis un vrai dépôt Git, vérifier que `.env` n’est pas suivi, créer la PR et le tag demandé, puis noter la relecture.

## 9. Ce qui reste flou

J’ai mieux compris la différence entre une erreur de clé, une erreur de données et une erreur de configuration du serveur. Le problème de clé m’a aussi montré qu’un ancien processus peut garder l’ancienne configuration.

Le résultat final est de 18 tests réussis. Les deux avertissements concernent les dépendances de test Starlette/httpx et AnyIO ; ils ne font pas échouer les tests. Les étapes restantes sont la vérification visuelle, les cinq tests supplémentaires si demandés, les informations du groupe et le rendu GitHub.
