# Compte rendu · TP3 : une interface Streamlit pour MaintAI

**Groupe :** 5IAD G1  ·  **Membre :** BENHAYOUN SADAFI Mohamed  ·  **Coordinateur :** Yasser Hadli  ·  **Date :** 20/11/2026

---

## 1. Mise en place (dans `TP/TP3`)

Dernière ligne de `python -m pytest` juste après la mise en place :

```
5 failed, 18 passed, 1 warning in 34.49s
```

Quels fichiers de tests échouent, et quelle erreur donnent-ils ?

Deux fichiers échouent, soit 5 tests sur 23 :

- `tests/test_client.py` : les 3 tests (`test_client_returns_prediction`, `test_client_wrong_key_raises_401`, `test_client_bad_reading_raises_422`) donnent `NotImplementedError: TP3 sprint 2 : MaintaiClient.predict`. La méthode `predict` du client n'est pas encore écrite et lève volontairement cette exception.
- `tests/test_ui.py` : les 2 tests (`test_ui_shows_the_data`, `test_ui_shows_the_prediction`) donnent `IndexError: list index out of range`. Les tests lisent `at.metric[0]` et `at.metric[-1]`, mais l'application n'affiche encore aucun `st.metric` : les deux onglets ne contiennent qu'un `st.info(...)`.

Les 18 autres tests (`test_api.py`, `test_data.py`, `test_security.py`, `test_train.py`) passent déjà : ils portent sur le code des TP1 et TP2, fourni complet dans le starter. Ils servent de tests de non-régression. Le seul avertissement est un `StarletteDeprecationWarning` qui vient du `TestClient` de FastAPI, pas de notre code.

## 2. Lancer Streamlit

Lancez d'abord `streamlit run ui/app.py`. Que s'affiche-t-il dans le navigateur ?

```
ModuleNotFoundError: No module named 'maintai'
Traceback:
File "C:\Users\user\Desktop\5 IAD G1 Developpement applications intelligentes\Tp\TP3\maintai-starter-tp3\ui\app.py", line 9, in <module>
    from maintai.client import APIError, MaintaiClient
```

Lancez ensuite `python -m streamlit run ui/app.py`. Pourquoi cette commande fonctionne-t-elle et pas la première ?

Python cherche les modules dans les dossiers de `sys.path`. Avec `streamlit run ui/app.py`, Streamlit ajoute à `sys.path` le **dossier du script**, c'est-à-dire `ui/`. Or le paquet `maintai/` se trouve à la racine du projet, pas dans `ui/` : `from maintai.client import ...` échoue donc à la ligne 9.

Avec `python -m streamlit run ui/app.py`, c'est Python qui lance Streamlit comme un module. L'option `-m` place le **dossier courant**, la racine du projet, en tête de `sys.path`. `maintai/` y est, donc l'import fonctionne.

Autre intérêt de `python -m` : on utilise le Streamlit du Python actif. Une fois le venv activé, c'est bien celui du projet (Streamlit 1.66.1) et pas une autre installation de la machine.

## 3. Sprint 1 : l'onglet « Données »

Votre code de l'onglet « Données » (de `c1, c2, c3 = ...` jusqu'au nuage de points) :

```python
    c1, c2, c3 = st.columns(3)
    c1.metric("Relevés", len(df))
    c2.metric("Taux de panne", f"{df.failure.mean():.1%}")
    c3.metric("Température moyenne", f"{df.temperature_c.mean():.1f} °C")

    var = st.selectbox("Variable", FEATURES)
    taux = df.groupby(pd.cut(df[var], bins=10))["failure"].mean()
    taux.index = taux.index.astype(str)  # tranches en texte, sinon st.bar_chart plante
    st.bar_chart(taux, x_label=var, y_label="Taux de panne", sort=False)

    points = df.assign(panne=df.failure.map({0: "non", 1:"oui"}))
    st.scatter_chart(points, x="temperature_c", y="vibration_mm_s", color="panne")
```

Remarques sur ce code :

- `len(df)` est passé tel quel à `.metric`, car le test attend exactement la chaîne `"2000"` (sur ses propres données de 2000 relevés).
- Les tranches renvoyées par `pd.cut` sont des objets `Interval`, que `st.bar_chart` ne sait pas afficher (`SchemaValidationError`). Je les convertis donc en texte. `sort=False` garde ensuite les tranches dans l'ordre croissant, au lieu d'un tri alphabétique où `(11.1, ...]` passerait avant `(2.2, ...]`.
- `failure` vaut 0 ou 1. Utilisé tel quel comme couleur, Streamlit afficherait un dégradé. La colonne texte `panne` (« oui » / « non ») donne deux couleurs nettes.

| Indicateur | Valeur affichée |
|---|---|
| Relevés | 5000 |
| Taux de panne | 13.6% |
| Température moyenne | 59.8 °C |

Choisissez `vibration_mm_s` dans la liste. Comment évolue le taux de panne quand la vibration augmente ?

Le taux de panne **augmente fortement et régulièrement** avec la vibration :

| Vibration (mm/s) | Taux de panne | Nombre de relevés |
|---|---|---|
| 0 à 2,2 | 2,0 % | 2195 |
| 2,2 à 4,5 | 9,8 % | 1798 |
| 4,5 à 6,7 | 32,3 % | 678 |
| 6,7 à 8,9 | 66,4 % | 241 |
| 8,9 à 11,1 | 83,6 % | 61 |
| au-delà de 11,1 | 100 % | 27 |

On passe de 2 % de pannes pour une machine qui vibre peu à plus de 80 % au-delà de 9 mm/s. Le saut le plus net se situe entre 4,5 et 9 mm/s. Au-delà de 11 mm/s, tous les relevés sont des pannes, mais il n'y en a que 27 : ces dernières barres reposent sur très peu de données. Les tranches entre 15,6 et 20 mm/s ne contiennent aucun relevé et n'apparaissent donc pas dans le graphique.

Ajoutez un `print("relance")` en haut de `ui/app.py`, puis déplacez un curseur. Que voyez-vous dans le terminal, et qu'est-ce que cela montre sur le fonctionnement de Streamlit ?

J'ai compté les lignes `relance` dans le terminal à chaque action :

| Action | `relance` affiché ? |
|---|---|
| Ouverture ou rechargement de la page | oui, une fois |
| Changement de variable dans la liste déroulante | oui, une fois |
| Saisie de la clé API puis Entrée | oui, une fois |
| Clic sur « Prédire » | oui, une fois |
| Déplacement d'un curseur de l'onglet « Prédiction » | **non** |
| Passage d'un onglet à l'autre | non |

Cela montre que Streamlit **réexécute tout le script, de la première à la dernière ligne, à chaque interaction** avec un widget. Il n'y a pas de fonctions appelées sur un événement comme en JavaScript : le script entier est relancé avec les nouvelles valeurs des widgets.

Les curseurs de l'onglet « Prédiction » sont placés dans un `st.form`. Un formulaire regroupe les saisies et ne relance le script qu'au clic sur le bouton d'envoi (« Prédire »). Déplacer un curseur n'affiche donc rien : la nouvelle valeur n'est envoyée qu'au moment de « Prédire ». Un curseur placé hors formulaire relancerait le script à chaque déplacement.

À quoi sert `@st.cache_data` sur la fonction `charger` ?

Comme tout le script est relancé à chaque interaction, `pd.read_csv` relirait le fichier de 5000 lignes à chaque clic. `@st.cache_data` garde en mémoire le résultat de `charger` pour un argument donné, ici le chemin du fichier :

- au premier appel, le CSV est lu et le DataFrame est mis en cache ;
- aux relances suivantes, avec le même chemin, Streamlit renvoie directement une copie du DataFrame en cache, sans relire le disque ;
- si le chemin change, le fichier est relu.

L'application reste donc rapide malgré les relances. Il y a une limite : si le fichier `sensors.csv` est modifié sur le disque sans que son chemin change, le cache renvoie les anciennes données. Il faut alors vider le cache (menu ⋮ puis « Clear cache ») ou ajouter un `ttl`.

## 4. Sprint 2 : l'onglet « Prédiction »

Votre méthode `predict` (`maintai/client.py`) :

```python
    def predict(self, reading: dict) -> dict:
        """Envoie un relevé ; lève APIError si l'API refuse, httpx.HTTPError si elle est injoignable."""
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        r = self.http.post("/v1/predict", json=reading, headers=headers)
        if r.status_code != 200:
            raise APIError(r.status_code, r.json().get("detail"))
        return r.json()
```

Ce que fait chaque ligne :

1. L'en-tête `X-API-Key` n'est ajouté que si une clé est fournie.
2. Le relevé est envoyé en JSON à `/v1/predict`.
3. Toute réponse autre que 200 (401 mauvaise clé, 422 relevé invalide, 503 modèle absent…) lève une `APIError` qui garde le code et le message `detail` de l'API.
4. Sinon, la réponse est renvoyée : `failure_probability`, `alert`, `threshold`, `model_version`.

Si l'API est injoignable, `self.http.post` lève lui-même une `httpx.HTTPError`.

Le code correspondant dans `ui/app.py` :

```python
    if envoye:
        try:
            resultat = MaintaiClient(url, api_key=cle).predict(releve)
        except APIError as e:
            st.warning(f"L'API refuse la requête (code {e.status_code}) : {e.detail}")
        except httpx.HTTPError:
            st.error(f"API injoignable à l'adresse {url} : uvicorn est-il lancé ?")
        else:
            pct = round(resultat["failure_probability"] * 100)
            st.metric("Probabilité de panne", f"{pct} %")
            if resultat["alert"]:
                st.error(f"ALERTE : risque de panne (seuil {resultat['threshold']:.0%})")
            else:
                st.success("Fonctionnement normal")
```

Dernière ligne de `python -m pytest` à la fin du sprint 2 :

```
23 passed, 1 warning in 10.75s
```

Avec l'API lancée (`uvicorn maintai.api.main:app --reload --env-file .env`) :

| Essai | Ce qu'affiche Streamlit |
|---|---|
| valeurs par défaut, bonne clé | Indicateur « Probabilité de panne : **95 %** », puis un encadré rouge « ALERTE : risque de panne (seuil 30%) » |
| valeurs par défaut, fausse clé | Encadré jaune « L'API refuse la requête (code 401) : Clé API absente ou invalide » |
| API arrêtée (Ctrl+C dans le terminal d'uvicorn) | Encadré rouge « API injoignable à l'adresse http://localhost:8000 : uvicorn est-il lancé ? » |

Les valeurs par défaut (72 °C, 6 mm/s, 85 %, 15 000 h) donnent une probabilité de 95 %, au-dessus du seuil de 30 %, d'où l'alerte. Avec un relevé peu risqué (45 °C, 1 mm/s, 30 % de charge), l'application affiche « 0 % » puis « Fonctionnement normal » en vert. Dans les trois cas, l'application ne plante pas : chaque erreur est attrapée et affichée proprement.

J'ai rencontré deux pièges pendant ces essais. Après avoir changé la clé dans `.env`, il faut **redémarrer uvicorn**, car `--reload` surveille les fichiers `.py`, pas `.env`. Et si la clé est écrite entre guillemets dans `.env`, l'API retire les guillemets : il faut taper la clé sans eux dans Streamlit.

Le port 8501 n'est pas dans `MAINTAI_CORS_ORIGINS`, et pourtant la prédiction fonctionne. Pourquoi ?

Le CORS est une protection **appliquée par le navigateur** : quand du JavaScript d'une page (origine A) appelle une API d'une autre origine (B), le navigateur vérifie que B autorise A (en-tête `Access-Control-Allow-Origin`). Sinon, il bloque la réponse.

Avec Streamlit, le navigateur n'appelle jamais l'API. Il ne communique qu'avec le serveur Streamlit, sur le port 8501. C'est le **processus Python de Streamlit**, côté serveur, qui envoie la requête à l'API avec `httpx`. Une requête de serveur à serveur ne passe par aucun navigateur : personne ne vérifie le CORS, et `httpx` n'envoie même pas d'en-tête `Origin`. `MAINTAI_CORS_ORIGINS` n'intervient donc pas. Seule la clé API protège l'accès.

## 5. Questions de compréhension

1. **Dans la page web du TP2, qui envoie la requête à l'API ? Et dans Streamlit ?**
   Dans le TP2, c'est le **navigateur de l'utilisateur** : le JavaScript de `web/index.html` appelle l'API avec `fetch`. La requête part du poste de l'utilisateur, d'où le CORS, et la clé API se trouve dans le navigateur.
   Dans Streamlit, c'est le **serveur Streamlit**, en Python : `MaintaiClient`, avec `httpx`, envoie la requête depuis la machine où tourne Streamlit. Le navigateur se contente d'afficher la page et d'envoyer les clics au serveur Streamlit.

2. **Pourquoi les tests de `test_client.py` n'ont-ils pas besoin d'uvicorn ?**
   La fixture `client` de `conftest.py` crée un `TestClient(app)` de FastAPI. Il appelle l'application ASGI **directement en mémoire**, dans le même processus, sans réseau ni port ouvert. `MaintaiClient` accepte un paramètre `http` : les tests y passent ce `TestClient` à la place d'un `httpx.Client`. Les deux ont la même interface (`get`, `post`, `.status_code`, `.json()`), donc le code de `predict` est exactement le même qu'en production, mais sans serveur. C'est de l'injection de dépendance : les tests sont rapides et ne dépendent pas d'un uvicorn lancé à côté.

3. **Page HTML + JavaScript (TP2) ou Streamlit (TP3) : laquelle choisiriez-vous pour une démonstration à un client ? Et pour l'application d'un technicien de maintenance ? Pourquoi ?**
   - **Pour une démonstration à un client : Streamlit.** En une cinquantaine de lignes de Python, on obtient des indicateurs, des graphiques interactifs et un formulaire de prédiction, sans écrire de HTML, CSS ni JavaScript. C'est idéal pour montrer rapidement les données et le modèle, et pour modifier l'interface pendant ou entre deux réunions.
   - **Pour l'application d'un technicien : la page HTML + JavaScript**, ou une vraie application front.
     - Elle est légère et rapide : pas de relance de tout le script à chaque clic, et pas de serveur Python à maintenir pour chaque utilisateur connecté.
     - On contrôle entièrement l'ergonomie, pour l'adapter à une tablette ou un téléphone sur le terrain.
     - Elle peut être servie comme de simples fichiers statiques.
     - Il faudrait toutefois remplacer la clé API unique, visible dans le navigateur, par une vraie authentification de chaque technicien.

## 6. Le rendu Git

- Dépôt GitHub : https://github.com/Yasser-Hadli/Applications-Intelligentes, branche `TP`
- Lien de la pull request : pas de pull request. Le groupe publie directement sur la branche `TP`. Mon rendu est le commit [`c9bb3ff` — TP3 - Ajout du rendu de Mohamed Benhayoun Sadafi](https://github.com/Yasser-Hadli/Applications-Intelligentes/commit/c9bb3ff35e90acf0ef74341953565f0517676f6c). Il contient uniquement les trois fichiers du dossier `TP/TP 3/BENHAYOUN_SADAFI_Mohamed/` : `app.py`, `client.py` et `COMPTE_RENDU_BENHAYOUN_SADAFI_Mohamed.md`.
- Le tag `tp3` existe sur GitHub : non (`git ls-remote --tags origin` ne renvoie aucun tag)
- `git status` avant le commit : `.env`, `data/` ou `models/` apparaissaient-ils ? non. Le `.gitignore` du dépôt exclut `.env`, `data/`, `models/`, `*.csv`, `*.joblib` et `venv/`, et je n'ai ajouté (`git add`) que mon dossier de rendu.

## 7. Ce qui reste flou

- **Les variables d'environnement et le rechargement :** `uvicorn --reload` ne relit pas `.env`. J'ai dû redémarrer l'API après avoir changé la clé. Je ne sais pas encore comment on gère proprement les secrets en production (variables d'environnement du serveur, coffre à secrets, `st.secrets` côté Streamlit…).
- **Le Python utilisé :** un Streamlit lancé dans un terminal où le venv n'était pas activé utilisait le Python de Windows (Streamlit 1.60) au lieu de celui du projet (1.66.1), sans aucun message d'erreur. Je voudrais savoir comment éviter ce genre de mélange de façon fiable.
- **Streamlit avec plusieurs utilisateurs :** la clé API est saisie dans la barre latérale et tout le script est relancé à chaque clic. Je ne vois pas encore bien comment Streamlit se comporte avec de nombreux utilisateurs simultanés, ni comment donner à chacun ses propres droits d'accès.
