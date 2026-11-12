# Compte rendu · TP3 : une interface Streamlit pour MaintAI

**Groupe :** Yasser Hadli - Adam Biloul - Mohammed Be · **Membre :** Yasser Hadli · **Date :** 12/11/2026

## 1. Mise en place

J'ai utilisé l'environnement virtuel `.venv` du projet pour lancer l'application et les tests avec les mêmes dépendances.

Je n'ai pas conservé la sortie du premier lancement de `python -m pytest`. Je ne peux donc pas donner sa dernière ligne exacte. Avant de compléter les TODO, les tests concernés étaient ceux de `tests/test_client.py` et de `tests/test_ui.py` :

- Le client levait `NotImplementedError("TP3 sprint 2 : MaintaiClient.predict")`.
- L'interface ne contenait pas encore les indicateurs et le résultat de prédiction attendus par les tests.

Le résultat final des tests est indiqué dans la partie 4.

## 2. Lancer Streamlit

J'ai rencontré un problème avec la commande `streamlit run ui/app.py` lorsque l'environnement virtuel n'était pas activé. Le problème se situe dans le terminal, avant l'ouverture de l'application dans le navigateur. Je n'ai pas conservé le message exact.

PowerShell cherche la commande `streamlit` dans les dossiers du PATH. Sans activation de `.venv`, il ne trouve pas forcément la version installée dans mon projet.

J'ai donc utilisé cette commande depuis le dossier `maintai-starter-tp3` :

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Avec `-m streamlit`, Python exécute le module Streamlit de son propre environnement. `python -m streamlit` fonctionne à condition que `python` désigne bien un interpréteur dans lequel Streamlit est installé. Le chemin explicite vers `.venv` permet de choisir le bon interpréteur.

## 3. Sprint 1 : l'onglet « Données »

J'ai ajouté trois indicateurs, une liste pour choisir la variable et deux graphiques. J'ai aussi remplacé les intervalles affichés sur l'axe X du graphique en barres par la moyenne des valeurs réellement présentes dans chaque tranche.

```python
nombre, taux, temperature = st.columns(3)
nombre.metric("Nombre de relevés", len(df))
taux.metric("Taux de panne", f"{df.failure.mean():.1%}")
temperature.metric("Température moyenne", f"{df.temperature_c.mean():.1f} °C")

variable = st.selectbox("Variable à explorer", FEATURES)
tranches = pd.cut(df[variable], bins=10)
taux_par_tranche = df.groupby(tranches, observed=True).agg(
    moyenne=(variable, "mean"),
    taux_de_panne=("failure", "mean"),
)
st.subheader("Taux de panne par tranche")
st.bar_chart(
    taux_par_tranche.reset_index(drop=True),
    x="moyenne",
    y="taux_de_panne",
    x_label=f"Moyenne de {variable} par tranche",
    y_label="Taux de panne",
)

st.subheader("Température et vibration")
nuage = df.assign(panne=df.failure.map({0: "Normal", 1: "Panne"}))
st.scatter_chart(nuage, x="temperature_c", y="vibration_mm_s", color="panne")
```

Voici les valeurs obtenues avec le fichier `data/sensors.csv` du projet :

| Indicateur | Valeur affichée |
|---|---|
| Relevés | 5000 |
| Taux de panne | 13.6% |
| Température moyenne | 59.8 °C |

Avec `vibration_mm_s`, je constate que le taux de panne augmente avec la vibration. Il passe d'environ 2 % dans la première tranche à 32 % entre 4,462 et 6,683 mm/s, puis à 84 % entre 8,904 et 11,125 mm/s. Les tranches les plus élevées atteignent 100 %, mais cette valeur doit être interprétée en tenant compte du nombre de relevés dans chaque tranche.

Concernant `print("relance")`, le comportement attendu est une nouvelle écriture dans le terminal à chaque réexécution du script. Je n'ai pas conservé de sortie de cette manipulation. Dans mon application, les curseurs sont placés dans `st.form` : les modifications sont envoyées ensemble quand je clique sur « Prédire ». Déplacer un curseur du formulaire ne relance donc pas immédiatement le script. En revanche, changer la variable dans la liste déroulante, située hors du formulaire, provoque une réexécution.

`@st.cache_data` mémorise le résultat de `charger`. Cela évite de relire le CSV à chaque interaction si les arguments et le code de la fonction restent identiques. Une modification du contenu du fichier au même chemin ne suffit pas forcément à invalider le cache ; dans ce cas, il faut le vider ou prévoir une expiration.

## 4. Sprint 2 : l'onglet « Prédiction »

J'ai complété le client pour envoyer le relevé en JSON à `/v1/predict`. La clé est transmise dans l'en-tête `X-API-Key` lorsqu'elle est renseignée. Si l'API refuse la requête, je récupère le code HTTP et le message dans une `APIError`. Les erreurs de connexion remontent à l'interface sous forme de `httpx.HTTPError`.

```python
def predict(self, reading: dict) -> dict:
    """Envoie un relevé ; lève APIError si l'API refuse, httpx.HTTPError si elle est injoignable."""
    headers = {"X-API-Key": self.api_key} if self.api_key else {}
    response = self.http.post("/v1/predict", json=reading, headers=headers)
    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        raise APIError(response.status_code, detail)
    return response.json()
```

Dans `ui/app.py`, j'affiche un avertissement pour une erreur renvoyée par l'API, un message d'erreur si elle est injoignable et, en cas de succès, le pourcentage de panne accompagné d'une alerte ou d'un message normal. Le client HTTP est fermé après l'appel grâce au bloc `with`.

Après les modifications, la suite de tests a donné :

```text
23 passed, 1 warning in 12.09s
```

Cette sortie provient de la validation effectuée à la fin de l'implémentation. L'avertissement concerne la dépréciation de l'utilisation de `httpx` avec le `TestClient` de Starlette.

Les résultats ci-dessous correspondent au modèle local et aux messages prévus dans l'interface. Le pourcentage a été vérifié directement avec le modèle ; ce tableau ne constitue pas une capture de trois essais manuels dans le navigateur.

| Essai | Ce qu'affiche Streamlit |
|---|---|
| Valeurs par défaut, bonne clé | Probabilité de panne : **95 %** ; « ALERTE : risque de panne détecté. » |
| Valeurs par défaut, fausse clé | « Erreur 401 : Clé API absente ou invalide » |
| API arrêtée | « API injoignable : … », suivi du détail de l'erreur de connexion |

Avec les valeurs par défaut, le modèle local renvoie une probabilité de `0.9475`, arrondie à 95 % à l'affichage. Le seuil de décision est de `0.3`, donc l'alerte est active.

Le port 8501 n'a pas besoin de figurer dans `MAINTAI_CORS_ORIGINS` pour cette requête : c'est le serveur Python de Streamlit qui contacte l'API avec `httpx`. CORS est une protection appliquée par le navigateur aux requêtes entre origines différentes.

## 5. Questions de compréhension

1. **Qui envoie la requête à l'API ?**

   Dans la page du TP2, le JavaScript exécuté dans le navigateur envoie la requête avec `fetch`. Dans Streamlit, c'est le code Python exécuté sur le serveur qui utilise `MaintaiClient` et `httpx`.

2. **Pourquoi les tests de `test_client.py` n'ont-ils pas besoin d'uvicorn ?**

   Les tests injectent le `TestClient` de FastAPI dans `MaintaiClient` avec le paramètre `http`. Les requêtes sont traitées directement par l'application pendant le test, sans serveur réseau à lancer. Les fixtures préparent aussi les données, le modèle et la configuration nécessaires.

3. **Quelle interface choisir ?**

   Pour une démonstration à un client, je choisirais Streamlit : je peux rapidement présenter les données, modifier les paramètres et montrer une prédiction. Pour une application utilisée quotidiennement par un technicien de maintenance, je choisirais une interface HTML et JavaScript si j'ai besoin d'une ergonomie plus personnalisée, adaptée aux écrans mobiles et aux tâches du technicien. L'API peut rester la même dans les deux cas.

## 6. Le rendu Git

- **Lien de la pull request :** non disponible à ce stade.
- **Tag `tp3` sur GitHub :** non vérifié.
- **État Git avant le commit :** non disponible. Le dossier actuel n'est pas reconnu comme un dépôt Git (`fatal: not a git repository`).

J'ai vérifié le fichier `.gitignore` : il contient `.env`, `data/*.csv` et `models/`. Ces règles excluent normalement ces fichiers s'ils ne sont pas déjà suivis par Git. Je ne peux pas confirmer leur absence d'un ancien `git status` sans disposer du dépôt et de cette sortie.

## 7. Ce qui reste flou

Je comprends mieux la différence entre le navigateur, le serveur Streamlit et l'API. Je souhaite encore approfondir la gestion du cache quand le fichier CSV change, ainsi que la mise en production de l'application. Je dois également finaliser le rendu Git et vérifier la création de la pull request et du tag `tp3`.
