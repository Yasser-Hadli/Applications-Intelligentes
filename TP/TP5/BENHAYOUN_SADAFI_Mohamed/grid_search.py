"""Grid search : entraîne le gradient boosting pour chaque combinaison (learning rate, seuil).

3 learning rates x 10 seuils = 30 runs MLflow (maximum autorisé : 30).
Chaque run porte le tag grid=tp5 : dans mlflow ui, filtrer avec  tags.grid = "tp5"

Usage : python -m maintai.grid_search
"""
import itertools
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow

from maintai.train import main

LEARNING_RATES = [0.05, 0.1, 0.2]
THRESHOLDS = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35]
MIN_PRECISION = 0.5  # critère : au moins une alerte sur deux est une vraie panne


def run_grid() -> list[dict]:
    """Lance les 30 entraînements et renvoie, pour chacun, ses hyperparamètres et ses métriques."""
    grid = list(itertools.product(LEARNING_RATES, THRESHOLDS))
    assert len(grid) <= 30, "la grille ne doit pas dépasser 30 runs"
    results = []
    for i, (lr, threshold) in enumerate(grid, start=1):
        print(f"[{i}/{len(grid)}] learning_rate={lr}  threshold={threshold}")
        metrics = main(["--model", "gboost", "--lr", str(lr), "--threshold", str(threshold)])
        run_id = mlflow.last_active_run().info.run_id
        mlflow.MlflowClient().set_tag(run_id, "grid", "tp5")
        results.append({"run_id": run_id, "learning_rate": lr, **metrics["model"]})
    return results


def best_first(results: list[dict]) -> list[dict]:
    """Classe les runs : d'abord ceux qui respectent la precision minimale, puis par recall, puis par precision."""
    return sorted(
        results,
        key=lambda r: (r["precision"] >= MIN_PRECISION, r["recall"], r["precision"]),
        reverse=True,
    )


def to_markdown(results: list[dict]) -> str:
    lignes = [
        "| # | Run | learning_rate | threshold | recall | precision | accuracy | precision ≥ 0.5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, start=1):
        ok = "oui" if r["precision"] >= MIN_PRECISION else "non"
        lignes.append(
            f"| {i} | `{r['run_id'][:8]}` | {r['learning_rate']} | {r['threshold']} "
            f"| {r['recall']:.3f} | {r['precision']:.3f} | {r['accuracy']:.3f} | {ok} |"
        )
    return "\n".join(lignes)


if __name__ == "__main__":
    classement = best_first(run_grid())
    print(to_markdown(classement))
    print(f"Meilleur run : {classement[0]['run_id']}")
