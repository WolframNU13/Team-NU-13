"""
The two accuracy scores, kept separate everywhere (UI, PDF dossier, JSON report).

* Contact-map fit: how well a model reproduces the contact data it was BUILT FROM. It is measured
  live on the user's own window. It shows the optimisation converged. It is not evidence that
  the 3D model is right: any good optimiser scores high on its own input.
* Microscopy accuracy (benchmark): how well the METHOD predicts distances measured by imaging in
  cells it never saw (validation/, Bintu et al. 2018, held-out test datasets). It is a property of
  the method, read from validation/results.json. It is not measured on the user's data, which has
  no imaging ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "validation" / "results.json"

CONTACT_FIT_DEFINITION = ("Spearman rank correlation between the model's contacts (or closeness) and the input "
                          "contact map it was built from. Shows convergence; not evidence of accuracy.")
MICROSCOPY_DEFINITION = ("Share of the reproducible 3D folding pattern recovered, relative to how well the experiment "
                         "agrees with itself; trend-removed Spearman rho vs held-out chromatin-tracing distances "
                         "(Bintu et al., Science 2018). Overall = sum(model) / sum(ceiling) over 3 test datasets x 3 splits.")


def _rank(v: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(v)).astype(np.float64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def contact_fit_structure(coords: np.ndarray, ci: np.ndarray, cj: np.ndarray, cm: np.ndarray) -> float:
    """Contact-map fit of a single structure: Spearman(-distance, contact count) over observed pixels."""
    x = np.asarray(coords, dtype=np.float64)
    ci, cj = np.asarray(ci, np.int64), np.asarray(cj, np.int64)
    if ci.size < 3:
        return float("nan")
    d = np.linalg.norm(x[ci] - x[cj], axis=1)
    return spearman(-d, np.asarray(cm, np.float64))


def load_benchmark(path: Path | str = RESULTS) -> dict | None:
    """Summary of the held-out microscopy benchmark, or None if validation has not been run."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        s = data["summary"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    out = {"source": "validation/results.json", "definition": MICROSCOPY_DEFINITION, "models": {}}
    for model in ("single_structure_v3_2", "ensemble_v3_3"):
        if model not in s.get("overall", {}):
            continue
        out["models"][model] = {
            "overall_percent_of_ceiling": round(float(s["overall"][model]["percent_of_ceiling"]), 1),
            "per_dataset_percent_of_ceiling": {name: round(float(d[model]["percent_of_ceiling"]), 1)
                                               for name, d in s["datasets"].items() if model in d},
        }
    return out if out["models"] else None


def two_scores(model: str, contact_fit: float | None) -> dict:
    """The two labelled scores for one displayed model ("single_structure_v3_2" / "ensemble_v3_3" / "input")."""
    bench = load_benchmark()
    micro = None
    if bench and model in bench["models"]:
        m = bench["models"][model]
        micro = {"overall_percent_of_ceiling": m["overall_percent_of_ceiling"],
                 "per_dataset_percent_of_ceiling": m["per_dataset_percent_of_ceiling"],
                 "scope": "method-level benchmark on held-out imaging data; not measured on this structure",
                 "definition": MICROSCOPY_DEFINITION, "source": bench["source"]}
    fit = None if contact_fit is None or not np.isfinite(contact_fit) else round(float(contact_fit), 4)
    return {"contact_map_fit": {"value": fit, "scope": "measured on this window's input contacts",
                                "definition": CONTACT_FIT_DEFINITION},
            "microscopy_accuracy": micro}
