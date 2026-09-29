"""
REST API for programmatic use: /api/v1/reconstruct, /api/v1/metrics, /api/v1/benchmark.

The request handlers are plain functions (dict in -> dict out) and are tested without any web
framework. `create_app()` wraps them in FastAPI when it is installed:

    pip install fastapi uvicorn
    python -m chronocell.api --port 8000          # docs at http://127.0.0.1:8000/docs

Every request is written to a JSON-lines run log (`AuditLog`). This is a reproducibility record:
- what was run, when, with which parameters and code version;
- a SHA-256 of the exact input, so a result can be matched to its data;
- how long it took, and whether it failed.
Raw input data are not stored. It is a research log, not a regulatory, clinical or certified audit
trail, and it makes no compliance claim.

Accuracy in responses follows chronocell.accuracy: the contact-map fit (measured on the request's
own input) and the microscopy benchmark (a property of the method) are reported separately.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from . import accuracy as ACC
from . import physics

API_VERSION = "v1"
SOFTWARE = "ChronoCell-5D 3.3"
MAX_POPULATION_BEADS = 400
MAX_SINGLE_BEADS = 2000
DEFAULT_LOG = Path(__file__).resolve().parent.parent / ".chronocell_cache" / "api_run_log.jsonl"


class RequestError(ValueError):
    """Invalid request (maps to HTTP 400)."""


# ----------------------------------------------------------------------------------------
# Run log
# ----------------------------------------------------------------------------------------
def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


class AuditLog:
    """Append-only JSON-lines log of API runs (thread-safe). One line per request."""

    def __init__(self, path: Path | str | None = DEFAULT_LOG):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read(self) -> list[dict]:
        if self.path is None or not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _run(endpoint: str, payload: dict, handler, log: AuditLog | None) -> dict:
    if not isinstance(payload, dict):
        raise RequestError("The request body must be a JSON object.")
    run_id = uuid.uuid4().hex
    t0 = time.time()
    record = {"run_id": run_id, "endpoint": f"/api/{API_VERSION}/{endpoint}", "software": SOFTWARE,
              "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "input_sha256": _digest(payload), "parameters": _parameters(payload)}
    try:
        try:
            out = handler(payload)
        except ValueError as exc:                    # model-level input problems are the caller's to fix (400)
            if isinstance(exc, RequestError):
                raise
            raise RequestError(str(exc)) from exc
        record.update(status="ok", seconds=round(time.time() - t0, 3), result_summary=out.get("summary", {}))
        out.update(run_id=run_id, software=SOFTWARE, input_sha256=record["input_sha256"])
        return out
    except RequestError as exc:
        record.update(status="rejected", seconds=round(time.time() - t0, 3), error=str(exc))
        raise
    except Exception as exc:
        record.update(status="error", seconds=round(time.time() - t0, 3), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if log is not None:
            log.write(record)


def _parameters(payload: dict) -> dict:
    """Request parameters without the raw data (sizes only for arrays)."""
    out = {}
    for k, v in (payload or {}).items():
        if isinstance(v, (list, tuple)):
            out[k] = f"<array len={len(v)}>"
        elif isinstance(v, dict):
            out[k] = {kk: (f"<array len={len(vv)}>" if isinstance(vv, (list, tuple)) else vv) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------------------
# Input parsing
# ----------------------------------------------------------------------------------------
def _contacts(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    c = payload.get("contacts")
    if not isinstance(c, dict) or not all(k in c for k in ("i", "j", "count")):
        raise RequestError("'contacts' must be an object with arrays 'i', 'j' and 'count'.")
    try:
        ci = np.asarray(c["i"], dtype=np.int64)
        cj = np.asarray(c["j"], dtype=np.int64)
        cm = np.asarray(c["count"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"contacts are not numeric: {exc}") from exc
    if not (ci.ndim == cj.ndim == cm.ndim == 1 and len(ci) == len(cj) == len(cm)):
        raise RequestError("contacts.i, contacts.j and contacts.count must be 1-D arrays of equal length.")
    n = int(payload.get("n_beads") or (max(ci.max(initial=-1), cj.max(initial=-1)) + 1))
    if n < 4:
        raise RequestError("At least 4 beads are needed.")
    if len(ci) and (ci.min() < 0 or cj.min() < 0 or ci.max() >= n or cj.max() >= n):
        raise RequestError(f"contact indices must lie in [0, {n - 1}].")
    if np.any(~np.isfinite(cm)) or np.any(cm < 0):
        raise RequestError("contact counts must be finite and non-negative.")
    keep = (ci != cj) & (cm > 0)
    a, b = np.minimum(ci[keep], cj[keep]), np.maximum(ci[keep], cj[keep])
    return a, b, cm[keep], n


def _coords(payload: dict) -> np.ndarray:
    try:
        x = np.asarray(payload.get("coords_nm"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"coords_nm is not numeric: {exc}") from exc
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 4 or not np.isfinite(x).all():
        raise RequestError("coords_nm must be an N x 3 array of finite numbers with N >= 4.")
    return x


# ----------------------------------------------------------------------------------------
# Handlers (dict in -> dict out)
# ----------------------------------------------------------------------------------------
def _structure_metrics(x: np.ndarray, b0: float) -> dict:
    fit = physics.distance_scaling(x)
    ster = physics.loss_steric(x, b0)
    return {"n_beads": int(len(x)), "rg_nm": physics.radius_of_gyration(x), "end_to_end_nm": physics.end_to_end(x),
            "max_span_nm": physics.max_span(x), "nu": None if not np.isfinite(fit.nu) else fit.nu,
            "nu_se": None if not np.isfinite(fit.nu_se) else fit.nu_se, "regime": fit.regime,
            "overlaps": ster.overlaps, "mean_bond_nm": float(np.mean(physics.bond_lengths(x)))}


def handle_metrics(payload: dict) -> dict:
    """Polymer metrics of a structure; plus contact-map fit if contacts are given."""
    x = _coords(payload)
    b0 = float(payload.get("b0_nm") or physics.B0_NM)
    out = {"metrics": _structure_metrics(x, b0)}
    if payload.get("contacts") is not None:
        ci, cj, cm, n = _contacts(payload | {"n_beads": len(x)})
        out["accuracy"] = ACC.two_scores("input", ACC.contact_fit_structure(x, ci, cj, cm))
    out["summary"] = {"n_beads": len(x), "rg_nm": round(out["metrics"]["rg_nm"], 2)}
    return out


def handle_reconstruct(payload: dict) -> dict:
    """Rebuild 3D structure from contacts.

    model = "population" (default; v3.3 maximum-entropy ensemble, <= 400 beads) or "single" (v3.2 EGNN).
    Optional: b0_nm, p_adjacent (population), seed, include ("median_distance", "population").
    """
    model = str(payload.get("model", "population")).lower()
    if model not in ("population", "single"):
        raise RequestError("model must be 'population' or 'single'.")
    ci, cj, cm, n = _contacts(payload)
    b0 = float(payload.get("b0_nm") or physics.B0_NM)
    seed = int(payload.get("seed", 0))
    include = set(payload.get("include") or ())
    t0 = time.time()
    if model == "population":
        if n > MAX_POPULATION_BEADS:
            raise RequestError(f"population model: at most {MAX_POPULATION_BEADS} beads (got {n}).")
        from . import ensemble as ENS
        p_adj = float(payload.get("p_adjacent", 0.5))
        if not 0.05 <= p_adj <= 0.95:
            raise RequestError("p_adjacent must be between 0.05 and 0.95.")
        res = ENS.fit_from_counts(ci, cj, cm, n, b0_nm=b0, p_adjacent=p_adj, cfg=ENS.EnsembleConfig(seed=seed))
        coords = res.representative_nm
        c_fit = ACC.spearman(res.contact_probability[ci, cj], cm)
        out = {"model": "population_v3_3", "coords_nm": coords.round(3).tolist(),
               "coords_note": "representative member of the population (closest to the median distance map)",
               "accuracy": ACC.two_scores("ensemble_v3_3", c_fit),
               "telemetry": {"fit_seconds": round(res.config["fit_seconds"], 3),
                             "sampling_seconds": round(res.config["sampling_seconds"], 3),
                             "device": res.config["device_used"], "best_misfit": res.history["best_loss"][0],
                             "trajectories": int(res.trajectories_nm.shape[1]), "frames": int(res.trajectories_nm.shape[0])},
               "assumptions": {"p_adjacent": p_adj, "length_anchor_b0_nm": b0, "r_c_nm": res.config["r_c_nm"]}}
        if "median_distance" in include:
            out["median_distance_nm"] = res.median_distance_nm.round(2).tolist()
        if "population" in include:
            out["population_nm"] = res.frames_nm.round(2).tolist()
    else:
        if n > MAX_SINGLE_BEADS:
            raise RequestError(f"single-structure model: at most {MAX_SINGLE_BEADS} beads (got {n}).")
        from . import egnn
        feats = egnn.node_features(np.full(n, 0.42), np.zeros(n), np.ones(n, bool))
        res = egnn.fit_structure(n, feats, ci, cj, cm, egnn.FitConfig(seed=seed), b0=b0)
        coords = res.coords_nm
        out = {"model": "single_structure_v3_2", "coords_nm": coords.round(3).tolist(),
               "accuracy": ACC.two_scores("single_structure_v3_2", ACC.contact_fit_structure(coords, ci, cj, cm)),
               "telemetry": {"seconds": round(res.seconds, 3), "device": res.config["device_used"],
                             "final_loss": {k: res.history[k][-1] for k in ("contact", "smooth", "steric", "total")}}}
    out["metrics"] = _structure_metrics(coords, b0)
    out["summary"] = {"model": out["model"], "n_beads": n, "contacts": int(len(ci)),
                      "contact_map_fit": out["accuracy"]["contact_map_fit"]["value"],
                      "seconds": round(time.time() - t0, 3)}
    return out


def handle_benchmark(payload: dict | None = None) -> dict:
    """The held-out microscopy benchmark (method-level), as stored in validation/results.json."""
    bench = ACC.load_benchmark()
    if bench is None:
        raise RequestError("No benchmark found: run `python validation/validate_tracing.py` first.")
    return {"benchmark": bench, "summary": {k: v["overall_percent_of_ceiling"] for k, v in bench["models"].items()}}


# ----------------------------------------------------------------------------------------
# Public entry points (with run logging) and the FastAPI wrapper
# ----------------------------------------------------------------------------------------
def reconstruct(payload: dict, log: AuditLog | None = None) -> dict:
    return _run("reconstruct", payload, handle_reconstruct, log)


def metrics(payload: dict, log: AuditLog | None = None) -> dict:
    return _run("metrics", payload, handle_metrics, log)


def benchmark(log: AuditLog | None = None) -> dict:
    return _run("benchmark", {}, handle_benchmark, log)


def create_app(log: AuditLog | None = None):
    """FastAPI application (requires `pip install fastapi uvicorn`)."""
    try:
        from fastapi import Body, FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("The REST API needs FastAPI: pip install fastapi uvicorn") from exc
    log = log if log is not None else AuditLog()
    app = FastAPI(title="ChronoCell-5D API", version="3.3",
                  description="Chromatin 3D reconstruction. Research use only; not a clinical tool.")

    def call(fn, *args):
        try:
            return fn(*args, log=log)
        except RequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(f"/api/{API_VERSION}/reconstruct")
    def _reconstruct(payload: dict = Body(...)) -> dict:
        return call(reconstruct, payload)

    @app.post(f"/api/{API_VERSION}/metrics")
    def _metrics(payload: dict = Body(...)) -> dict:
        return call(metrics, payload)

    @app.get(f"/api/{API_VERSION}/benchmark")
    def _benchmark() -> dict:
        return call(benchmark)

    return app


def main() -> None:  # pragma: no cover - starts a server
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(create_app(), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
