#!/usr/bin/env python3
"""
Benchmark: Persistence Diagram & Landscape computation.

Compares our model's compute_persistence_diagram (gudhi.AlphaComplex) against
GUDHI Vietoris-Rips, Strong Witness, and our manual persistence landscape
implementation against gudhi.representations.Landscape.

Usage:
    python benchmark_pd_pl.py [--quick] [--output FILE]
"""

import argparse
import csv
import gc
import resource
import signal
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── ensure project root is importable ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


class _TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TimeoutError("timed out")


class Timeout:
    """Context manager that raises _TimeoutError after *seconds*."""
    def __init__(self, seconds):
        self.seconds = seconds
    def __enter__(self):
        self._prev = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(self.seconds))
        return self
    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._prev)

# ── gudhi imports ──────────────────────────────────────────────────────────
import gudhi as gd

_HAS_LANDSCAPE = False
try:
    from gudhi.representations import Landscape as GudhiLandscape
    _HAS_LANDSCAPE = True
except ImportError:
    pass

_HAS_HERA = False
try:
    from gudhi.hera import wasserstein_distance as wasserstein_distance_gudhi
    _HAS_HERA = True
except ImportError:
    pass

_HAS_STRONG_WITNESS = False
try:
    from gudhi import EuclideanStrongWitnessComplex
    _HAS_STRONG_WITNESS = True
except ImportError:
    pass

# ── our code ───────────────────────────────────────────────────────────────
from models import compute_persistence_diagram as our_compute_pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RIPS_MAX_SIZE = 300          # Rips is O(2^d); skip beyond this
WITNESS_MAX_SIZE = 300       # Strong witness is slow; cap low
METHOD_TIMEOUT = 60          # per-call timeout in seconds


# ---------------------------------------------------------------------------
# Memory measurement helper
# ---------------------------------------------------------------------------

def _get_rss_mb():
    """Peak resident set size in MB (macOS / POSIX)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Point-cloud generators
# ---------------------------------------------------------------------------

def make_circle_2d(n, seed=0):
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def make_sphere_3d(n, seed=0):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def make_torus_3d(n, R=2.0, r=0.6, seed=0):
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.column_stack([x, y, z])


SHAPE_REGISTRY = {
    "circle_2d": (make_circle_2d, 2),
    "sphere_3d": (make_sphere_3d, 3),
    "torus_3d": (make_torus_3d, 3),
}


# ---------------------------------------------------------------------------
# PD computation wrappers — all return (K,2) float32 numpy arrays
# ---------------------------------------------------------------------------

def pd_alpha_our(pc):
    """Our model's compute_persistence_diagram (gudhi.AlphaComplex, H1)."""
    import torch
    result = our_compute_pd(pc, homology_dim=1)
    if isinstance(result, torch.Tensor):
        arr = result.numpy()
    else:
        arr = np.asarray(result)
    return arr.astype(np.float32) if arr.size > 0 else np.zeros((0, 2), np.float32)


def pd_alpha_raw(pc):
    """Raw gudhi AlphaComplex H1 (pure numpy, no torch)."""
    pc = np.asarray(pc, dtype=np.float64)
    if pc.shape[0] < 3:
        return np.zeros((0, 2), np.float32)
    ac = gd.AlphaComplex(points=pc).create_simplex_tree()
    ac.persistence()
    dgm = ac.persistence_intervals_in_dimension(1)
    if dgm is None or len(dgm) == 0:
        return np.zeros((0, 2), np.float32)
    return np.array(dgm, dtype=np.float32)


def pd_rips(pc):
    """GUDHI Vietoris-Rips H1 with adaptive edge-length cutoff."""
    pc = np.asarray(pc, dtype=np.float64)
    n = pc.shape[0]
    if n < 3:
        return np.zeros((0, 2), np.float32)
    from scipy.spatial.distance import pdist
    max_edge = float(np.percentile(pdist(pc), 50))
    rips = gd.RipsComplex(points=pc, max_edge_length=max_edge)
    st = rips.create_simplex_tree(max_dimension=2)
    st.persistence()
    dgm = st.persistence_intervals_in_dimension(1)
    if dgm is None or len(dgm) == 0:
        return np.zeros((0, 2), np.float32)
    return np.array(dgm, dtype=np.float32)


def pd_strong_witness(pc):
    """GUDHI EuclideanStrongWitnessComplex H1."""
    pc = np.asarray(pc, dtype=np.float64)
    n = pc.shape[0]
    if n < 3:
        return np.zeros((0, 2), np.float32)
    landmark_count = min(n, max(10, int(np.sqrt(n))))
    rng = np.random.RandomState(0)
    landmarks = pc[rng.choice(n, landmark_count, replace=False)]
    swc = EuclideanStrongWitnessComplex(landmarks=landmarks, witnesses=pc)
    st = swc.create_simplex_tree(max_alpha_square=10.0)
    st.persistence()
    dgm = st.persistence_intervals_in_dimension(1)
    if dgm is None or len(dgm) == 0:
        return np.zeros((0, 2), np.float32)
    return np.array(dgm, dtype=np.float32)


# ---------------------------------------------------------------------------
# Persistence landscape — manual numpy (our implementation)
# ---------------------------------------------------------------------------

def persistence_landscape_manual(dgm, resolution=100, num_landscapes=5):
    """
    Persistence landscape via the tent-function / folding approach.

    Returns (num_landscapes, resolution) array.
    """
    import torch as _torch
    if isinstance(dgm, _torch.Tensor):
        dgm = dgm.numpy()
    if dgm is None or len(dgm) == 0:
        return np.zeros((num_landscapes, resolution))

    dgm = np.asarray(dgm, dtype=np.float64)
    dgm = dgm[np.isfinite(dgm).all(axis=1)]
    if len(dgm) == 0:
        return np.zeros((num_landscapes, resolution))

    births, deaths = dgm[:, 0], dgm[:, 1]
    lo, hi = float(births.min()), float(deaths.max())
    if hi - lo < 1e-12:
        return np.zeros((num_landscapes, resolution))

    grid = np.linspace(lo, hi, resolution)
    midpoints = (births + deaths) * 0.5
    halfpers = (deaths - births) * 0.5

    # tent[i](t) = max(0, halfpers_i - |t - midpoint_i|)
    tents = np.maximum(0, halfpers[:, None] - np.abs(grid[None] - midpoints[:, None]))
    # sort features by tent height at each grid point (descending)
    sorted_tents = -np.sort(-tents, axis=0)

    landscapes = np.zeros((num_landscapes, resolution))
    n_feat = min(num_landscapes, sorted_tents.shape[0])
    landscapes[:n_feat] = sorted_tents[:n_feat]
    return landscapes


# ---------------------------------------------------------------------------
# Time-and-measure a single PD call
# ---------------------------------------------------------------------------

def _timed_pd_call(method_fn, pc):
    """Run method_fn(pc) and return (dgm_array, elapsed_s, rss_delta_mb).
    Returns (None, elapsed, mem) on timeout or error."""
    gc.collect()
    rss0 = _get_rss_mb()
    t0 = time.perf_counter()
    try:
        with Timeout(METHOD_TIMEOUT):
            dgm = method_fn(pc)
    except _TimeoutError:
        return None, time.perf_counter() - t0, max(_get_rss_mb() - rss0, 0)
    except Exception:
        return None, time.perf_counter() - t0, max(_get_rss_mb() - rss0, 0)
    elapsed = time.perf_counter() - t0
    rss1 = _get_rss_mb()
    return dgm, elapsed, max(rss1 - rss0, 0)


# ---------------------------------------------------------------------------
# Section 1 — PD method comparison
# ---------------------------------------------------------------------------

def benchmark_pd_methods(shapes, sizes):
    """Compare all PD methods on the largest size for each shape."""
    max_n = sizes[-1]
    methods = [("alpha_ours", pd_alpha_our), ("alpha_raw", pd_alpha_raw)]

    rips_ok = max_n <= RIPS_MAX_SIZE
    if rips_ok:
        methods.append(("rips", pd_rips))
    else:
        # still benchmark Rips at RIPS_MAX_SIZE for reference
        methods.append(("rips", pd_rips))

    if _HAS_STRONG_WITNESS and max_n <= WITNESS_MAX_SIZE:
        methods.append(("strong_witness", pd_strong_witness))

    results = []
    for shape_name in shapes:
        gen_fn, dim = SHAPE_REGISTRY[shape_name]
        for method_name, method_fn in methods:
            # choose size: Rips gets capped
            n = max_n
            if method_name == "rips" and max_n > RIPS_MAX_SIZE:
                n = RIPS_MAX_SIZE
            if method_name == "strong_witness" and max_n > WITNESS_MAX_SIZE:
                continue

            pc = gen_fn(n, seed=42)
            try:
                # warm-up
                method_fn(gen_fn(min(n, 10), seed=0))
            except Exception:
                pass

            try:
                dgm, elapsed, mem = _timed_pd_call(method_fn, pc)
                if dgm is None:
                    raise RuntimeError("timeout or error")
                finite = dgm[np.isfinite(dgm).all(axis=1)] if len(dgm) > 0 else dgm
                lifetimes = (finite[:, 1] - finite[:, 0]).tolist() if len(finite) > 0 else []
                results.append(dict(
                    shape=shape_name, dim=dim, n_points=n,
                    method=method_name, time_s=elapsed, mem_mb=mem,
                    n_features=len(dgm),
                    mean_lifetime=float(np.mean(lifetimes)) if lifetimes else 0,
                    max_lifetime=float(np.max(lifetimes)) if lifetimes else 0,
                ))
            except Exception as e:
                results.append(dict(
                    shape=shape_name, dim=dim, n_points=n,
                    method=method_name, time_s=-1, mem_mb=-1,
                    n_features=-1, mean_lifetime=-1, max_lifetime=-1,
                    error=str(e)[:120],
                ))
    return results


# ---------------------------------------------------------------------------
# Section 2 — PD scaling across sizes
# ---------------------------------------------------------------------------

def benchmark_pd_scaling(sizes):
    """Time PD computation across sizes (sphere_3d)."""
    gen_fn = SHAPE_REGISTRY["sphere_3d"][0]

    methods = [("alpha_ours", pd_alpha_our), ("alpha_raw", pd_alpha_raw)]
    # Rips only for small sizes
    methods_rips_only = [("rips", pd_rips)]

    results = []
    for n in tqdm(sizes, desc="PD scaling"):
        pc = gen_fn(n, seed=42)
        for method_name, method_fn in methods:
            dgm, elapsed, mem = _timed_pd_call(method_fn, pc)
            if dgm is not None:
                results.append(dict(
                    shape="sphere_3d", dim=3, n_points=n,
                    method=method_name, time_s=elapsed, mem_mb=mem,
                    n_features=len(dgm),
                ))
            else:
                results.append(dict(
                    shape="sphere_3d", dim=3, n_points=n,
                    method=method_name, time_s=-1, mem_mb=-1,
                    n_features=-1, error="timeout or error",
                ))

        # Rips only for sizes ≤ RIPS_MAX_SIZE
        if n <= RIPS_MAX_SIZE:
            for method_name, method_fn in methods_rips_only:
                dgm, elapsed, mem = _timed_pd_call(method_fn, pc)
                if dgm is not None:
                    results.append(dict(
                        shape="sphere_3d", dim=3, n_points=n,
                        method=method_name, time_s=elapsed, mem_mb=mem,
                        n_features=len(dgm),
                    ))
                else:
                    results.append(dict(
                        shape="sphere_3d", dim=3, n_points=n,
                        method=method_name, time_s=-1, mem_mb=-1,
                        n_features=-1, error="timeout or error",
                    ))
    return results


# ---------------------------------------------------------------------------
# Section 3 — Persistence landscape comparison
# ---------------------------------------------------------------------------

def benchmark_persistence_landscape(sizes):
    """Compare manual PL vs gudhi Landscape."""
    if not _HAS_LANDSCAPE:
        print("[WARN] gudhi.representations.Landscape not available, skipping.")
        return []

    gen_fn = SHAPE_REGISTRY["sphere_3d"][0]
    results = []

    for n in tqdm(sizes, desc="PL benchmark"):
        dgm = pd_alpha_raw(gen_fn(n, seed=42))

        # ── manual ──
        t0 = time.perf_counter()
        pl_manual = persistence_landscape_manual(dgm, resolution=100,
                                                  num_landscapes=5)
        t_manual = time.perf_counter() - t0

        # ── gudhi ──
        t0 = time.perf_counter()
        gl = GudhiLandscape(num_landscapes=5, resolution=100)
        if len(dgm) > 0:
            pl_gudhi = gl.fit_transform([dgm]).reshape(5, 100)
        else:
            pl_gudhi = np.zeros((5, 100))
        t_gudhi = time.perf_counter() - t0

        l1 = float(np.abs(pl_manual - pl_gudhi).sum())
        l2 = float(np.sqrt(((pl_manual - pl_gudhi) ** 2).sum()))
        nm, ng = np.linalg.norm(pl_manual), np.linalg.norm(pl_gudhi)
        cosine = float(1.0 - np.dot(pl_manual.ravel(), pl_gudhi.ravel()) / (nm * ng)) \
            if nm > 1e-12 and ng > 1e-12 else 0.0

        results.append(dict(
            shape="sphere_3d", n_points=n,
            time_manual_s=t_manual, time_gudhi_s=t_gudhi,
            l1_distance=l1, l2_distance=l2, cosine_distance=cosine,
            n_pd_features=len(dgm),
        ))
    return results


# ---------------------------------------------------------------------------
# Section 4 — PD stability under perturbation
# ---------------------------------------------------------------------------

def benchmark_stability(sizes, n_trials=5, noise_sigma=0.05):
    """Measure bottleneck/Wasserstein distances between clean & perturbed PDs."""
    gen_fn = SHAPE_REGISTRY["sphere_3d"][0]
    results = []

    for n in tqdm(sizes, desc="Stability"):
        for trial in range(n_trials):
            pc_clean = gen_fn(n, seed=1000 + trial)
            rng = np.random.RandomState(2000 + trial)
            pc_pert = pc_clean + rng.normal(0, noise_sigma, pc_clean.shape)

            dgm_clean = pd_alpha_raw(pc_clean)
            dgm_pert = pd_alpha_raw(pc_pert)

            bottleneck = w1 = w2 = -1.0
            if len(dgm_clean) > 0 and len(dgm_pert) > 0:
                try:
                    bottleneck = float(gd.bottleneck_distance(dgm_clean, dgm_pert))
                except Exception:
                    pass
                if _HAS_HERA:
                    try:
                        w1 = float(wasserstein_distance_gudhi(
                            dgm_clean, dgm_pert, order=1))
                    except Exception:
                        pass
                    try:
                        w2 = float(wasserstein_distance_gudhi(
                            dgm_clean, dgm_pert, order=2))
                    except Exception:
                        pass

            results.append(dict(
                shape="sphere_3d", n_points=n, trial=trial,
                noise_sigma=noise_sigma,
                bottleneck_distance=bottleneck,
                wasserstein_1=w1, wasserstein_2=w2,
                n_features_clean=len(dgm_clean),
                n_features_perturbed=len(dgm_pert),
            ))
    return results


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_table(rows, keys=None, title=""):
    if not rows:
        return
    if keys is None:
        keys = list(rows[0].keys())
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")

    widths = {}
    for k in keys:
        widths[k] = max(len(k),
                        max(len(str(r.get(k, ""))) for r in rows))

    header = "  ".join(f"{k:>{widths[k]}}" for k in keys)
    print(header)
    print("  ".join("-" * widths[k] for k in keys))
    for r in rows:
        vals = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float):
                if v < 0:
                    vals.append(f"{'ERR':>{widths[k]}}")
                else:
                    vals.append(f"{v:>{widths[k]}.4f}")
            else:
                vals.append(f"{str(v):>{widths[k]}}")
        print("  ".join(vals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PD & PL computation: our model vs GUDHI")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer shapes and sizes")
    parser.add_argument("--output", type=str,
                        default="benchmark_pd_pl_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    if args.quick:
        sizes = [100, 300]
        shapes = ["sphere_3d"]
    else:
        sizes = [100, 300, 600, 1000]
        shapes = ["circle_2d", "sphere_3d", "torus_3d"]

    all_results = []
    all_keys = set()

    # ── 1. PD method comparison ────────────────────────────────────────────
    print("\n[1/4] PD method comparison ...")
    pd_method_results = benchmark_pd_methods(shapes, sizes)
    all_results.extend(pd_method_results)
    if pd_method_results:
        all_keys.update(pd_method_results[0].keys())
    print_table(
        [{k: r[k] for k in ("shape", "dim", "n_points", "method",
                              "time_s", "mem_mb", "n_features",
                              "mean_lifetime", "max_lifetime")}
         for r in pd_method_results if r.get("time_s", -1) >= 0],
        title="PD Method Comparison",
    )

    # ── 2. PD scaling ──────────────────────────────────────────────────────
    print("\n[2/4] PD scaling ...")
    scaling_results = benchmark_pd_scaling(sizes)
    all_results.extend(scaling_results)
    if scaling_results:
        all_keys.update(scaling_results[0].keys())
    print_table(
        [{k: r[k] for k in ("shape", "n_points", "method",
                              "time_s", "mem_mb", "n_features")}
         for r in scaling_results if r.get("time_s", -1) >= 0],
        title="PD Scaling (sphere_3d)",
    )

    # ── 3. Persistence landscape comparison ─────────────────────────────────
    print("\n[3/4] Persistence landscape comparison ...")
    pl_results = benchmark_persistence_landscape(sizes)
    all_results.extend(pl_results)
    if pl_results:
        all_keys.update(pl_results[0].keys())
        print_table(
            [{k: r[k] for k in ("n_points", "time_manual_s", "time_gudhi_s",
                                  "l1_distance", "l2_distance",
                                  "cosine_distance", "n_pd_features")}
             for r in pl_results],
            title="PL: Manual vs Gudhi Landscape",
        )

    # ── 4. PD stability ────────────────────────────────────────────────────
    print("\n[4/4] PD stability ...")
    stability_results = benchmark_stability(sizes)
    all_results.extend(stability_results)
    if stability_results:
        all_keys.update(stability_results[0].keys())
        agg = defaultdict(lambda: {"bn": [], "w1": [], "w2": []})
        for r in stability_results:
            n = r["n_points"]
            if r["bottleneck_distance"] >= 0:
                agg[n]["bn"].append(r["bottleneck_distance"])
            if r["wasserstein_1"] >= 0:
                agg[n]["w1"].append(r["wasserstein_1"])
            if r["wasserstein_2"] >= 0:
                agg[n]["w2"].append(r["wasserstein_2"])
        summary = []
        for n in sorted(agg):
            a = agg[n]
            summary.append(dict(
                n_points=n,
                bottleneck_mean=np.mean(a["bn"]) if a["bn"] else -1,
                bottleneck_std=np.std(a["bn"]) if a["bn"] else -1,
                wasserstein_1_mean=np.mean(a["w1"]) if a["w1"] else -1,
                wasserstein_2_mean=np.mean(a["w2"]) if a["w2"] else -1,
                n_trials=len(a["bn"]),
            ))
        print_table(summary, title="PD Stability Summary")

    # ── write CSV ───────────────────────────────────────────────────────────
    if all_results:
        all_keys.add("error")
        for r in all_results:
            for k in all_keys:
                if k not in r:
                    r[k] = ""
        sorted_keys = sorted(all_keys)
        outpath = Path(args.output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted_keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {outpath.resolve()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
