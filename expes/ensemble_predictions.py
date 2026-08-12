"""
Aggregate saved per-run test probabilities into seed / model ensembles.

Each train_enhanced.py run with --save-artifacts writes
    results/enhanced/logits/<ds>_<model>_<fraction>pct_t<trial>_<clf>[_aug][_ms].npz
containing softmax probs, labels, class names, and the run config.  This script
loads every artifact for a (dataset, fraction) and reports:

  - per-model single-trial mean accuracy (with TTA, if saved so)
  - per-model seed ensemble: softmax average over the trials of one model
  - cross-model ensemble: softmax average over all runs (all models x trials)
  - best single artifact accuracy (ceiling of oracle selection)

Usage:
    python ensemble_predictions.py [dataset [fraction_pct]] [--no-tta] \
        [--tta-only] [--models MODEL ...] [--tag TAG] [--save-csv]
"""
import os, sys, json, glob, argparse
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "results", "enhanced", "logits")
DATASETS = ["CBF", "ECG200", "ECG5000", "GunPoint", "Plane", "PowerCons",
            "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "TwoLeadECG", "UMD"]

parser = argparse.ArgumentParser()
parser.add_argument("dataset", nargs="?", default=None)
parser.add_argument("fraction_pct", nargs="?", type=int, default=100)
parser.add_argument("--models", nargs="+", default=None,
                    help="Restrict ensemble to these model names")
parser.add_argument("--no-tta", action="store_true",
                    help="Only use artifacts with tta_views == 0")
parser.add_argument("--tta-only", action="store_true",
                    help="Only use artifacts with tta_views > 0")
parser.add_argument("--save-csv", action="store_true",
                    help="Also append results to ensemble_results.csv")
args = parser.parse_args()

datasets = [args.dataset] if args.dataset else DATASETS
fraction = args.fraction_pct


def _cfg(z):
    return json.loads(str(z["config"][0]))


def _acc(probs, labels):
    return float(np.mean(probs.argmax(-1) == labels))


rows = []
for ds in datasets:
    files = sorted(glob.glob(os.path.join(RESULTS, f"{ds}_*.npz")))
    if not files:
        print(f"# {ds} {fraction}%: no artifacts")
        continue
    runs = []
    seen = set()
    for f in files:
        z = np.load(f, allow_pickle=True)
        c = _cfg(z)
        if c["fraction_pct"] != fraction:
            continue
        if c.get("multi_scale", False):
            continue
        if args.models and c["model"] not in args.models:
            continue
        tta = c.get("tta_views", 0)
        if args.no_tta and tta > 0:
            continue
        if args.tta_only and tta == 0:
            continue
        # Dedupe: mlp/xgboost-tagged runs of the same trial train identically.
        dkey = (c["dataset"], c["model"], c["fraction_pct"], c["trial"],
                tta, c.get("seed"), c.get("head_epochs", 0))
        if dkey in seen:
            continue
        seen.add(dkey)
        xp = z["xgb_probs"] if "xgb_probs" in z else None
        if xp is not None and xp.size == 0:
            xp = None
        runs.append((f, c, z["probs"], z["labels"], z["class_names"], xp))
    if not runs:
        print(f"# {ds} {fraction}%: no matching artifacts")
        continue

    # Sanity: labels/classes must agree across runs (same test set).
    ref_labels = runs[0][3]
    ref_classes = runs[0][4]
    for f, c, p, lab, cl, xp in runs[1:]:
        if not np.array_equal(lab, ref_labels):
            print(f"  !! label mismatch in {f}; skipping run")
        if not np.array_equal(cl, ref_classes):
            print(f"  !! class mismatch in {f}; skipping run")
    good = [r for r in runs if np.array_equal(r[3], ref_labels)]
    runs = good
    labels = ref_labels

    def _model_key(c):
        return (c["model"], c.get("tta_views", 0))

    by_model = {}
    for f, c, p, lab, cl, xp in runs:
        by_model.setdefault(_model_key(c), []).append((p, xp))

    print(f"# {ds} {fraction}%  ({len(runs)} runs)")
    single = []
    for (mname, tta), items in sorted(by_model.items()):
        ps = [it[0] for it in items]
        xps = [it[1] for it in items if it[1] is not None and it[1].size]
        accs = [_acc(p, labels) for p in ps]
        seed = float(np.mean(np.mean(ps, axis=0).argmax(-1) == labels))
        tag = f"tta{tta}" if tta else "notta"
        line = f"  {mname:38s} {tag:5s}  single-mean={np.mean(accs):.4f} " \
               f"seed-ensemble={seed:.4f}  (n={len(ps)})"
        rows.append({"dataset": ds, "fraction": fraction, "kind": "seed",
                     "model": mname, "tag": tag, "acc": seed, "n": len(ps)})
        if len(xps) == len(ps):
            xseed = float(np.mean(np.mean(xps, axis=0).argmax(-1) == labels))
            # Blend: average MLP and XGB seed probabilities (equal weight).
            blend = float(np.mean(
                ((np.mean(ps, axis=0) + np.mean(xps, axis=0)) / 2).argmax(-1) == labels))
            line += f"  xgb-seed={xseed:.4f}  blend={blend:.4f}"
            rows.append({"dataset": ds, "fraction": fraction, "kind": "xgb-seed",
                         "model": mname, "tag": tag, "acc": xseed, "n": len(xps)})
            rows.append({"dataset": ds, "fraction": fraction, "kind": "blend",
                         "model": mname, "tag": tag, "acc": blend, "n": len(ps) + len(xps)})
        print(line)

    # Cross-model ensemble: average every run's softmax together.
    all_ps = [p for _, _, p, _, _, _ in runs]
    ens = float(np.mean(np.mean(all_ps, axis=0).argmax(-1) == labels))
    print(f"  {'ALL-MODELS':38s}        cross-ensemble={ens:.4f}  (n={len(all_ps)})")
    rows.append({"dataset": ds, "fraction": fraction, "kind": "cross",
                 "model": "ALL", "tag": "all", "acc": ens, "n": len(all_ps)})

    # Best single artifact (oracle upper bound).
    best_acc, best_f = 0.0, None
    for f, c, p, lab, cl, xp in runs:
        a = _acc(p, labels)
        if a > best_acc:
            best_acc, best_f = a, f
    print(f"  best single = {best_acc:.4f}  ({os.path.basename(best_f)})")

if args.save_csv:
    import csv
    csv_path = os.path.join(os.path.dirname(RESULTS), "ensemble_results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Appended {len(rows)} rows to {csv_path}")
