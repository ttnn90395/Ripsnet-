#!/usr/bin/env python3
"""
Consolidate shape classification results from cluster into a single CSV.

Downloads JSON result files from the Polytechnique cluster and merges them
into a unified results_all.csv with per-trial and mean±std statistics.

Usage:
    python consolidate_results.py [--download] [--local DIR]
    
    --download   rsync results from cluster (default)
    --local DIR  use a local directory of JSON files instead
"""

import json
import os
import sys
import glob
import subprocess
import csv
import argparse
from collections import defaultdict
import numpy as np

import os
CLUSTER = os.environ.get("RIPSNET_CLUSTER_USER", "ten.nguyen-hanaoka@dindon.polytechnique.fr")
REMOTE_DIR = os.environ.get("RIPSNET_REMOTE_DIR", "/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-/shape/results")
SSH_PASS = os.environ.get("RIPSNET_SSH_PASS", "")

DATASETS = [
    "circles",
    "circles_noisy",
    "shapes3d_topology",
    "shapes3d_geometry",
    "shapes3d_complex",
    "shapes3d_8way",
]

MODELS = [
    "PersNet",
    "RipsPointNet",
    "ScalarInputMLP",
    "ScalarDistanceDeepSet",
    "TensorFieldNetwork",
    "GTTensorFieldNetworkV2",
    "HierarchicalTensorFieldNetwork",
    "StochasticTensorFieldNetwork",
    "OnEquivariantTensorFieldNetwork",
    "AttentionTensorFieldNetwork",
    "RelaxedOnEquivariantTensorFieldNetwork",
    "HybridOnEquivariantTensorFieldNetwork",
]

MODEL_SHORT = {
    "PersNet": "PersNet",
    "RipsPointNet": "RipsPointNet",
    "ScalarInputMLP": "ScalarMLP",
    "ScalarDistanceDeepSet": "ScalarDeepSet",
    "TensorFieldNetwork": "TFN",
    "GTTensorFieldNetworkV2": "GTTFNv2",
    "HierarchicalTensorFieldNetwork": "HierTFN",
    "StochasticTensorFieldNetwork": "StochTFN",
    "OnEquivariantTensorFieldNetwork": "OnEquivTFN",
    "AttentionTensorFieldNetwork": "AttnTFN",
    "RelaxedOnEquivariantTensorFieldNetwork": "RelaxTFN",
    "HybridOnEquivariantTensorFieldNetwork": "HybridTFN",
}


def download_results():
    """Download results from cluster via rsync over sshpass."""
    local_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(local_dir, exist_ok=True)
    cmd = (
        f'sshpass -p \'{SSH_PASS}\' rsync -avz '
        f'-e "ssh -o StrictHostKeyChecking=no" '
        f'{CLUSTER}:{REMOTE_DIR}/ {local_dir}/'
    )
    print(f"Downloading results from cluster...")
    subprocess.run(cmd, shell=True, check=True)
    print(f"  Downloaded to {local_dir}/")


def load_results(results_dir):
    """Load all JSON result files."""
    pattern = os.path.join(results_dir, "shape_*.json")
    files = sorted(glob.glob(pattern))
    results = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            d["_file"] = os.path.basename(f)
            results.append(d)
        except Exception as e:
            print(f"  WARNING: Failed to load {f}: {e}")
    return results


def build_table(results):
    """Build a per-trial table from raw results."""
    rows = []
    for r in results:
        ds = r.get("dataset", "")
        model = r.get("model", "")
        trial = r.get("trial", 0)
        
        # Build variant key from robustness flags
        flags = []
        if r.get("attn_pool"):
            flags.append("attn-pool")
        if r.get("noise_aug"):
            flags.append("noise-aug")
        if r.get("dtm_filter"):
            flags.append("dtm-filter")
        if r.get("robust_knn"):
            flags.append("robust-knn")
        if r.get("train_robust_knn"):
            flags.append("train-robust-knn")
        if r.get("dtm_readout"):
            flags.append("dtm-readout")
        if r.get("readout_pool") and r.get("readout_pool") != "sum":
            flags.append("pool-%s" % r.get("readout_pool"))
        if r.get("norm_readout"):
            flags.append("norm-readout")
        if r.get("cov_feat"):
            flags.append("cov-feat")
        if r.get("aug_frac"):
            flags.append("frac%s" % r.get("aug_frac"))
        if r.get("multiscale"):
            flags.append("multiscale")
        if r.get("pd_fusion"):
            flags.append("pd-fusion")
        if r.get("denoise"):
            flags.append("denoise")
        if r.get("geom_reg"):
            flags.append("geom-reg")
        if r.get("feat_pd"):
            flags.append("feat-pd")
        if r.get("consistency"):
            flags.append("consistency-lam%s" % r.get("cons_lambda"))
        if r.get("train_dtm_readout"):
            flags.append("train-dtm-readout")
        for hp_name, hp_flag in (("max_order", "mo"), ("hidden_channels", "hc"),
                                 ("num_layers", "nl"), ("k_neighbors", "kn"),
                                 ("num_rbf", "rbf")):
            hpv = r.get(hp_name)
            if hpv is not None:
                flags.append("%s%s" % (hp_flag, hpv))
        variant = "_".join(flags) if flags else "baseline"
        
        # Handle different accuracy key names
        val_acc = r.get("best_val_accuracy", r.get("val_accuracy", None))
        clean_acc = r.get("clean_accuracy", None)
        noisy_acc = r.get("noisy_accuracy", None)
        epochs = r.get("num_epochs", None)
        params = r.get("params", None)
        n_train = r.get("n_train", None)
        n_test = r.get("n_test", None)
        n_points = r.get("n_points", None)
        num_classes = r.get("num_classes", None)
        batch_size = r.get("batch_size", None)
        
        rows.append({
            "dataset": ds,
            "model": model,
            "model_short": MODEL_SHORT.get(model, model),
            "variant": variant,
            "trial": trial,
            "val_acc": val_acc,
            "clean_acc": clean_acc,
            "noisy_acc": noisy_acc,
            "epochs": epochs,
            "params": params,
            "n_train": n_train,
            "n_test": n_test,
            "n_points": n_points,
            "num_classes": num_classes,
            "batch_size": batch_size,
        })
    return rows


def compute_summary(rows):
    """Compute mean±std summary grouped by (dataset, model, variant)."""
    grouped = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["model"], r["variant"])
        grouped[key].append(r)
    
    summary = []
    for (ds, model, variant), trials in sorted(grouped.items()):
        val_accs = [t["val_acc"] for t in trials if t["val_acc"] is not None]
        clean_accs = [t["clean_acc"] for t in trials if t["clean_acc"] is not None]
        noisy_accs = [t["noisy_acc"] for t in trials if t["noisy_acc"] is not None]
        
        entry = {
            "dataset": ds,
            "model": model,
            "model_short": MODEL_SHORT.get(model, model),
            "variant": variant,
            "n_trials": len(trials),
            "epochs": trials[0].get("epochs"),
            "params": trials[0].get("params"),
            "n_points": trials[0].get("n_points"),
            "num_classes": trials[0].get("num_classes"),
        }
        
        if val_accs:
            entry["val_acc_mean"] = np.mean(val_accs)
            entry["val_acc_std"] = np.std(val_accs)
        if clean_accs:
            entry["clean_acc_mean"] = np.mean(clean_accs)
            entry["clean_acc_std"] = np.std(clean_accs)
        if noisy_accs:
            entry["noisy_acc_mean"] = np.mean(noisy_accs)
            entry["noisy_acc_std"] = np.std(noisy_accs)
        
        summary.append(entry)
    return summary


def write_csv(rows, summary, output_dir):
    """Write per-trial and summary CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Per-trial CSV
    trial_path = os.path.join(output_dir, "results_all.csv")
    fieldnames = [
        "dataset", "model", "model_short", "variant", "trial",
        "val_acc", "clean_acc", "noisy_acc",
        "epochs", "params", "n_points", "num_classes", "batch_size",
    ]
    with open(trial_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: (x["dataset"], x["model"], x["variant"], x["trial"])):
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  Per-trial results: {trial_path} ({len(rows)} rows)")
    
    # Summary CSV
    summary_path = os.path.join(output_dir, "results_summary.csv")
    sfields = [
        "dataset", "model", "model_short", "variant", "n_trials",
        "val_acc_mean", "val_acc_std",
        "clean_acc_mean", "clean_acc_std",
        "noisy_acc_mean", "noisy_acc_std",
        "epochs", "params", "n_points", "num_classes",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sfields)
        writer.writeheader()
        for s in summary:
            writer.writerow({k: f"{s[k]:.4f}" if isinstance(s.get(k), float) else s.get(k, "") for k in sfields})
    print(f"  Summary (mean±std): {summary_path} ({len(summary)} rows)")
    
    # Missing report
    present = set((r["dataset"], r["model"]) for r in rows)
    missing = []
    for ds in DATASETS:
        for m in MODELS:
            if (ds, m) not in present:
                missing.append(f"  {ds} / {m}")
    if missing:
        print(f"\n  Missing ({len(missing)}):")
        for m in missing:
            print(f"    {m}")
    else:
        print(f"\n  All {len(DATASETS)*len(MODELS)} combinations present!")
    
    return trial_path, summary_path


def print_leaderboard(summary):
    """Print a quick leaderboard per dataset, with baseline vs robustness variants."""
    by_ds = defaultdict(list)
    for s in summary:
        by_ds[s["dataset"]].append(s)
    
    for ds in sorted(by_ds.keys()):
        items = sorted(by_ds[ds], key=lambda x: x.get("clean_acc_mean", 0), reverse=True)
        n_classes = items[0].get("num_classes", "?")
        print(f"\n{'='*80}")
        print(f"  {ds} ({n_classes} classes)")
        print(f"{'='*80}")
        print(f"  {'Model + Variant':<40s} {'Clean Acc':>12s} {'Noisy Acc':>12s}")
        print(f"  {'-'*40} {'-'*12} {'-'*12}")
        for s in items:
            label = s['model_short']
            if s.get('variant', 'baseline') != 'baseline':
                label += f" +{s['variant']}"
            clean = f"{s.get('clean_acc_mean',0)*100:.1f}±{s.get('clean_acc_std',0)*100:.1f}"
            noisy = f"{s.get('noisy_acc_mean',0)*100:.1f}±{s.get('noisy_acc_std',0)*100:.1f}"
            n = s['n_trials']
            print(f"  {label:<40s} {clean:>12s} {noisy:>12s}  (n={n})")


def main():
    parser = argparse.ArgumentParser(description="Consolidate shape classification results")
    parser.add_argument("--download", action="store_true", default=True,
                        help="Download from cluster before consolidating")
    parser.add_argument("--no-download", action="store_false", dest="download",
                        help="Skip download, use local results/ dir")
    parser.add_argument("--local", type=str, default=None,
                        help="Use local directory of JSON files")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.local:
        results_dir = args.local
    else:
        results_dir = os.path.join(script_dir, "results")
        if args.download:
            try:
                download_results()
            except Exception as e:
                print(f"  Download failed: {e}")
                print(f"  Using existing local results/ directory")
    
    print(f"\nLoading results from {results_dir}/")
    results = load_results(results_dir)
    print(f"  Found {len(results)} JSON result files")
    
    if not results:
        print("  No results found!")
        sys.exit(1)
    
    rows = build_table(results)
    summary = compute_summary(rows)
    
    print(f"\nWriting CSVs...")
    write_csv(rows, summary, script_dir)
    
    print(f"\n{'='*70}")
    print(f"  LEADERBOARD")
    print(f"{'='*70}")
    print_leaderboard(summary)


if __name__ == "__main__":
    main()
