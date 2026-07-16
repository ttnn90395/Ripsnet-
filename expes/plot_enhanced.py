"""Enhanced TFN results: MLP vs XGBoost comparison plots."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "enhanced")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Load all results
data = []
for f in glob.glob(os.path.join(RESULTS_DIR, "train_*.json")):
    with open(f) as fh:
        data.append(json.load(fh))

print("Loaded %d results" % len(data))

DATASETS = ["CBF", "ECG200", "ECG5000", "GunPoint", "Plane", "PowerCons",
            "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "TwoLeadECG", "UMD"]
FRACTIONS = [10, 20, 30, 50, 70, 100]

DS_LABELS = {
    "CBF": "CBF", "ECG200": "ECG200", "ECG5000": "ECG5000",
    "GunPoint": "GunPoint", "Plane": "Plane", "PowerCons": "PowerCons",
    "SonyAIBORobotSurface1": "SonyAI-1", "SonyAIBORobotSurface2": "SonyAI-2",
    "TwoLeadECG": "TwoLeadECG", "UMD": "UMD"
}

# ─── Data structure ──────────────────────────────────────────────────────────
# Collect: per dataset, per model, per fraction, per trial
# mlp_accs[ds][model][frac] = [trial values]
# xgb_accs[ds][model][frac] = [trial values]
mlp_accs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
xgb_accs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for d in data:
    ds = d["dataset"]
    model = d["model"]
    frac = d["fraction_pct"]
    mlp = d.get("mlp_test_acc")
    xgb = d.get("xgb_test_acc")
    if mlp is not None and not (isinstance(mlp, float) and np.isnan(mlp)):
        mlp_accs[ds][model][frac].append(mlp)
    if xgb is not None and not (isinstance(xgb, float) and np.isnan(xgb)):
        xgb_accs[ds][model][frac].append(xgb)

MODELS = sorted(set(list(mlp_accs[DATASETS[0]].keys()) + list(xgb_accs[DATASETS[0]].keys())))


# ─── Plot 1: MLP vs XGBoost per-dataset at 100% (grouped bar) ──────────────
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx // 5, idx % 5]
    mlp_means = [np.mean(mlp_accs[ds].get(m, {}).get(100, [0])) * 100 for m in MODELS]
    xgb_means = [np.mean(xgb_accs[ds].get(m, {}).get(100, [0])) * 100 for m in MODELS]
    x = np.arange(len(MODELS))
    w = 0.35
    ax.bar(x - w / 2, mlp_means, w, label="MLP(+aug)", color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, xgb_means, w, label="XGBoost", color="coral", alpha=0.8)
    ax.set_title(DS_LABELS.get(ds, ds), fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("TensorFieldNetwork", "TFN").replace("OnEquivariant", "OnEq.").replace("Hybrid", "Hyb.").replace("Attention", "Attn.").replace("DistanceMatrixRaggedModel", "DistMat").replace("PointNet3D", "PN3D") for m in MODELS], rotation=45, ha="right", fontsize=6)
    ax.set_ylim(0, 105)
    if idx == 0:
        ax.legend(fontsize=8)
fig.suptitle("Enhanced TFN: MLP(+aug) vs XGBoost at 100% Training Data", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "enhanced_mlp_vs_xgb_100pct.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: enhanced_mlp_vs_xgb_100pct.png")

# ─── Plot 2: Data efficiency curves (MLP vs XGBoost averaged across datasets) ─
fig, ax = plt.subplots(figsize=(10, 6))
mlp_avg, xgb_avg = [], []
mlp_se, xgb_se = [], []
for frac in FRACTIONS:
    m_vals = []
    x_vals = []
    for ds in DATASETS:
        m_vals.extend(mlp_accs[ds].get("all_models", {}).get(frac, []))
        x_vals.extend(xgb_accs[ds].get("all_models", {}).get(frac, []))
    # Compute across all models
    m_all, x_all = [], []
    for ds in DATASETS:
        for model in MODELS:
            m_all.extend([v * 100 for v in mlp_accs[ds].get(model, {}).get(frac, [])])
            x_all.extend([v * 100 for v in xgb_accs[ds].get(model, {}).get(frac, [])])
    mlp_avg.append(np.mean(m_all) if m_all else 0)
    xgb_avg.append(np.mean(x_all) if x_all else 0)
    mlp_se.append(np.std(m_all) / np.sqrt(len(m_all)) if m_all else 0)
    xgb_se.append(np.std(x_all) / np.sqrt(len(x_all)) if x_all else 0)

ax.errorbar(FRACTIONS, mlp_avg, yerr=mlp_se, marker="o", capsize=3, label="MLP(+aug)", linewidth=2, color="steelblue")
ax.errorbar(FRACTIONS, xgb_avg, yerr=xgb_se, marker="s", capsize=3, label="XGBoost", linewidth=2, color="coral")
ax.set_xlabel("Training Fraction (%)", fontsize=12)
ax.set_ylabel("Mean Test Accuracy (%)", fontsize=12)
ax.set_title("Data Efficiency: MLP(+aug) vs XGBoost (mean ± SE across all models & datasets)", fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(FRACTIONS)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "enhanced_data_efficiency.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: enhanced_data_efficiency.png")

# ─── Plot 3: Per-model comparison at 100% (MLP vs XGBoost) ──────────────────
fig, ax = plt.subplots(figsize=(10, 6))
model_mlp = defaultdict(list)
model_xgb = defaultdict(list)
for ds in DATASETS:
    for model in MODELS:
        model_mlp[model].extend([v * 100 for v in mlp_accs[ds].get(model, {}).get(100, [])])
        model_xgb[model].extend([v * 100 for v in xgb_accs[ds].get(model, {}).get(100, [])])

model_names_short = {
    "AttentionTensorFieldNetwork": "AttentionTFN",
    "DistanceMatrixRaggedModel": "DistMatrixRagged",
    "HybridOnEquivariantTensorFieldNetwork": "HybridOnEqTFN",
    "OnEquivariantTensorFieldNetwork": "OnEqTFN",
    "PointNet3D": "PointNet3D",
    "TensorFieldNetwork": "TensorFieldNetwork",
}

models_sorted = sorted(model_mlp.keys())
x = np.arange(len(models_sorted))
w = 0.35
mlp_means = [np.mean(model_mlp[m]) if model_mlp[m] else 0 for m in models_sorted]
xgb_means = [np.mean(model_xgb[m]) if model_xgb[m] else 0 for m in models_sorted]
mlp_errs = [np.std(model_mlp[m]) / np.sqrt(len(model_mlp[m])) if len(model_mlp[m]) > 1 else 0 for m in models_sorted]
xgb_errs = [np.std(model_xgb[m]) / np.sqrt(len(model_xgb[m])) if len(model_xgb[m]) > 1 else 0 for m in models_sorted]

ax.barh(x + w / 2, mlp_means, w, xerr=mlp_errs, capsize=3, label="MLP(+aug)", color="steelblue", alpha=0.8)
ax.barh(x - w / 2, xgb_means, w, xerr=xgb_errs, capsize=3, label="XGBoost", color="coral", alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels([model_names_short.get(m, m) for m in models_sorted], fontsize=10)
ax.set_xlabel("Mean Test Accuracy (%)", fontsize=12)
ax.set_title("Per-Model: MLP(+aug) vs XGBoost at 100% Training", fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "enhanced_per_model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: enhanced_per_model_comparison.png")

# ─── Plot 4: Per-dataset heatmap of XGBoost at 100% by model ───────────────
fig, ax = plt.subplots(figsize=(12, 5))
heatmap = np.zeros((len(MODELS), len(DATASETS)))
for i, model in enumerate(MODELS):
    for j, ds in enumerate(DATASETS):
        vals = xgb_accs[ds].get(model, {}).get(100, [])
        heatmap[i, j] = np.mean(vals) * 100 if vals else 0

im = ax.imshow(heatmap, cmap="RdYlGn", vmin=40, vmax=100)
ax.set_xticks(range(len(DATASETS)))
ax.set_xticklabels([DS_LABELS.get(ds, ds) for ds in DATASETS], rotation=45, ha="right", fontsize=10)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels([model_names_short.get(m, m) for m in MODELS], fontsize=10)
for i in range(len(MODELS)):
    for j in range(len(DATASETS)):
        ax.text(j, i, "%.1f" % heatmap[i, j], ha="center", va="center", fontsize=8,
                color="white" if heatmap[i, j] < 55 else "black")
plt.colorbar(im, ax=ax, label="Test Accuracy (%)")
ax.set_title("XGBoost Test Accuracy at 100% by Dataset & Model", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "enhanced_xgb_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: enhanced_xgb_heatmap.png")

# ─── Plot 5: Per-dataset heatmap of MLP(+aug) at 100% by model ─────────────
fig, ax = plt.subplots(figsize=(12, 5))
heatmap_mlp = np.zeros((len(MODELS), len(DATASETS)))
for i, model in enumerate(MODELS):
    for j, ds in enumerate(DATASETS):
        vals = mlp_accs[ds].get(model, {}).get(100, [])
        heatmap_mlp[i, j] = np.mean(vals) * 100 if vals else 0

im = ax.imshow(heatmap_mlp, cmap="RdYlGn", vmin=40, vmax=100)
ax.set_xticks(range(len(DATASETS)))
ax.set_xticklabels([DS_LABELS.get(ds, ds) for ds in DATASETS], rotation=45, ha="right", fontsize=10)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels([model_names_short.get(m, m) for m in MODELS], fontsize=10)
for i in range(len(MODELS)):
    for j in range(len(DATASETS)):
        ax.text(j, i, "%.1f" % heatmap_mlp[i, j], ha="center", va="center", fontsize=8,
                color="white" if heatmap_mlp[i, j] < 55 else "black")
plt.colorbar(im, ax=ax, label="Test Accuracy (%)")
ax.set_title("MLP(+aug) Test Accuracy at 100% by Dataset & Model", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "enhanced_mlp_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: enhanced_mlp_heatmap.png")

print("\nDone! All plots saved to", PLOT_DIR)
