import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAMES = [
    'OnEquivariantTensorFieldNetwork',
    'GTTensorFieldNetwork',
    'TensorFieldNetwork',
    'GTTensorFieldNetworkV2',
    'HierarchicalGTTFN',
    'HierarchicalTensorFieldNetwork',
    'AttentionTensorFieldNetwork',
    'StochasticTensorFieldNetwork',
    'CrossAttentionTensorFieldNetwork',
    'RelaxedOnEquivariantTensorFieldNetwork',
    'HybridOnEquivariantTensorFieldNetwork',
    'PointNet3D',
    'ScalarDistanceDeepSet',
    'PointNetTutorial',
    'ScalarInputMLP',
    'MultiInputModel',
    'RaggedPersistenceModel',
    'DistanceMatrixRaggedModel',
]

MODEL_COLORS = {
    'OnEquivariantTensorFieldNetwork': '#1f77b4',
    'GTTensorFieldNetwork': '#ff7f0e',
    'TensorFieldNetwork': '#2ca02c',
    'GTTensorFieldNetworkV2': '#d62728',
    'HierarchicalGTTFN': '#9467bd',
    'HierarchicalTensorFieldNetwork': '#8c564b',
    'AttentionTensorFieldNetwork': '#e377c2',
    'StochasticTensorFieldNetwork': '#7f7f7f',
    'CrossAttentionTensorFieldNetwork': '#bcbd22',
    'RelaxedOnEquivariantTensorFieldNetwork': '#17becf',
    'HybridOnEquivariantTensorFieldNetwork': '#aec7e8',
    'PointNet3D': '#ffbb78',
    'ScalarDistanceDeepSet': '#98df8a',
    'PointNetTutorial': '#ff9896',
    'ScalarInputMLP': '#c5b0d5',
    'MultiInputModel': '#c49c94',
    'RaggedPersistenceModel': '#f7b6d2',
    'DistanceMatrixRaggedModel': '#c7c7c7',
}

FRACTIONS = [10, 20, 30, 50, 70, 100]
TRIALS = [0, 1, 2, 3]


def load_results(fs):
    by_dataset = {}
    for fpath in fs:
        fname = os.path.basename(fpath)
        parts = fname.split('_')
        dataset = parts[2]
        by_dataset.setdefault(dataset, []).append(fpath)
    return by_dataset


def gather_model_data(file_list):
    results_by_model = {}
    for fpath in file_list:
        with open(fpath) as f:
            data = json.load(f)
        model = data['model']
        frac = data['fraction_pct']
        trial = data['trial']
        te_acc = data.get('xgb_test_acc')
        if te_acc is None or np.isnan(te_acc):
            continue
        results_by_model.setdefault(model, []).append((frac, trial, te_acc))
    return results_by_model


def compute_model_stats(results_by_model, model_order):
    stats = {}
    for model in model_order:
        if model not in results_by_model:
            continue
        frac_vals = {}
        for frac, trial, acc in results_by_model[model]:
            frac_vals.setdefault(frac, []).append(acc)
        fracs_sorted = sorted(frac_vals.keys())
        means = [np.mean(frac_vals[f]) * 100 for f in fracs_sorted]
        stds = [np.std(frac_vals[f]) * 100 for f in fracs_sorted]
        stats[model] = (fracs_sorted, means, stds)
    return stats


def draw_grouped_bar(ax, stats, title):
    models_present = [m for m in MODEL_NAMES if m in stats]
    if not models_present:
        return
    n_groups = len(FRACTIONS)
    n_models = len(models_present)
    bar_width = 0.8 / n_models
    x = np.arange(n_groups)

    for i, model in enumerate(models_present):
        fracs_sorted, means, stds = stats[model]
        frac_indices = [FRACTIONS.index(f) for f in fracs_sorted]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x[frac_indices] + offset, means, bar_width,
               yerr=stds, label=model, color=MODEL_COLORS.get(model, '#333333'),
               capsize=2, error_kw={'linewidth': 0.8})

    ax.set_xlabel('Training fraction (%)', fontsize=13)
    ax.set_ylabel('Test accuracy (%)', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(FRACTIONS)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=7, loc='best', ncol=2)


fs = glob.glob(os.path.join(RESULTS_DIR, 'ablation_train_*.json'))
by_dataset = load_results(fs)
dataset_names = sorted(by_dataset.keys())

# ─── Per-dataset bar plots ────────────────────────────────────────────────
for ds in dataset_names:
    fig, ax = plt.subplots(figsize=(max(10, len(MODEL_NAMES) * 0.5 + 4), 8))
    results_by_model = gather_model_data(by_dataset[ds])
    stats = compute_model_stats(results_by_model, MODEL_NAMES)
    draw_grouped_bar(ax, stats, f'{ds}')
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f'ablation_train_bar_{ds}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

# ─── Overall summary ──────────────────────────────────────────────────────
all_stats = {}
for model in MODEL_NAMES:
    model_data = {}
    for fpath in fs:
        with open(fpath) as f:
            data = json.load(f)
        if data['model'] != model:
            continue
        te_acc = data.get('xgb_test_acc')
        if te_acc is None or np.isnan(te_acc):
            continue
        model_data.setdefault(data['fraction_pct'], []).append(te_acc)
    if model_data:
        fracs_sorted = sorted(model_data.keys())
        means = [np.mean(model_data[f]) * 100 for f in fracs_sorted]
        stds = [np.std(model_data[f]) * 100 for f in fracs_sorted]
        all_stats[model] = (fracs_sorted, means, stds)

fig, ax = plt.subplots(figsize=(max(14, len(MODEL_NAMES) * 0.5 + 4), 9))
draw_grouped_bar(ax, all_stats, 'Training ablation — average across all datasets')
fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'ablation_train_bar_overall.png')
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  Saved {out_path}")

# ─── Working datasets summary ────────────────────────────────────────────
WORKING_DATASETS = {'CBF', 'ECG200', 'ECG5000', 'GunPoint', 'Plane',
                    'PowerCons', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2',
                    'TwoLeadECG', 'UMD'}
working_stats = {}
for model in MODEL_NAMES:
    model_data = {}
    for fpath in fs:
        with open(fpath) as f:
            data = json.load(f)
        if data['model'] != model or data['dataset'] not in WORKING_DATASETS:
            continue
        te_acc = data.get('xgb_test_acc')
        if te_acc is None or np.isnan(te_acc):
            continue
        model_data.setdefault(data['fraction_pct'], []).append(te_acc)
    if model_data:
        fracs_sorted = sorted(model_data.keys())
        means = [np.mean(model_data[f]) * 100 for f in fracs_sorted]
        stds = [np.std(model_data[f]) * 100 for f in fracs_sorted]
        working_stats[model] = (fracs_sorted, means, stds)

fig, ax = plt.subplots(figsize=(max(14, len(MODEL_NAMES) * 0.5 + 4), 9))
draw_grouped_bar(ax, working_stats, 'Training ablation — average across 10 working TFN datasets')
fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'ablation_train_bar_working_datasets.png')
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  Saved {out_path}")

print(f"\nDone! {len(dataset_names)} per-dataset + 2 summary bar plots in {OUT_DIR}")
