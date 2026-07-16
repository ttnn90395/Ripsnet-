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

MODEL_STYLES = {}
for m in MODEL_NAMES:
    if 'TensorField' in m or 'GTTFN' in m or 'TFN' in m or 'GTTensor' in m:
        MODEL_STYLES[m] = '-'
    elif 'PointNet' in m:
        MODEL_STYLES[m] = '--'
    elif 'Ragged' in m or 'DistanceMatrix' in m:
        MODEL_STYLES[m] = ':'
    elif 'ScalarInputMLP' in m:
        MODEL_STYLES[m] = '-.'
    elif 'ScalarDistanceDeepSet' in m:
        MODEL_STYLES[m] = (0, (3, 1, 1, 1))
    elif 'MultiInput' in m:
        MODEL_STYLES[m] = (0, (3, 1, 1, 1, 1, 1))
    else:
        MODEL_STYLES[m] = '-'

FRACTIONS = [10, 20, 30, 50, 70, 100]
TRIALS = [0, 1, 2, 3]

fs = glob.glob(os.path.join(RESULTS_DIR, 'ablation_train_*.json'))
by_dataset = {}
for fpath in fs:
    fname = os.path.basename(fpath)
    parts = fname.split('_')
    dataset = parts[2]
    model = parts[3]
    by_dataset.setdefault(dataset, []).append(fpath)

dataset_names = sorted(by_dataset.keys())

for ds in dataset_names:
    fig, ax = plt.subplots(figsize=(12, 8))
    results_by_model = {}
    for fpath in by_dataset[ds]:
        with open(fpath) as f:
            data = json.load(f)
        model = data['model']
        frac = data['fraction_pct']
        trial = data['trial']
        te_acc = data.get('xgb_test_acc')
        if te_acc is None or np.isnan(te_acc):
            continue
        results_by_model.setdefault(model, []).append((frac, trial, te_acc))

    for model in MODEL_NAMES:
        if model not in results_by_model:
            continue
        frac_vals = {}
        for frac, trial, acc in results_by_model[model]:
            frac_vals.setdefault(frac, []).append(acc)
        fracs_sorted = sorted(frac_vals.keys())
        means = [np.mean(frac_vals[f]) * 100 for f in fracs_sorted]
        stds = [np.std(frac_vals[f]) * 100 for f in fracs_sorted]
        color = MODEL_COLORS.get(model, '#333333')
        style = MODEL_STYLES.get(model, '-')
        ax.errorbar(fracs_sorted, means, yerr=stds, label=model,
                    color=color, linestyle=style, marker='o', capsize=3, linewidth=1.5)

    ax.set_xlabel('Training fraction (%)', fontsize=13)
    ax.set_ylabel('Test accuracy (%)', fontsize=13)
    ax.set_title(f'{ds}', fontsize=15, fontweight='bold')
    ax.set_xticks(FRACTIONS)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', ncol=2)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f'ablation_train_{ds}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

# ─── Overall summary: average across all datasets ──────────────────────────
all_models_data = {}
for fpath in fs:
    with open(fpath) as f:
        data = json.load(f)
    model = data['model']
    frac = data['fraction_pct']
    te_acc = data.get('xgb_test_acc')
    if te_acc is None or np.isnan(te_acc):
        continue
    all_models_data.setdefault(model, {}).setdefault(frac, []).append(te_acc)

fig, ax = plt.subplots(figsize=(14, 9))
for model in MODEL_NAMES:
    if model not in all_models_data:
        continue
    fracs_sorted = sorted(all_models_data[model].keys())
    means = [np.mean(all_models_data[model][f]) * 100 for f in fracs_sorted]
    stds = [np.std(all_models_data[model][f]) * 100 for f in fracs_sorted]
    color = MODEL_COLORS.get(model, '#333333')
    style = MODEL_STYLES.get(model, '-')
    ax.errorbar(fracs_sorted, means, yerr=stds, label=model,
                color=color, linestyle=style, marker='o', capsize=3, linewidth=1.5)

ax.set_xlabel('Training fraction (%)', fontsize=14)
ax.set_ylabel('Test accuracy (%)', fontsize=14)
ax.set_title('Training ablation — average across all datasets', fontsize=16, fontweight='bold')
ax.set_xticks(FRACTIONS)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='best', ncol=2)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'ablation_train_overall.png')
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  Saved {out_path}")

# ─── Summary: 10 working TFN datasets ──────────────────────────────────────
WORKING_DATASETS = {'CBF', 'ECG200', 'ECG5000', 'GunPoint', 'Plane',
                    'PowerCons', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2',
                    'TwoLeadECG', 'UMD'}
working_data = {}
for fpath in fs:
    with open(fpath) as f:
        data = json.load(f)
    if data['dataset'] not in WORKING_DATASETS:
        continue
    model = data['model']
    frac = data['fraction_pct']
    te_acc = data.get('xgb_test_acc')
    if te_acc is None or np.isnan(te_acc):
        continue
    working_data.setdefault(model, {}).setdefault(frac, []).append(te_acc)

fig, ax = plt.subplots(figsize=(14, 9))
for model in MODEL_NAMES:
    if model not in working_data:
        continue
    fracs_sorted = sorted(working_data[model].keys())
    means = [np.mean(working_data[model][f]) * 100 for f in fracs_sorted]
    stds = [np.std(working_data[model][f]) * 100 for f in fracs_sorted]
    color = MODEL_COLORS.get(model, '#333333')
    style = MODEL_STYLES.get(model, '-')
    ax.errorbar(fracs_sorted, means, yerr=stds, label=model,
                color=color, linestyle=style, marker='o', capsize=3, linewidth=1.5)

ax.set_xlabel('Training fraction (%)', fontsize=14)
ax.set_ylabel('Test accuracy (%)', fontsize=14)
ax.set_title('Training ablation — average across 10 working TFN datasets', fontsize=16, fontweight='bold')
ax.set_xticks(FRACTIONS)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='best', ncol=2)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'ablation_train_working_datasets.png')
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  Saved {out_path}")

print(f"\nDone! {len(dataset_names)} per-dataset + 2 summary plots in {OUT_DIR}")
