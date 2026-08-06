"""
Robustness comparison: Direct PV-space distance analysis for PointNet, TFN, DMR families.

For each (dataset, model_group):
  1. Load trained checkpoints
  2. Compute PVs on clean test set
  3. Apply isometry augmentation (2D rotation + translation)
  4. Compute PVs on augmented test set
  5. Report L2 distances and accuracy drop per model family

Groups:
  - PointNet: PointNetTutorial, PointNet3D
  - TFN: OnEquivariantTensorFieldNetwork, AttentionTensorFieldNetwork, TensorFieldNetwork, GTTensorFieldNetwork
  - DMR: DistanceMatrixRaggedModel, ScalarDistanceDeepSet

Usage:
    python robustness_comparison.py [results_dir]
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/ablations'
os.makedirs('ablation_plots', exist_ok=True)

FAMILY_MAP = {
    'PointNetTutorial': 'PointNet', 'PointNetTutorial_GS': 'PointNet',
    'PointNet3D': 'PointNet', 'PointNet3D_GS': 'PointNet',
    'OnEquivariantTensorFieldNetwork': 'TFN', 'OnEquivariantTensorFieldNetwork_GS': 'TFN',
    'AttentionTensorFieldNetwork': 'TFN', 'AttentionTensorFieldNetwork_GS': 'TFN',
    'TensorFieldNetwork': 'TFN', 'TensorFieldNetwork_GS': 'TFN',
    'GTTensorFieldNetwork': 'TFN', 'GTTensorFieldNetwork_GS': 'TFN',
    'GTTensorFieldNetworkV2': 'TFN', 'GTTensorFieldNetworkV2_GS': 'TFN',
    'HierarchicalGTTFN': 'TFN', 'HierarchicalGTTFN_GS': 'TFN',
    'HierarchicalTensorFieldNetwork': 'TFN', 'HierarchicalTensorFieldNetwork_GS': 'TFN',
    'StochasticTensorFieldNetwork': 'TFN', 'StochasticTensorFieldNetwork_GS': 'TFN',
    'CrossAttentionTensorFieldNetwork': 'TFN', 'CrossAttentionTensorFieldNetwork_GS': 'TFN',
    'DistanceMatrixRaggedModel': 'DMR', 'DistanceMatrixRaggedModel_GS': 'DMR',
    'ScalarDistanceDeepSet': 'DMR', 'ScalarDistanceDeepSet_GS': 'DMR',
}

# ─── Load isometry results ──────────────────────────────────────────────────
iso_files = glob.glob(os.path.join(results_dir, 'isometry_*.json'))
iso_data = []
for f in iso_files:
    try:
        d = json.load(open(f))
        if not np.isnan(d.get('mean_l2_dist', float('nan'))):
            iso_data.append(d)
    except Exception:
        pass

if not iso_data:
    print("No isometry results found in", results_dir)
    print("Expected files: isometry_<dataset>_<model>_aug<N>_t<trial>.json")
    sys.exit(1)

df_iso = pd.DataFrame(iso_data)
df_iso['family'] = df_iso['model_label'].map(FAMILY_MAP).fillna('Other')
df_iso['has_gs'] = df_iso['model_label'].str.endswith('_GS')

print(f"Loaded {len(df_iso)} isometry results")
print(f"  Datasets: {df_iso['dataset'].nunique()}")
print(f"  Models: {df_iso['model_label'].nunique()}")
print(f"  Families: {df_iso['family'].unique().tolist()}")

# ─── Summary table ──────────────────────────────────────────────────────────
print("\n" + "="*80)
print("ISOMETRY ROBUSTNESS SUMMARY")
print("="*80)
print(f"{'Dataset':<35s} {'Family':<10s} {'Model':<45s} {'L2_dist':>10s} {'Clean%':>8s} {'Aug%':>8s}")
print("-"*80)

for ds in sorted(df_iso['dataset'].unique()):
    sub = df_iso[df_iso['dataset'] == ds].sort_values('mean_l2_dist')
    for _, row in sub.iterrows():
        print(f"{row['dataset']:<35s} {row['family']:<10s} {row['model_label']:<45s} "
              f"{row['mean_l2_dist']:10.4f} {row['accuracy_clean']*100:7.1f}% "
              f"{row['accuracy_augmented']*100:7.1f}%")

# ─── Per-family aggregation ──────────────────────────────────────────────────
print("\n" + "="*80)
print("MEAN L2 DISTANCE BY FAMILY (lower = more robust)")
print("="*80)

fam_agg = df_iso.groupby(['family', 'has_gs']).agg(
    mean_l2=('mean_l2_dist', 'mean'), std_l2=('mean_l2_dist', 'std'),
    mean_clean=('accuracy_clean', 'mean'), mean_aug=('accuracy_augmented', 'mean'),
    n=('dataset', 'count')).reset_index()

for family in ['PointNet', 'TFN', 'DMR']:
    sub = fam_agg[fam_agg['family'] == family]
    if sub.empty:
        continue
    for _, row in sub.iterrows():
        variant = 'GS' if row['has_gs'] else 'raw'
        print(f"  {family:10s} {variant:4s}: L2={row['mean_l2']:.4f}±{row['std_l2']:.4f}  "
              f"clean={row['mean_clean']*100:.1f}%  aug={row['mean_aug']*100:.1f}%  "
              f"(n={int(row['n'])})")

# ─── Accuracy drop per family ────────────────────────────────────────────────
print("\n" + "="*80)
print("ACCURACY DROP UNDER ISOMETRY (lower = more robust)")
print("="*80)

df_iso['acc_drop'] = df_iso['accuracy_clean'] - df_iso['accuracy_augmented']
drop_agg = df_iso.groupby(['family', 'has_gs']).agg(
    mean_drop=('acc_drop', 'mean'), std_drop=('acc_drop', 'std'),
    n=('dataset', 'count')).reset_index()

for family in ['PointNet', 'TFN', 'DMR']:
    sub = drop_agg[drop_agg['family'] == family]
    if sub.empty:
        continue
    for _, row in sub.iterrows():
        variant = 'GS' if row['has_gs'] else 'raw'
        print(f"  {family:10s} {variant:4s}: drop={row['mean_drop']*100:.1f}%±{row['std_drop']*100:.1f}%  (n={int(row['n'])})")

# ─── Plots ──────────────────────────────────────────────────────────────────

# 1. Grouped bar chart: L2 distance by family
fig, ax = plt.subplots(figsize=(8, 5))
families = ['PointNet', 'TFN', 'DMR']
variants = [False, True]
x = np.arange(len(families))
width = 0.35
for vi, (has_gs, variant_label) in enumerate(zip(variants, ['Raw', 'GS'])):
    vals, errs = [], []
    for fam in families:
        sub = fam_agg[(fam_agg['family'] == fam) & (fam_agg['has_gs'] == has_gs)]
        vals.append(sub['mean_l2'].values[0] if len(sub) else 0)
        errs.append(sub['std_l2'].values[0] if len(sub) else 0)
    ax.bar(x + vi * width, vals, width, yerr=errs, label=variant_label, capsize=4)
ax.set_xlabel('Model family')
ax.set_ylabel('Mean L2 distance in PV space')
ax.set_title('Isometry robustness: PV-space distance under rotation + translation')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(families)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('ablation_plots/isometry_l2_by_family.png', dpi=150)
plt.close(fig)
print("\nSaved ablation_plots/isometry_l2_by_family.png")

# 2. Grouped bar chart: accuracy drop by family
fig, ax = plt.subplots(figsize=(8, 5))
for vi, (has_gs, variant_label) in enumerate(zip(variants, ['Raw', 'GS'])):
    vals, errs = [], []
    for fam in families:
        sub = drop_agg[(drop_agg['family'] == fam) & (drop_agg['has_gs'] == has_gs)]
        vals.append(sub['mean_drop'].values[0] * 100 if len(sub) else 0)
        errs.append(sub['std_drop'].values[0] * 100 if len(sub) else 0)
    ax.bar(x + vi * width, vals, width, yerr=errs, label=variant_label, capsize=4)
ax.set_xlabel('Model family')
ax.set_ylabel('Accuracy drop under isometry (%)')
ax.set_title('Classification accuracy drop under SO(2) + translation')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(families)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('ablation_plots/isometry_acc_drop_by_family.png', dpi=150)
plt.close(fig)
print("Saved ablation_plots/isometry_acc_drop_by_family.png")

# 3. Per-dataset heatmap of L2 distances
datasets = sorted(df_iso['dataset'].unique())
models = sorted(df_iso['model_label'].unique())
l2_matrix = np.full((len(datasets), len(models)), np.nan)
for i, ds in enumerate(datasets):
    for j, ml in enumerate(models):
        sub = df_iso[(df_iso['dataset'] == ds) & (df_iso['model_label'] == ml)]
        if len(sub):
            l2_matrix[i, j] = sub['mean_l2_dist'].values[0]

fig, ax = plt.subplots(figsize=(max(12, len(models) * 0.6), max(6, len(datasets) * 0.4)))
im = ax.imshow(l2_matrix, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(datasets)))
ax.set_yticklabels(datasets, fontsize=7)
ax.set_title('PV-space L2 distance under isometry (lower = more robust)')
fig.colorbar(im, ax=ax, label='L2 distance')
fig.tight_layout()
fig.savefig('ablation_plots/isometry_l2_heatmap.png', dpi=150)
plt.close(fig)
print("Saved ablation_plots/isometry_l2_heatmap.png")

# 4. Scatter: clean accuracy vs isometry L2 distance (per model)
fig, ax = plt.subplots(figsize=(10, 6))
color_map = {'PointNet': 'C0', 'TFN': 'C1', 'DMR': 'C2'}
for _, row in df_iso.iterrows():
    fam = row['family']
    c = color_map.get(fam, 'gray')
    marker = '^' if row['has_gs'] else 'o'
    ax.scatter(row['accuracy_clean'] * 100, row['mean_l2_dist'],
               c=c, marker=marker, s=60, alpha=0.7, edgecolors='black', linewidths=0.5)
# Legend entries
for fam, c in color_map.items():
    ax.scatter([], [], c=c, marker='o', s=60, label=f'{fam} (raw)')
    ax.scatter([], [], c=c, marker='^', s=60, label=f'{fam} (GS)')
ax.set_xlabel('Clean test accuracy (%)')
ax.set_ylabel('Mean L2 distance in PV space')
ax.set_title('Accuracy vs. isometry robustness')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('ablation_plots/isometry_scatter_acc_vs_l2.png', dpi=150)
plt.close(fig)
print("Saved ablation_plots/isometry_scatter_acc_vs_l2.png")

print("\nDone.")
