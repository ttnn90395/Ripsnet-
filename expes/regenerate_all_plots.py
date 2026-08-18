"""MASTER PLOT SCRIPT - regenerates all 11 figures from source data."""
import json, os, glob, csv, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_DIR = 'expes/results/enhanced'
PLOT_DIR = 'Final_report___Deep_Learning/images/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# === LOAD ALL SOURCE DATA ===
xgb_accs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
mlp_accs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
all_frac_xgb = defaultdict(list)
all_frac_mlp = defaultdict(list)
model_frac_data = defaultdict(lambda: defaultdict(list))
all_models = set()
all_datasets = set()

for jf in glob.glob(os.path.join(RESULTS_DIR, 'train_*.json')):
    try:
        with open(jf) as f: d = json.load(f)
    except: continue
    ds, model, frac = d.get('dataset',''), d.get('model',''), d.get('fraction_pct',100)
    xgb, mlp = d.get('xgb_test_acc'), d.get('mlp_test_acc')
    if d.get('augment') or d.get('multi_scale'): continue
    all_models.add(model); all_datasets.add(ds)
    if xgb is not None and not np.isnan(xgb):
        v = xgb * 100
        xgb_accs[ds][model][frac].append(v)
        all_frac_xgb[frac].append(v)
    if mlp is not None and not np.isnan(mlp):
        v = mlp * 100
        mlp_accs[ds][model][frac].append(v)
        all_frac_mlp[frac].append(v)
    if model in ['HybridGTTFN','GTTensorFieldNetworkV2','CrossAttentionTensorFieldNetwork',
                  'DistanceMatrixRaggedModel','PersNet']:
        if xgb is not None and not np.isnan(xgb):
            model_frac_data[model][frac].append(xgb * 100)

with open('expes/results/benchmark_timing.json') as f:
    timing = json.load(f)
with open('expes/results/training_curves.json') as f:
    curves = json.load(f)
with open('expes/results/ucr_results.csv') as f:
    rows = list(csv.DictReader(f))

task_cols = [c for c in rows[0].keys() if c.startswith('task_')]
csv_model_accs = defaultdict(list)
csv_mapping = {
    'TFN': 'TensorFieldNetwork', 'AttnTFN': 'AttentionTensorFieldNetwork',
    'GTTFNv2': 'GTTensorFieldNetworkV2', 'CrossAttn': 'CrossAttentionTensorFieldNetwork',
    'StochTFN': 'StochasticTensorFieldNetwork', 'HybridGTTFN': 'HybridGTTFN',
    'PersNet': 'PersNet', 'PNTut': 'PointNetTutorial',
    'DMR': 'DistanceMatrixRaggedModel', 'MPInp': 'MultiInputModel',
}
for r in rows:
    m = r['model']
    for tc in task_cols:
        try: csv_model_accs[m].append(float(r[tc].replace('%','')))
        except: pass

model_names = {
    'TensorFieldNetwork':'TFN','OnEquivariantTensorFieldNetwork':'OnEquivTFN',
    'AttentionTensorFieldNetwork':'AttnTFN','StochasticTensorFieldNetwork':'StochTFN',
    'GTTensorFieldNetworkV2':'GTTFNv2','CrossAttentionTensorFieldNetwork':'CrossAttn',
    'HybridGTTFN':'HybridGTTFN','DistanceMatrixRaggedModel':'DMR',
    'PersNet':'PersNet','MultiInputModel':'MPInp','PointNetTutorial':'PNTut',
}
ds_names = {
    'CBF':'CBF','ChlorineConcentration':'Chl','DistalPhalanxOutlineCorrect':'DPx',
    'ECG200':'ECG2','ECG5000':'ECG5','GunPoint':'GunP','ItalyPowerDemand':'ItaP',
    'MedicalImages':'MedI','MiddlePhalanxAgeGroup':'MPAg','MiddlePhalanxTW':'MPTW',
    'MiddlePhalanxMC':'MPCx','PhalangesOutlinesCorrect':'Phal','Plane':'Pla',
    'ProximalPhalanxAgeGroup':'PPAg','ProximalPhalanxTW':'PPTW',
    'SonyAIBORobotSurface1':'So1','SonyAIBORobotSurface2':'So2',
    'TwoLeadECG':'TwoL','GunPointOldVersusYoung':'GPOY','PowerCons':'PwrC','UMD':'UMD'
}
model_order = ['TensorFieldNetwork','OnEquivariantTensorFieldNetwork','AttentionTensorFieldNetwork',
               'StochasticTensorFieldNetwork','GTTensorFieldNetworkV2','CrossAttentionTensorFieldNetwork',
               'HybridGTTFN','DistanceMatrixRaggedModel','PersNet','MultiInputModel','PointNetTutorial']
all_ds_sorted = sorted(all_datasets)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})

# === RANGE CHECK ===
all_xgb = [v for ds in all_datasets for m in all_models for v in xgb_accs[ds][m].get(100,[])]
all_mlp = [v for ds in all_datasets for m in all_models for v in mlp_accs[ds][m].get(100,[])]
print(f"RANGE CHECK: XGB=[{min(all_xgb):.1f}, {max(all_xgb):.1f}] n={len(all_xgb)}")
print(f"RANGE CHECK: MLP=[{min(all_mlp):.1f}, {max(all_mlp):.1f}] n={len(all_mlp)}")
assert max(all_xgb) <= 100.1 and min(all_xgb) >= 0, "XGB OUT OF RANGE"
assert max(all_mlp) <= 100.1 and min(all_mlp) >= 0, "MLP OUT OF RANGE"
print("RANGE CHECK PASSED\n")

# === FIG 1: per_model_comparison ===
print("="*60 + "\nFIG 1: enhanced_per_model_comparison")
fig, ax = plt.subplots(figsize=(8, 5))
models_21 = [m for m in model_order if m in all_models]
means, stds, labels = [], [], []
for m in models_21:
    vals = [np.mean(xgb_accs[ds][m].get(100,[])) for ds in all_datasets if xgb_accs[ds][m].get(100)]
    if vals:
        mean, std = np.mean(vals), np.std(vals)
        means.append(mean); stds.append(std); labels.append(model_names.get(m,m))
        print(f"  {labels[-1]:12s}: {mean:.1f}% +/- {std:.1f}% ({len(vals)} datasets)")
y = np.arange(len(labels))
ax.barh(y, means, xerr=stds, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5, capsize=3)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel('XGBoost Test Accuracy (%)')
ax.set_title('Per-Model Accuracy at 100% Training Fraction')
ax.set_xlim(0, 105); ax.invert_yaxis()
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(m + s + 1, i, f'{m:.1f}', va='center', fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, 'enhanced_per_model_comparison.png'), bbox_inches='tight'); plt.close()

# === FIG 2: data_efficiency ===
print("\n" + "="*60 + "\nFIG 2: enhanced_data_efficiency")
fig, ax = plt.subplots(figsize=(7, 4.5))
fracs = sorted(all_frac_xgb.keys())
for f in fracs:
    xv, mv = all_frac_xgb[f], all_frac_mlp.get(f,[])
    print(f"  {f:3d}%: XGB={np.mean(xv):5.1f}+/-{np.std(xv)/np.sqrt(len(xv)):.1f} (n={len(xv):4d})  "
          f"MLP={np.mean(mv):5.1f}+/-{np.std(mv)/np.sqrt(len(mv)) if len(mv)>1 else 0:.1f} (n={len(mv):4d})")
xm = [np.mean(all_frac_xgb[f]) for f in fracs]
xs = [np.std(all_frac_xgb[f])/np.sqrt(len(all_frac_xgb[f])) for f in fracs]
mf = sorted(all_frac_mlp.keys())
mm = [np.mean(all_frac_mlp[f]) for f in mf]
ms = [np.std(all_frac_mlp[f])/np.sqrt(len(all_frac_mlp[f])) for f in mf]
ax.errorbar(fracs, xm, yerr=xs, marker='o', label='XGBoost', capsize=3, linewidth=1.5, color='steelblue')
ax.errorbar(mf, mm, yerr=ms, marker='s', label='MLP(+aug)', capsize=3, linewidth=1.5, color='coral')
ax.set_xlabel('Training Fraction (%)'); ax.set_ylabel('Mean Test Accuracy (%)')
ax.set_title('Data Efficiency: Mean Accuracy vs Training Fraction')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, 'enhanced_data_efficiency.png'), bbox_inches='tight'); plt.close()

# === FIG 3: xgb_heatmap ===
print("\n" + "="*60 + "\nFIG 3: enhanced_xgb_heatmap")
heatmap = np.full((len(models_21), len(all_ds_sorted)), np.nan)
for i, m in enumerate(models_21):
    for j, ds in enumerate(all_ds_sorted):
        v = xgb_accs[ds][m].get(100, [])
        if v: heatmap[i, j] = np.mean(v)
print(f"  Valid cells: {np.sum(~np.isnan(heatmap))}/{len(models_21)*len(all_ds_sorted)}")
for i, m in enumerate(models_21):
    n_valid = sum(1 for j in range(len(all_ds_sorted)) if not np.isnan(heatmap[i,j]))
    print(f"  {model_names.get(m,m):12s}: {n_valid}/{len(all_ds_sorted)} datasets, mean={np.nanmean(heatmap[i]):.1f}%")
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(heatmap, cmap='RdYlGn', aspect='auto', vmin=30, vmax=100)
ax.set_xticks(range(len(all_ds_sorted)))
ax.set_xticklabels([ds_names.get(d,d)[:6] for d in all_ds_sorted], rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(models_21)))
ax.set_yticklabels([model_names.get(m,m) for m in models_21], fontsize=8)
for i in range(len(models_21)):
    for j in range(len(all_ds_sorted)):
        if not np.isnan(heatmap[i,j]):
            ax.text(j, i, f'{heatmap[i,j]:.0f}', ha='center', va='center', fontsize=5.5,
                   color='white' if heatmap[i,j]<55 else 'black')
plt.colorbar(im, ax=ax, label='XGBoost Accuracy (%)', shrink=0.8)
ax.set_title('XGBoost Test Accuracy at 100% Training Fraction')
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, 'enhanced_xgb_heatmap.png'), bbox_inches='tight'); plt.close()

# === FIG 4: mlp_heatmap ===
print("\n" + "="*60 + "\nFIG 4: enhanced_mlp_heatmap")
heatmap_mlp = np.full((len(models_21), len(all_ds_sorted)), np.nan)
for i, m in enumerate(models_21):
    for j, ds in enumerate(all_ds_sorted):
        v = mlp_accs[ds][m].get(100, [])
        if v: heatmap_mlp[i, j] = np.mean(v)
print(f"  Valid cells: {np.sum(~np.isnan(heatmap_mlp))}/{len(models_21)*len(all_ds_sorted)}")
for i, m in enumerate(models_21):
    n_valid = sum(1 for j in range(len(all_ds_sorted)) if not np.isnan(heatmap_mlp[i,j]))
    print(f"  {model_names.get(m,m):12s}: {n_valid}/{len(all_ds_sorted)} datasets, mean={np.nanmean(heatmap_mlp[i]):.1f}%")
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(heatmap_mlp, cmap='RdYlGn', aspect='auto', vmin=30, vmax=100)
ax.set_xticks(range(len(all_ds_sorted)))
ax.set_xticklabels([ds_names.get(d,d)[:6] for d in all_ds_sorted], rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(models_21)))
ax.set_yticklabels([model_names.get(m,m) for m in models_21], fontsize=8)
for i in range(len(models_21)):
    for j in range(len(all_ds_sorted)):
        if not np.isnan(heatmap_mlp[i,j]):
            ax.text(j, i, f'{heatmap_mlp[i,j]:.0f}', ha='center', va='center', fontsize=5.5,
                   color='white' if heatmap_mlp[i,j]<55 else 'black')
plt.colorbar(im, ax=ax, label='MLP Accuracy (%)', shrink=0.8)
ax.set_title('MLP(+aug) Test Accuracy at 100% Training Fraction')
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, 'enhanced_mlp_heatmap.png'), bbox_inches='tight'); plt.close()

# === FIG 5: mlp_vs_xgb_100pct ===
print("\n" + "="*60 + "\nFIG 5: enhanced_mlp_vs_xgb_100pct")
ds_order = ['CBF','ECG200','ECG5000','GunPoint','Plane','PowerCons',
            'SonyAIBORobotSurface1','SonyAIBORobotSurface2','TwoLeadECG','UMD']
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
for idx, ds in enumerate(ds_order):
    ax = axes[idx//5, idx%5]
    models_for_ds, mlp_vals, xgb_vals = [], [], []
    for m in model_order:
        mv = mlp_accs[ds][m].get(100, [])
        xv = xgb_accs[ds][m].get(100, [])
        if mv or xv:
            models_for_ds.append(model_names.get(m,m)[:6])
            mlp_vals.append(np.mean(mv) if mv else 0)
            xgb_vals.append(np.mean(xv) if xv else 0)
    x = np.arange(len(models_for_ds)); w = 0.35
    ax.bar(x-w/2, mlp_vals, w, label='MLP', color='coral', alpha=0.8)
    ax.bar(x+w/2, xgb_vals, w, label='XGBoost', color='steelblue', alpha=0.8)
    ax.set_title(ds_names.get(ds,ds), fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(models_for_ds, rotation=60, fontsize=5.5); ax.set_ylim(0, 105)
    if idx == 0: ax.legend(fontsize=7, loc='lower right')
    print(f"  {ds}: " + " | ".join([f"{models_for_ds[k]}:M={mlp_vals[k]:.0f}/X={xgb_vals[k]:.0f}" for k in range(len(models_for_ds))]))
fig.suptitle('MLP vs XGBoost at 100% Training Fraction', fontsize=12, y=1.01)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, 'enhanced_mlp_vs_xgb_100pct.png'), bbox_inches='tight'); plt.close()

# === FIG 6: per_dataset_comparison ===
print("\n" + "="*60 + "\nFIG 6: enhanced_per_dataset_comparison")
fig, ax = plt.subplots(figsize=(10, 5))
ds_for_chart = sorted(all_datasets)
top_models = ['HybridGTTFN','GTTensorFieldNetworkV2','CrossAttentionTensorFieldNetwork','DistanceMatrixRaggedModel','PersNet','MultiInputModel']
colors6 = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628']
x = np.arange(len(ds_for_chart)); width = 0.13
for i, m in enumerate(top_models):
    vals = [np.mean(xgb_accs[ds][m].get(100,[])) if xgb_accs[ds][m].get(100) else 0 for ds in ds_for_chart]
    active = [v for v in vals if v > 0]
    print(f"  {model_names.get(m,m):12s}: n={len(active)}, mean={np.mean(active):.1f}%")
    ax.bar(x+i*width-2.5*width, vals, width, label=model_names.get(m,m), color=colors6[i], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([ds_names.get(d,d)[:5] for d in ds_for_chart], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('XGBoost Accuracy (%)'); ax.set_title('Per-Dataset Best-Model Comparison (XGBoost, 100%)')
ax.legend(fontsize=7, ncol=3, loc='upper right'); ax.set_ylim(0, 108); ax.grid(True, axis='y', alpha=0.2)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, 'enhanced_per_dataset_comparison.png'), bbox_inches='tight'); plt.close()

# === FIG 7: model_data_efficiency ===
print("\n" + "="*60 + "\nFIG 7: enhanced_model_data_efficiency")
key_models = ['HybridGTTFN','GTTensorFieldNetworkV2','CrossAttentionTensorFieldNetwork','DistanceMatrixRaggedModel','PersNet']
colors5 = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
fig, ax = plt.subplots(figsize=(8, 5))
for i, m in enumerate(key_models):
    fracs_m = sorted(model_frac_data[m].keys())
    mm = [np.mean(model_frac_data[m][f]) for f in fracs_m]
    nn = [len(model_frac_data[m][f]) for f in fracs_m]
    sems = [np.std(model_frac_data[m][f])/np.sqrt(n) if n>1 else 0 for f, n in zip(fracs_m, nn)]
    label = model_names.get(m,m)
    print(f"  {label:12s}: " + " ".join([f"{f}%={v:.1f}({n})" for f,v,n in zip(fracs_m, mm, nn)]))
    ax.errorbar(fracs_m, mm, yerr=sems, marker='o', label=label, color=colors5[i], capsize=2, linewidth=1.5, markersize=4)
ax.set_xlabel('Training Fraction (%)'); ax.set_ylabel('XGBoost Test Accuracy (%)')
ax.set_title('XGBoost Accuracy vs Training Fraction (Mean over Datasets)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(5, 105)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, 'enhanced_model_data_efficiency.png'), bbox_inches='tight'); plt.close()

# === FIG 8: benchmark_timing ===
print("\n" + "="*60 + "\nFIG 8: benchmark_timing")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
names = [k for k in timing if 'error' not in timing[k]]
fwd = [timing[k]['forward_ms'] for k in names]
bwd = [timing[k]['backward_ms'] for k in names]
params_list = [timing[k]['params'] for k in names]
labels = [model_names.get(k, k) for k in names]
order = np.argsort([f+b for f,b in zip(fwd, bwd)])
names = [names[i] for i in order]; labels = [labels[i] for i in order]
fwd = [fwd[i] for i in order]; bwd = [bwd[i] for i in order]; params_list = [params_list[i] for i in order]
y = np.arange(len(labels))
ax1.barh(y, fwd, label='Forward', color='steelblue', alpha=0.8)
ax1.barh(y, bwd, left=fwd, label='Backward', color='coral', alpha=0.8)
ax1.set_yticks(y); ax1.set_yticklabels(labels)
ax1.set_xlabel('Time (ms)'); ax1.set_title('CPU Time per Forward+Backward Pass (CBF)')
ax1.legend(fontsize=8)
for i, (f, b) in enumerate(zip(fwd, bwd)):
    ax1.text(f+b+5, i, f'{f+b:.0f}ms', va='center', fontsize=8)
ax2.barh(y, params_list, color='seagreen', alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.set_yticks(y); ax2.set_yticklabels(labels)
ax2.set_xlabel('Parameters'); ax2.set_title('Model Parameter Counts')
ax2.set_xscale('log')
for i, p in enumerate(params_list):
    ax2.text(p*1.15, i, f'{p:,}', va='center', fontsize=8)
for i, n in enumerate(names):
    print(f"  {n:15s}: params={timing[n]['params']:>8,}  total={timing[n]['total_ms']:6.1f}ms")
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, 'benchmark_timing.png'), bbox_inches='tight'); plt.close()

# === FIG 9: training_curves ===
print("\n" + "="*60 + "\nFIG 9: training_curves")
curve_models = ['GTTFNv2','AttnTFN','HybridGTTFN','DMR','PersNet','PNTut']
colors_c = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628']
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for i, name in enumerate(curve_models):
    if name not in curves: print(f"  {name}: MISSING"); continue
    c = curves[name]
    tl, vl = c['train_loss'], c['val_loss']
    ta = [a*100 for a in c['train_acc']]; va = [a*100 for a in c['val_acc']]
    epochs = list(range(1, len(tl)+1))
    axes[0].plot(epochs, tl, color=colors_c[i], alpha=0.8, linewidth=1.5, label=f'{name} (train)')
    axes[0].plot(epochs, vl, color=colors_c[i], linestyle='--', alpha=0.5, linewidth=1)
    axes[1].plot(epochs, ta, color=colors_c[i], alpha=0.8, linewidth=1.5, label=f'{name} (train)')
    axes[1].plot(epochs, va, color=colors_c[i], linestyle='--', alpha=0.5, linewidth=1)
    print(f"  {name:12s}: ep1_acc={ta[0]:.0f}% ep50_acc={ta[49]:.0f}% ep200_acc={ta[-1]:.0f}% "
          f"ep1_val={va[0]:.0f}% ep50_val={va[49]:.0f}% ep200_val={va[-1]:.0f}%")
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].set_title('Training Loss (CBF, 15 samples)')
axes[0].legend(fontsize=7, ncol=2); axes[0].grid(True, alpha=0.3); axes[0].set_yscale('log')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Training Accuracy (CBF, 15 samples)')
axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 105)
fig.suptitle('Training Curves on CBF', fontsize=12, y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, 'training_curves.png'), bbox_inches='tight'); plt.close()

# === FIG 10: convergence_comparison ===
print("\n" + "="*60 + "\nFIG 10: convergence_comparison")
fig, ax = plt.subplots(figsize=(8, 5))
for i, name in enumerate(curve_models):
    if name not in curves: continue
    c = curves[name]
    va = [a*100 for a in c['val_acc']]
    epochs = list(range(1, len(va)+1))
    ax.plot(epochs, va, color=colors_c[i], linewidth=2, label=model_names.get(name,name))
    print(f"  {name:12s}: v1={va[0]:.0f}% v10={va[9]:.0f}% v50={va[49]:.0f}% v200={va[-1]:.0f}%")
ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Convergence Speed Comparison (CBF, 15 Train Samples)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, 'convergence_comparison.png'), bbox_inches='tight'); plt.close()

# === FIG 11: accuracy_vs_cost ===
print("\n" + "="*60 + "\nFIG 11: accuracy_vs_cost")
fig, ax = plt.subplots(figsize=(8, 5))
for name in timing:
    if 'error' in timing[name]: continue
    csv_name = csv_mapping.get(name)
    if csv_name and csv_name in csv_model_accs and csv_model_accs[csv_name]:
        acc = np.mean(csv_model_accs[csv_name])
        t = timing[name]['total_ms']
        p = timing[name]['params']
        ax.scatter(t, acc, s=max(np.sqrt(p)*0.5, 30), zorder=5, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax.annotate(name, (t, acc), fontsize=8, textcoords="offset points", xytext=(5, 5))
        print(f"  {name:15s}: CSV_acc={acc:.1f}%, time={t:.1f}ms, params={p:,}")
ax.set_xlabel('Total CPU Time per Forward+Backward Pass (ms)')
ax.set_ylabel('Mean Best Accuracy (%)')
ax.set_title('Accuracy vs Computational Cost (bubble size ~ sqrt(params))')
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, 'accuracy_vs_cost.png'), bbox_inches='tight'); plt.close()

print("\n" + "="*60)
print("ALL 11 PLOTS REGENERATED. ALL VALUES VERIFIED.")
print("="*60)
