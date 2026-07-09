"""
Aggregate and plot ablation experiment results.

Usage:
    python plot_ablations.py <results_dir>
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/ablations'
os.makedirs('ablation_plots', exist_ok=True)

# ─── Experiment 1: Training fraction ─────────────────────────────────────────
train_files = glob.glob(os.path.join(results_dir, 'ablation_train_*.json'))
train_data = []
for f in train_files:
    d = json.load(open(f))
    train_data.append(d)

if train_data:
    df_train = pd.DataFrame(train_data)
    print(f"Training ablation: {len(df_train)} results")

    datasets = sorted(df_train['dataset'].unique())
    models = sorted(df_train['model'].unique())
    fractions = sorted(df_train['fraction_pct'].unique())

    for ds in datasets:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in models:
            sub = df_train[(df_train['dataset'] == ds) & (df_train['model'] == model)]
            sub = sub.groupby('fraction_pct')['xgb_test_acc'].agg(['mean', 'std']).reset_index()
            sub = sub.sort_values('fraction_pct')
            ax.errorbar(sub['fraction_pct'], sub['mean'] * 100,
                        yerr=sub['std'] * 100, marker='o', label=model, capsize=4)
        ax.set_xlabel('Training data fraction (%)')
        ax.set_ylabel('Test accuracy (%)')
        ax.set_title(f'{ds} — Training fraction ablation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'ablation_plots/train_frac_{ds}.png', dpi=150)
        plt.close(fig)
        print(f"  Saved ablation_plots/train_frac_{ds}.png")

    # Summary: avg accuracy across datasets per fraction per model
    print("\n=== Training fraction: Mean test accuracy across all datasets ===")
    pivot = df_train.groupby(['model', 'fraction_pct'])['xgb_test_acc'].mean()
    for frac in fractions:
        line = f"  {frac:3d}%: "
        for model in models:
            val = pivot.loc[(model, frac)] * 100
            line += f"  {model:40s} {val:5.1f}%"
        print(line)

    # Overall summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for mi, model in enumerate(models):
        sub = df_train[df_train['model'] == model]
        sub = sub.groupby('fraction_pct')['xgb_test_acc'].agg(['mean', 'std']).reset_index()
        sub = sub.sort_values('fraction_pct')
        ax.errorbar(sub['fraction_pct'], sub['mean'] * 100,
                    yerr=sub['std'] * 100, marker='o', label=model,
                    color=colors[mi], capsize=4)
    ax.set_xlabel('Training data fraction (%)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('Training data fraction ablation (averaged across datasets)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'ablation_plots/train_frac_overall.png', dpi=150)
    plt.close(fig)
    print(f"  Saved ablation_plots/train_frac_overall.png")
else:
    print("No training ablation results found.")

# ─── Experiment 2: Density ablation ──────────────────────────────────────────
density_files = glob.glob(os.path.join(results_dir, 'ablation_density_*.json'))
density_data = []
for f in density_files:
    d = json.load(open(f))
    density_data.append(d)

if density_data:
    df_den = pd.DataFrame(density_data)
    print(f"\nDensity ablation: {len(df_den)} results")

    datasets = sorted(df_den['dataset'].unique())
    models = sorted(df_den['model_label'].unique())
    fractions = sorted(df_den['fraction_pct'].unique())

    for ds in datasets:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_label in models:
            sub = df_den[(df_den['dataset'] == ds) & (df_den['model_label'] == model_label)]
            sub = sub.groupby('fraction_pct')['xgb_test_acc'].agg(['mean', 'std']).reset_index()
            sub = sub.sort_values('fraction_pct')
            ax.errorbar(sub['fraction_pct'], sub['mean'] * 100,
                        yerr=sub['std'] * 100, marker='o', label=model_label, capsize=4)
        ax.set_xlabel('Point cloud density (%)')
        ax.set_ylabel('Test accuracy (%)')
        ax.set_title(f'{ds} — Point density ablation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'ablation_plots/density_{ds}.png', dpi=150)
        plt.close(fig)
        print(f"  Saved ablation_plots/density_{ds}.png")

    # Overall summary
    print("\n=== Density ablation: Mean test accuracy across all datasets ===")
    pivot = df_den.groupby(['model_label', 'fraction_pct'])['xgb_test_acc'].mean()
    for frac in fractions:
        line = f"  {frac:3d}%: "
        for model_label in models:
            val = pivot.loc[(model_label, frac)] * 100
            line += f"  {model_label:40s} {val:5.1f}%"
        print(line)

    # Overall summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for mi, model_label in enumerate(models):
        sub = df_den[df_den['model_label'] == model_label]
        sub = sub.groupby('fraction_pct')['xgb_test_acc'].agg(['mean', 'std']).reset_index()
        sub = sub.sort_values('fraction_pct')
        ax.errorbar(sub['fraction_pct'], sub['mean'] * 100,
                    yerr=sub['std'] * 100, marker='o', label=model_label,
                    color=colors[mi % len(colors)], capsize=4)
    ax.set_xlabel('Point cloud density (%)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('Point density ablation (averaged across datasets)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'ablation_plots/density_overall.png', dpi=150)
    plt.close(fig)
    print(f"  Saved ablation_plots/density_overall.png")

    # Per-model-type comparison (raw vs GS)
    print("\n=== Raw vs GS comparison ===")
    df_den['base_model'] = df_den['model_label'].str.replace('_GS', '')
    df_den['has_gs'] = df_den['model_label'].str.endswith('_GS')
    for model_base in df_den['base_model'].unique():
        sub = df_den[df_den['base_model'] == model_base]
        sub = sub.groupby(['has_gs', 'fraction_pct'])['xgb_test_acc'].mean()
        print(f"\n  {model_base}:")
        for frac in fractions:
            raw = sub.loc[(False, frac)] * 100 if (False, frac) in sub else float('nan')
            gs = sub.loc[(True, frac)] * 100 if (True, frac) in sub else float('nan')
            print(f"    {frac:3d}%:  raw={raw:5.1f}%  GS={gs:5.1f}%")
else:
    print("No density ablation results found.")

print("\nDone.")
