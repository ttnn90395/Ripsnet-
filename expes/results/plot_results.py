import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

DATASETS = [
    "CBF", "ChlorineConcentration", "DistalPhalanxOutlineCorrect", "ECG200", "ECG5000",
    "GunPoint", "ItalyPowerDemand", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "PhalangesOutlinesCorrect", "Plane",
    "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxTW", "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2", "TwoLeadECG", "GunPointOldVersusYoung", "PowerCons", "UMD",
]

df = pd.read_csv("ucr_results.csv", index_col=0)
out_dir = Path("task_plots")
out_dir.mkdir(exist_ok=True)

bl_names = ["Gudhi (clean)", "Gudhi (noise)", "DTW k-NN", "Euclidean k-NN"]
bl_df = df.loc[[n for n in bl_names if n in df.index]]
nn_df = df.drop(index=[n for n in bl_names if n in df.index], errors="ignore")

nn_df = nn_df.map(lambda x: float(x.strip("%")) if isinstance(x, str) and "%" in x else None)
bl_df = bl_df.map(lambda x: float(x.strip("%")) if isinstance(x, str) and "%" in x else None)

task_cols = [c for c in df.columns if c.startswith("task_")]

for col in task_cols:
    vals = nn_df[col].dropna().sort_values()
    baseline_vals = bl_df[col].dropna()

    colors = ["#2196F3" if "GS" in n else "#64B5F6" for n in vals.index]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = range(len(vals))
    bars = ax.barh(list(vals.index), list(vals.values), color=colors, edgecolor="white", linewidth=0.5)

    for bname, bval in baseline_vals.items():
        short = bname.replace(" (clean)", "").replace(" (noise)", "").replace(" k-NN", "kNN")
        ax.axvline(x=bval, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(bval + 0.3, len(vals) - 1, f"{short}={bval:.1f}%", fontsize=7,
                color="gray", va="bottom")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Test Accuracy (%)")
    tshort = col.replace("task_", "")
    dataset = DATASETS[int(tshort)] if tshort.isdigit() and int(tshort) < len(DATASETS) else tshort
    ax.set_title(f"{dataset} — NN Model XGB Test Accuracies")
    ax.tick_params(axis="y", labelsize=6)
    for v, bar in zip(vals.values, bars):
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=5, color="#333")

    fig.tight_layout()
    fig.savefig(out_dir / f"task_{tshort}_{dataset}_barplot.png", dpi=150)
    plt.close(fig)

print(f"Saved {len(task_cols)} plots to {out_dir}/")
