import pandas as pd
import numpy as np

DATASETS = [
    "CBF", "ChlorineConcentration", "DistalPhalanxOutlineCorrect", "ECG200", "ECG5000",
    "GunPoint", "ItalyPowerDemand", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "PhalangesOutlinesCorrect", "Plane",
    "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxTW", "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2", "TwoLeadECG", "GunPointOldVersusYoung", "PowerCons", "UMD",
]

# Dataset metadata (from DataSummary.csv for our 21 datasets)
DS_META = {
    "CBF": dict(type="Simulated", train=30, test=900, classes=3, length=128, ed=14.78, dtw=0.44),
    "ChlorineConcentration": dict(type="Sensor", train=467, test=3840, classes=3, length=166, ed=35.00, dtw=35.00),
    "DistalPhalanxOutlineCorrect": dict(type="Image", train=600, test=276, classes=2, length=80, ed=28.26, dtw=27.54),
    "ECG200": dict(type="ECG", train=100, test=100, classes=2, length=96, ed=12.00, dtw=12.00),
    "ECG5000": dict(type="ECG", train=500, test=4500, classes=5, length=140, ed=7.51, dtw=7.49),
    "GunPoint": dict(type="Motion", train=50, test=150, classes=2, length=150, ed=8.67, dtw=8.67),
    "ItalyPowerDemand": dict(type="Sensor", train=67, test=1029, classes=2, length=24, ed=4.47, dtw=4.47),
    "MedicalImages": dict(type="Image", train=381, test=760, classes=10, length=99, ed=31.58, dtw=25.26),
    "MiddlePhalanxOutlineAgeGroup": dict(type="Image", train=400, test=154, classes=3, length=80, ed=48.05, dtw=48.05),
    "MiddlePhalanxOutlineCorrect": dict(type="Image", train=600, test=291, classes=2, length=80, ed=23.37, dtw=23.37),
    "MiddlePhalanxTW": dict(type="Image", train=399, test=154, classes=6, length=80, ed=48.70, dtw=49.35),
    "PhalangesOutlinesCorrect": dict(type="Image", train=1800, test=858, classes=2, length=80, ed=23.89, dtw=23.89),
    "Plane": dict(type="Sensor", train=105, test=105, classes=7, length=144, ed=3.81, dtw=0.00),
    "ProximalPhalanxOutlineAgeGroup": dict(type="Image", train=400, test=205, classes=3, length=80, ed=21.46, dtw=21.46),
    "ProximalPhalanxTW": dict(type="Image", train=400, test=205, classes=6, length=80, ed=29.27, dtw=24.39),
    "SonyAIBORobotSurface1": dict(type="Sensor", train=20, test=601, classes=2, length=70, ed=30.45, dtw=30.45),
    "SonyAIBORobotSurface2": dict(type="Sensor", train=27, test=953, classes=2, length=65, ed=14.06, dtw=14.06),
    "TwoLeadECG": dict(type="ECG", train=23, test=1139, classes=2, length=82, ed=25.29, dtw=13.17),
    "GunPointOldVersusYoung": dict(type="Motion", train=136, test=315, classes=2, length=150, ed=4.76, dtw=3.49),
    "PowerCons": dict(type="Power", train=180, test=180, classes=2, length=144, ed=6.67, dtw=7.78),
    "UMD": dict(type="Simulated", train=36, test=144, classes=3, length=150, ed=23.61, dtw=2.78),
}

tfns = [
    "TensorFieldNetwork", "GTTensorFieldNetwork", "GTTensorFieldNetworkV2",
    "HierarchicalGTTFN", "HierarchicalTensorFieldNetwork",
    "OnEquivariantTensorFieldNetwork", "AttentionTensorFieldNetwork",
    "StochasticTensorFieldNetwork", "CrossAttentionTensorFieldNetwork",
    "RelaxedOnEquivariantTensorFieldNetwork", "HybridOnEquivariantTensorFieldNetwork",
]
tfns_gs = [m + "_GS" for m in tfns]
dmr = "DistanceMatrixRaggedModel"
dmr_gs = dmr + "_GS"
pointnets = ["PointNet3D", "PointNetTutorial"]
pointnets_gs = [m + "_GS" for m in pointnets]

df = pd.read_csv("ucr_results.csv", index_col=0)

def parse(x):
    if isinstance(x, str) and "%" in x:
        return float(x.strip("%"))
    return None

dfp = df.map(parse)

task_cols = [c for c in df.columns if c.startswith("task_")]

print("=" * 120)
print("MODEL GROUP PERFORMANCE ACROSS 21 UCR DATASETS")
print("=" * 120)

for group_name, models in [
    ("TFN (raw)", tfns),
    ("TFN (GS)", tfns_gs),
    ("DistanceMatrixRaggedModel", [dmr, dmr_gs]),
    ("PointNet (raw)", pointnets),
    ("PointNet (GS)", pointnets_gs),
]:
    avail = [m for m in models if m in dfp.index]
    if not avail:
        continue
    group_df = dfp.loc[avail]
    mean_acc = group_df.mean(skipna=True)
    print(f"\n--- {group_name} ---")
    print(f"  Mean accuracy across all datasets: {mean_acc.mean():.2f}%")

# For each dataset, find which group wins
print("\n" + "=" * 120)
print("BEST GROUP PER DATASET")
print("=" * 120)

# Compare groups using best non-SKIP model per group
groups = {
    "TFN": [m for m in tfns if m not in ("GTTensorFieldNetworkV2", "HierarchicalTensorFieldNetwork")] + tfns_gs,
    "DMR": [dmr, dmr_gs],
    "PointNet": pointnets + pointnets_gs,
}

for idx, col in enumerate(task_cols):
    ds_name = DATASETS[idx]
    meta = DS_META[ds_name]
    best_group = None
    best_score = -1
    results = {}
    for gname, models in groups.items():
        avail = [m for m in models if m in dfp.index]
        vals = dfp.loc[avail, col].dropna()
        if len(vals) > 0:
            best = vals.max()
            results[gname] = best
            if best > best_score:
                best_score = best
                best_group = gname
    if best_group:
        line = f"  {ds_name:35s}  type={meta['type']:10s}  train={meta['train']:5d}  classes={meta['classes']:2d}  len={meta['length']:5d}"
        line += f"  BEST={best_group:5s}  ({best_score:.1f}%)"
        for g, s in sorted(results.items()):
            line += f"  {g}={s:.1f}"
        print(line)

print("\n" + "=" * 120)
print("WHICH DATASET TYPES FAVOR EACH GROUP")
print("=" * 120)

for gname in groups:
    wins = []
    for idx, col in enumerate(task_cols):
        ds_name = DATASETS[idx]
        meta = DS_META[ds_name]
        avail = [m for m in groups[gname] if m in dfp.index]
        vals = dfp.loc[avail, col].dropna()
        if len(vals) == 0:
            continue
        best_this = vals.max()
        other_best = -1
        for og, om in groups.items():
            if og == gname:
                continue
            o_avail = [m for m in om if m in dfp.index]
            o_vals = dfp.loc[o_avail, col].dropna()
            if len(o_vals) > 0:
                other_best = max(other_best, o_vals.max())
        if best_this >= other_best and best_this > 0:
            wins.append(ds_name)
    print(f"\n{gname} wins on {len(wins)}/{len(task_cols)} datasets:")
    for ds in wins:
        meta = DS_META[ds]
        print(f"  {ds:35s}  type={meta['type']:10s}  classes={meta['classes']:2d}  train={meta['train']:5d}  len={meta['length']:5d}")

print("\n" + "=" * 120)
print("AVG ACCURACY BY DATASET TYPE")
print("=" * 120)
ds_types = {}
for idx, col in enumerate(task_cols):
    ds_name = DATASETS[idx]
    meta = DS_META[ds_name]
    t = meta["type"]
    if t not in ds_types:
        ds_types[t] = {"ds_names": [], "count": 0}
    ds_types[t]["ds_names"].append(ds_name)
    ds_types[t]["count"] += 1

for t, info in sorted(ds_types.items()):
    names = info["ds_names"]
    print(f"\n  {t:15s} ({info['count']} datasets)")
    for gname in groups:
        scores = []
        for ds_name in names:
            idx = DATASETS.index(ds_name)
            col = task_cols[idx]
            avail = [m for m in groups[gname] if m in dfp.index]
            vals = dfp.loc[avail, col].dropna()
            if len(vals) > 0:
                scores.append(vals.max())
        if scores:
            print(f"    {gname:10s}: avg best = {np.mean(scores):.1f}%")
