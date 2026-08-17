import json, os, math

RESULTS_DIR = "/Users/statgen/Ripsnet-/expes/results/enhanced"

ALL_DATASETS = [
    "CBF", "ChlorineConcentration", "DistalPhalanxOutlineCorrect", "ECG200", "ECG5000",
    "GunPoint", "ItalyPowerDemand", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "PhalangesOutlinesCorrect", "Plane",
    "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxTW", "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2", "TwoLeadECG", "GunPointOldVersusYoung", "PowerCons", "UMD",
]

MODELS = ["GTTensorFieldNetworkV2", "CrossAttentionTensorFieldNetwork", "HybridGTTFN"]

def nanmean(vals):
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    if not valid:
        return None
    return sum(valid) / len(valid)

# Collect all results for CSV update
all_results = {}
for model in MODELS:
    all_results[model] = {}
    print(f"\n{'='*80}")
    print(f"  {model}")
    print(f"{'='*80}")
    
    for ds in ALL_DATASETS:
        results = []
        for frac in [10, 20, 30, 50, 70, 100]:
            for trial in [0, 1, 2, 3]:
                for clf in ("xgboost", "mlp"):
                    for ms in (False, True):
                        if clf == "xgboost":
                            tag = "_ms" if ms else ""
                        else:
                            tag = "_aug_ms" if ms else "_aug"
                        fname = f"train_{ds}_{model}_{frac}pct_t{trial}_{clf}{tag}.json"
                        fpath = os.path.join(RESULTS_DIR, fname)
                        if os.path.exists(fpath):
                            try:
                                with open(fpath) as f:
                                    data = json.load(f)
                                acc_key = "mlp_test_acc" if clf == "mlp" else "xgb_test_acc"
                                val = data.get(acc_key)
                                results.append(val)
                            except:
                                pass
        
        mean_acc = nanmean(results)
        all_results[model][ds] = mean_acc
        if mean_acc is not None:
            print(f"  {ds:40s}: {mean_acc:6.2f}%  (n={len([v for v in results if v is not None and not math.isnan(v)])}/{len(results)})")
        else:
            print(f"  {ds:40s}: NO DATA")
