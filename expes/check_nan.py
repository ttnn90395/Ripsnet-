import os, json
r = "results"
targets = ["TensorFieldNetwork", "CrossAttentionTensorFieldNetwork", 
           "GTTensorFieldNetwork", "AttentionTensorFieldNetwork",
           "StochasticTensorFieldNetwork", "HierarchicalGTTFN",
           "HierarchicalTensorFieldNetwork", "GTTensorFieldNetworkV2"]
failing_ds = ["ItalyPowerDemand", "ChlorineConcentration", "DistalPhalanxOutlineCorrect",
              "GunPointOldVersusYoung", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
              "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "PhalangesOutlinesCorrect",
              "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxTW"]
for model in targets:
    count = 0
    nan_count = 0
    for ds in failing_ds:
        for f in os.listdir(r):
            if f.startswith(f"ablation_train_{ds}_{model}_"):
                count += 1
                d = json.load(open(os.path.join(r, f)))
                if d.get("xgb_test_acc") is None or (isinstance(d.get("xgb_test_acc"), float) and d["xgb_test_acc"] != d["xgb_test_acc"]):
                    nan_count += 1
    print(f"{model}: {count} total, {nan_count} NaN (of 264)")
