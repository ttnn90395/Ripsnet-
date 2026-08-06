import os, json
r = "results"
models = ["TensorFieldNetwork", "GTTensorFieldNetwork", "GTTensorFieldNetworkV2",
          "HierarchicalGTTFN", "HierarchicalTensorFieldNetwork",
          "AttentionTensorFieldNetwork", "StochasticTensorFieldNetwork"]
for m in models:
    ds = set()
    for f in os.listdir(r):
        if f.startswith("ablation_train_") and ("_" + m + "_") in f:
            try: ds.add(json.load(open(os.path.join(r,f)))["dataset"])
            except: pass
    print(f"{m}: {len(ds)} datasets - {sorted(ds)}")
