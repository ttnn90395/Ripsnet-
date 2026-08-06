import os, json
results_dir = 'results'
for model in ['HierarchicalGTTFN', 'HierarchicalTensorFieldNetwork']:
    datasets = set()
    for f in os.listdir(results_dir):
        if f.startswith('ablation_train_') and model in f:
            try:
                d = json.load(open(os.path.join(results_dir, f)))
                datasets.add(d['dataset'])
            except: pass
    n = len([f for f in os.listdir(results_dir) if f.startswith('ablation_train_') and model in f])
    print(f"{model}: {n} JSONs, datasets: {sorted(datasets)}")
