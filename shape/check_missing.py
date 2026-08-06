#!/usr/bin/env python3
import os, json, glob

datasets = ['circles', 'circles_noisy', 'shapes3d_topology', 'shapes3d_geometry', 'shapes3d_complex', 'shapes3d_8way']
models = ['PointNet3D', 'RipsPointNet', 'ScalarDistanceDeepSet',
          'TensorFieldNetwork', 'HierarchicalTensorFieldNetwork', 'StochasticTensorFieldNetwork',
          'OnEquivariantTensorFieldNetwork', 'AttentionTensorFieldNetwork',
          'RelaxedOnEquivariantTensorFieldNetwork', 'HybridOnEquivariantTensorFieldNetwork']
trials = [0, 1, 2]

results_dir = 'results'
total = len(datasets) * len(models) * len(trials)
found = 0
missing = {}
for ds in datasets:
    ds_missing = []
    for m in models:
        for t in trials:
            jname = f'shape_{ds}_{m}_t{t}'
            if os.path.exists(os.path.join(results_dir, f'{jname}.json')):
                found += 1
            else:
                ds_missing.append(f'{m}_t{t}')
    if ds_missing:
        missing[ds] = ds_missing

print(f'{found}/{total} results complete ({100*found//total}%)')
for ds, m_list in missing.items():
    print(f'  {ds}: {len(m_list)} missing -> {m_list}')
