#!/usr/bin/env python3
from datasets.shapes3d import get_3d_shape_dataset
for name in ['shapes3d_topology', 'shapes3d_geometry', 'shapes3d_complex', 'shapes3d_8way']:
    ds = get_3d_shape_dataset(name, split='train')
    print(f'{name}: {len(ds)} samples')
    sample = ds[0]
    if hasattr(sample, 'shape'):
        print(f'  shape: {sample.shape}')
    elif hasattr(sample, 'x'):
        print(f'  x: {sample.x.shape}')
