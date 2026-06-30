# TFN Model Improvements

## Goal
Make TFN models learn more effectively on UCR time-series data by:
1. Removing rotational equivariance (`max_order=0`)
2. Scaling model capacity to dataset size
3. Replacing max-pooling with sum-pooling
4. Simplifying the classifier head

---

## Step 1 — `gt_tfn_layer.py` (base TFN model)

### 1a. Remove `use_dist_features` (revert)
- **Line 579**: Delete `use_dist_features: bool = True,` line
- **Lines ~598-600**: Change back to:
  ```python
  init_scalar_c = 1 + node_attr_dim
  self.node_attr_dim = node_attr_dim
  ```
- **Lines ~733-737** (`_encode_single` initial features): Remove the `if self.use_dist_features` block, restore to:
  ```python
  f0_parts = [pos.norm(dim=-1, keepdim=True)]
  if node_attr is not None and self.node_attr_dim > 0:
      f0_parts.append(self.attr_proj(node_attr))
  ```
- **Lines ~670-676** (`_encode_batch` initial features): Same revert, restore original `f0_parts` block (only norm + node_attr)

### 1b. Sum-pooling
- **Line 711**: `node_inv.max(dim=1).values` → `node_inv.sum(dim=1)`
- **Line 782**: `node_inv.max(dim=0).values` → `node_inv.sum(dim=0)`

### 1c. Simplify rho head
- **Lines 628-633**: Replace with:
  ```python
  self.rho = nn.Sequential(
      nn.Linear(inv_dim, 64), nn.SiLU(), nn.LayerNorm(64),
      nn.Linear(64, num_classes),
  )
  ```

---

## Step 2 — `gt_improvements.py`

### 2a. Revert `StochasticEquivariantTFN` distance features
- **Line ~1208**: Remove `use_dist_features: bool = True,` from `__init__`
- **Lines ~1229-1232**: Change back to:
  ```python
  init_scalar_c = 1 + node_attr_dim
  self.node_attr_dim = node_attr_dim
  ```
- **Lines ~1282-1284** (`_encode_features`): Remove RBF-mean injection, restore to:
  ```python
  f0_parts = [pos.norm(dim=-1, keepdim=True)]
  if node_attr is not None and self.node_attr_dim > 0:
      f0_parts.append(self.attr_proj(node_attr))
  ```

### 2b. Sum-pooling (5 locations)
Find each `node_inv.max(dim=...).values` and change to `node_inv.sum(dim=...)`:
- `HierarchicalGTTFN._encode_single` ~line 441
- `GTTensorFieldNetworkWithAttention._encode_batch` ~line 906
- `GTTensorFieldNetworkWithAttention._encode_single` ~line 960
- `StochasticEquivariantTFN._encode_features` ~line 1319
- `EquivariantGraphMambaNetwork._encode_single` ~line 1654

### 2c. Simplify rho heads (4 classes)
Find each `self.rho = nn.Sequential(...)` in `__init__` and replace with two-layer pattern:
```python
self.rho = nn.Sequential(
    nn.Linear(inv_dim, 64), nn.SiLU(), nn.LayerNorm(64),
    nn.Linear(64, num_classes),
)
```
Affected classes:
- `HierarchicalGTTFN.__init__` (~line 410 area)
- `GTTensorFieldNetworkWithAttention.__init__` (~line 870 area)
- `EquivariantGraphMambaNetwork.__init__` (~line 1600 area)
- `StochasticEquivariantTFN.__init__` — skip, it uses `_StochasticMixtureHead` which already has the right pattern

---

## Step 3 — `TFN.py`

### 3a. Sum-pooling
- **Line 247**: `torch.cat([f0, f1.norm(dim=-1)], dim=-1).max(dim=0).values` → `torch.cat([f0, f1.norm(dim=-1)], dim=-1).sum(dim=0)`

### 3b. Simplify rho
Find the `self.rho` construction and replace with the two-layer pattern.

---

## Step 4 — `expes/train_nn.py`

### 4a. max_order defaults
In `build_model_by_name`, for all 9 TFN model branches:
```
hp.get('max_order', 1)  →  hp.get('max_order', 0)
```
Models: `TensorFieldNetwork`, `GTTensorFieldNetwork`, `GTTensorFieldNetworkV2`, `HierarchicalGTTFN`, `HierarchicalTensorFieldNetwork`, `OnEquivariantTensorFieldNetwork`, `AttentionTensorFieldNetwork`, `StochasticTensorFieldNetwork`

### 4b. Adaptive scaling
After `_npts = data_train_torch[0].shape[0]` (~line 432), add:
```python
if _npts < 50:
    _hc, _nl, _cd = 4, 1, [8]
elif _npts < 150:
    _hc, _nl, _cd = 8, 1, [16]
elif _npts < 300:
    _hc, _nl, _cd = 16, 2, [32, 16]
else:
    _hc, _nl, _cd = 32, 3, [64, 32]
```

Then in each TFN branch replace:
- `hp.get('hidden_channels', 32)` → `hp.get('hidden_channels', _hc)`
- `hp.get('num_layers', 3)` → `hp.get('num_layers', _nl)` (for models that have num_layers)
- `hp.get('classifier_dims', [64, 32])` → `hp.get('classifier_dims', _cd)`

Note: `HierarchicalGTTFN` and `HierarchicalTensorFieldNetwork` don't have `num_layers`; they use `stage_sizes`. Leave those as-is.

---

## Step 5 — `expes/analysis_nn.py`

### 5a. max_order defaults
Same as step 4a — find the equivalent model construction sections and change `max_order` default from `1` to `0`.

### 5b. Adaptive scaling
Same as step 4b — add the adaptive heuristic and use it as defaults.

### 5c. `_infer_tfn_architecture` `.isdecimal()` guard (line 373)
Add `.isdecimal()` check before `int()` conversion so non-numeric `rho` submodule keys (`logit_net`, `mu_net`, `encoder_mlp`) from `_StochasticMixtureHead` are skipped instead of crashing with `ValueError: invalid literal for int()`.

---

## Step 6 — `models.py`

Check if `GTTensorFieldNetworkV2` (line ~248), `TensorFieldNetwork` (line ~298), `AttentionTensorFieldNetwork` (line ~869), `StochasticTensorFieldNetwork` (line ~920) constructors pass `use_dist_features` or need any other changes. Likely no changes needed since they delegate to base classes.

---

## Step 7 — Verify & Deploy

```bash
python3 -m py_compile gt_tfn_layer.py gt_improvements.py TFN.py expes/train_nn.py expes/analysis_nn.py models.py
git add -A && git commit -m "refactor: max_order=0, sum-pooling, simpelr rho, adaptive scaling for TFN models"
git push
ssh rcc-cloud "cd /hs/work0/home/users/\$USER/exp/ripsnet/Ripsnet- && bash expes/env/auto_deploy.sh --force"
```
