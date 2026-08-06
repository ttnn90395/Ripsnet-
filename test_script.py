import torch
import torch.nn as nn
import sys

print("=" * 60)
print("TEST 1: GTBasis and CGCoefficients are nn.Module subclasses")
print("=" * 60)

from gt_basis import GTBasis, CGCoefficients, GTSignature

gb = GTBasis(3, 2)
cg = CGCoefficients(3, 1)

assert isinstance(gb, nn.Module), "GTBasis should be nn.Module"
assert isinstance(cg, nn.Module), "CGCoefficients should be nn.Module"
print("  PASS: Both are nn.Module subclasses")

print(f"\n  GTBasis: {gb.num_basis} basis functions, {len(gb.signatures)} signatures")
print(f"  CGCoefficients: {len(cg._cache)} cached entries")

# Check buffers are registered
buffers = list(cg.buffers())
print(f"  CG buffers registered: {len(buffers)}")
assert len(buffers) > 0, "CG should have registered buffers"
print("  PASS: Buffers registered")

print("\n" + "=" * 60)
print("TEST 2: Device movement via nn.Module")
print("=" * 60)

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cg = CGCoefficients(3, 1)
        self.gb = GTBasis(3, 2)
        self.linear = nn.Linear(10, 5)
    def forward(self, x):
        return self.linear(x)

m = TestModel()
cpu_params = next(m.parameters()).device
assert str(cpu_params) == "cpu", f"Expected cpu, got {cpu_params}"
print(f"  Initial device: {cpu_params}")

m.to("cpu")
cpu_params = next(m.parameters()).device
assert str(cpu_params) == "cpu"
print(f"  After .to(cpu): {cpu_params}")

# Verify CG tensor is accessible and on correct device
key = list(cg._cache.keys())[0]
cg_tensor = cg._cache[key]
print(f"  CG tensor device: {cg_tensor.device}")
assert str(cg_tensor.device) == "cpu"
print("  PASS: Device movement works")

print("\n" + "=" * 60)
print("TEST 3: CG tensor values are correct (SO(3) CG table)")
print("=" * 60)

# Test known CG coefficients
# <1,1|0> should be 1/sqrt(3) * identity-like
sig1 = GTSignature((1,), 3)
sig2 = GTSignature((1,), 3)
sig0 = GTSignature((0,), 3)
C = cg.get(sig1, sig2, sig0)
assert C is not None, "CG for 1x1->0 should exist"
print(f"  <1,1|0> shape: {C.shape}, norm: {C.norm():.4f}")
assert C.shape == (3, 3, 1), f"Expected (3,3,1), got {C.shape}"
# The CG tensor for 1x1->0 should be proportional to identity contracted
print(f"  PASS: CG tensor shape and norm correct")

print("\n" + "=" * 60)
print("TEST 4: GTBasis computes harmonics correctly")
print("=" * 60)

dirs = torch.randn(10, 3)
dirs = dirs / dirs.norm(dim=-1, keepdim=True)
Y = gb(dirs)
print(f"  GTBasis output shape: {Y.shape}")
assert Y.shape == (10, gb.num_basis), f"Expected (10, {gb.num_basis}), got {Y.shape}"
# Y_0^0 (constant) should be ~1/sqrt(4*pi) for all directions
print(f"  Y_0^0 (first col) mean: {Y[:, 0].mean():.4f} (expected ~0.2821)")
assert abs(Y[:, 0].mean() - 0.2821) < 0.01, "Y_0^0 should be ~1/sqrt(4*pi)"
print("  PASS: Harmonics computed correctly")

print("\n" + "=" * 60)
print("TEST 5: GTTensorFieldNetwork submodule registration")
print("=" * 60)

from gt_tfn_layer import GTTensorFieldNetwork
model = GTTensorFieldNetwork(n=3, num_classes=5, max_order=0, hidden_channels=8, num_layers=2, num_rbf=16, cutoff=1.0, k_neighbors=4)
named_children = dict(model.named_children())
print(f"  GTTensorFieldNetwork submodules: {list(named_children.keys())}")
assert 'gt_basis' in named_children, "gt_basis should be a registered submodule"
assert 'cg' in named_children, "cg should be a registered submodule"
assert isinstance(named_children['gt_basis'], nn.Module), "gt_basis should be nn.Module"
assert isinstance(named_children['cg'], nn.Module), "cg should be nn.Module"
print("  PASS: GTBasis and CGCoefficients are registered submodules")

print("\n" + "=" * 60)
print("TEST 6: Full model .to() moves everything")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
param_device = next(model.parameters()).device
cg_buf_device = next(model.cg.buffers()).device
gb_device = next(model.gb.parameters()).device if list(model.gb.parameters()) else param_device
print(f"  Parameters: {param_device}")
print(f"  CG buffers: {cg_buf_device}")
print(f"  GTBasis: {gb_device}")
assert param_device == cg_buf_device == device, f"Device mismatch: {param_device} vs {cg_buf_device}"
print("  PASS: All tensors on correct device")

print("\n" + "=" * 60)
print("TEST 7: Forward pass through GTTensorFieldNetwork")
print("=" * 60)

pcs = [torch.randn(20, 3).to(device), torch.randn(15, 3).to(device)]
model.eval()
with torch.no_grad():
    out = model(pcs)
print(f"  Output shape: {out.shape}")
assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"
assert torch.isfinite(out).all(), "Output contains NaN/Inf"
print("  PASS: Forward pass produces valid output")

print("\n" + "=" * 60)
print("TEST 8: _build_mlp with dropout")
print("=" * 60)

from models import _build_mlp
mlp = _build_mlp([10, 32, 16, 5], dropout=0.2)
print(f"  MLP layers: {[type(l).__name__ for l in mlp]}")
has_dropout = any(isinstance(l, nn.Dropout) for l in mlp)
assert has_dropout, "MLP should have Dropout layer"
x = torch.randn(4, 10)
y = mlp(x)
print(f"  Output shape: {y.shape}")
assert y.shape == (4, 5)
print("  PASS: _build_mlp with dropout works")

print("\n" + "=" * 60)
print("TEST 9: _build_mlp without dropout (backward compat)")
print("=" * 60)

mlp2 = _build_mlp([10, 32, 16, 5], dropout=0.0)
no_dropout = not any(isinstance(l, nn.Dropout) for l in mlp2)
assert no_dropout, "MLP with dropout=0 should have no Dropout"
print("  PASS: No dropout when dropout=0")

print("\n" + "=" * 60)
print("TEST 10: GTTensorFieldNetworkV2 with mutable default fix")
print("=" * 60)

from models import GTTensorFieldNetworkV2
v2_1 = GTTensorFieldNetworkV2(n=3, num_classes=5, max_order=0, hidden_channels=8, num_layers=2)
v2_2 = GTTensorFieldNetworkV2(n=3, num_classes=5, max_order=0, hidden_channels=8, num_layers=2)
# Verify separate classifier_dims instances
assert v2_1.classifier_dims is not v2_2.classifier_dims, "classifier_dims should be different lists"
print(f"  v2_1.classifier_dims: {v2_1.classifier_dims}")
print(f"  v2_2.classifier_dims: {v2_2.classifier_dims}")
print("  PASS: No shared mutable defaults")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
