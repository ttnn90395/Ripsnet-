import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adamax
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import List, Tuple
import gc

# --- Global Parameters ---
N_sets_train = 900
N_sets_test  = 300
N_points     = 600
N_noise      = 200

# Subsample each point cloud to this size before feeding into TFN.
# Reduces the O(N²) pairwise matrix from 600×600 → 128×128 (22× fewer pairs).
TFN_N_SUBSAMPLE = 128

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Generation Functions ---
def create_circle(N_points, r, x_0, y_0):
    theta = np.random.uniform(0, 2 * np.pi, N_points)
    return np.stack([r * np.cos(theta) + x_0, r * np.sin(theta) + y_0], axis=1)

def create_1_circle_clean(N_points):
    r = 2
    x_0, y_0 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5
    return create_circle(N_points, r, x_0, y_0)

def create_2_circle_clean(N_points):
    r1, r2 = 5, 3
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while np.sqrt((x_0 - x_1) ** 2 + (y_0 - y_1) ** 2) <= r1 + r2:
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    c1 = create_circle(N_points // 2, r1, x_0, y_0)
    c2 = create_circle(N_points - N_points // 2, r2, x_1, y_1)
    X = np.concatenate([c1, c2], axis=0)
    np.random.shuffle(X)
    return X

def create_3_circle_clean(N_points):
    r0, r1, r2 = 5, 3, 2
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while np.sqrt((x_0 - x_1) ** 2 + (y_0 - y_1) ** 2) <= r0 + r1:
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while (np.sqrt((x_0 - x_2) ** 2 + (y_0 - y_2) ** 2) <= r0 + r2) or \
          (np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2) <= r1 + r2):
        x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    n = N_points // 3
    X = np.concatenate([
        create_circle(n, r0, x_0, y_0),
        create_circle(n, r1, x_1, y_1),
        create_circle(n, r2, x_2, y_2),
    ], axis=0)
    np.random.shuffle(X)
    return X

def _add_noise(X, N_noise, x_min, x_max, y_min, y_max):
    noise = np.stack([
        np.random.uniform(x_min, x_max, N_noise),
        np.random.uniform(y_min, y_max, N_noise),
    ], axis=1)
    idx = np.random.choice(len(X), size=N_noise, replace=False)
    X[idx] = noise
    return X

def create_1_circle_noisy(N_points, N_noise):
    r, x_0, y_0 = 2, 10 * np.random.rand() - 5, 10 * np.random.rand() - 5
    X = create_circle(N_points, r, x_0, y_0)
    return _add_noise(X, N_noise, x_0 - r, x_0 + r, y_0 - r, y_0 + r)

def create_2_circle_noisy(N_points, N_noise):
    r1, r2 = 5, 3
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while np.sqrt((x_0 - x_1) ** 2 + (y_0 - y_1) ** 2) <= r1 + r2:
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    c1 = create_circle(N_points // 2, r1, x_0, y_0)
    c2 = create_circle(N_points - N_points // 2, r2, x_1, y_1)
    X = np.concatenate([c1, c2], axis=0)
    np.random.shuffle(X)
    return _add_noise(X, N_noise,
                      min(x_0 - r1, x_1 - r2), max(x_0 + r1, x_1 + r2),
                      min(y_0 - r1, y_1 - r2), max(y_0 + r1, y_1 + r2))

def create_3_circle_noisy(N_points, N_noise):
    r0, r1, r2 = 5, 3, 2
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while np.sqrt((x_0 - x_1) ** 2 + (y_0 - y_1) ** 2) <= r0 + r1:
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while (np.sqrt((x_0 - x_2) ** 2 + (y_0 - y_2) ** 2) <= r0 + r2) or \
          (np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2) <= r1 + r2):
        x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    n = N_points // 3
    X = np.concatenate([
        create_circle(n, r0, x_0, y_0),
        create_circle(n, r1, x_1, y_1),
        create_circle(n, r2, x_2, y_2),
    ], axis=0)
    np.random.shuffle(X)
    return _add_noise(X, N_noise,
                      min(x_0 - r0, x_1 - r1, x_2 - r2),
                      max(x_0 + r0, x_1 + r1, x_2 + r2),
                      min(y_0 - r0, y_1 - r1, y_2 - r2),
                      max(y_0 + r0, y_1 + r1, y_2 + r2))

def create_multiple_circles(N_sets, N_points, noisy=False, N_noise=0):
    """
    Returns data as a list of CPU numpy arrays (never moved to GPU en masse).
    Labels are returned as a numpy int array.
    """
    data, labels = [], np.zeros(N_sets, dtype=np.int64)
    fns = (
        (create_1_circle_noisy, create_2_circle_noisy, create_3_circle_noisy)
        if noisy else
        (create_1_circle_clean, create_2_circle_clean, create_3_circle_clean)
    )
    third = N_sets // 3
    for label_idx, (start, end, fn) in enumerate(zip(
        [0, third, 2 * third],
        [third, 2 * third, N_sets],
        fns,
    ), start=1):
        for i in tqdm(range(start, end)):
            data.append(fn(N_points, N_noise) if noisy else fn(N_points))
            labels[i] = label_idx

    shuffler = np.random.permutation(N_sets)
    labels = labels[shuffler]
    data = [data[p] for p in shuffler]
    return data, labels

# --- TFN Utilities ---
def convert_2d_to_3d_numpy(pc_list: List[np.ndarray]) -> List[np.ndarray]:
    """Adds a zero z-column in-place (numpy only, no GPU memory used)."""
    return [
        np.concatenate([pc, np.zeros((pc.shape[0], 1), dtype=pc.dtype)], axis=1)
        if pc.shape[1] == 2 else pc
        for pc in pc_list
    ]

def subsample_pc(pc: np.ndarray, n: int) -> np.ndarray:
    """
    Randomly subsample a point cloud to n points.
    If pc already has ≤ n points, return it unchanged.
    """
    if len(pc) <= n:
        return pc
    idx = np.random.choice(len(pc), size=n, replace=False)
    return pc[idx]

def numpy_to_tensor(pc: np.ndarray) -> torch.Tensor:
    """Convert a single numpy point cloud to a float32 CUDA/CPU tensor."""
    return torch.tensor(pc, dtype=torch.float32, device=device)

# --- RBF & TFN Model ---
class RBFExpansion(nn.Module):
    def __init__(self, num_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.register_buffer("centers", torch.linspace(0.0, cutoff, num_rbf))
        self.width = (cutoff / num_rbf) ** 2

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(distances.unsqueeze(-1) - self.centers) ** 2 / self.width)


class TFNLayer(nn.Module):
    def __init__(self, in_f0, in_f1, out_f0, out_f1, num_rbf=16):
        super().__init__()
        self.W00 = nn.Sequential(nn.Linear(num_rbf, 32), nn.SiLU(), nn.Linear(32, in_f0 * out_f0))
        self.W10 = nn.Sequential(nn.Linear(num_rbf, 32), nn.SiLU(), nn.Linear(32, in_f1 * out_f0))
        self.W01 = nn.Sequential(nn.Linear(num_rbf, 32), nn.SiLU(), nn.Linear(32, in_f0 * out_f1))
        self.W11 = nn.Sequential(nn.Linear(num_rbf, 32), nn.SiLU(), nn.Linear(32, in_f1 * out_f1))
        self.norm0 = nn.LayerNorm(out_f0)
        self.norm1 = nn.LayerNorm(out_f1)
        self.in_f0, self.in_f1 = in_f0, in_f1
        self.out_f0, self.out_f1 = out_f0, out_f1

    def forward(self, pos, f0, f1, rbf, r_hat, mask):
        N = pos.shape[0]

        w00 = self.W00(rbf).view(N, N, self.in_f0, self.out_f0)
        msg00 = torch.einsum("ijd,ijdo->io", f0.unsqueeze(0).expand(N, -1, -1) * mask.unsqueeze(-1), w00)

        dot = torch.einsum("ijk,jlk->ijl", r_hat, f1)
        w10 = self.W10(rbf).view(N, N, self.in_f1, self.out_f0)
        msg10 = torch.einsum("ijc,ijco->io", dot * mask.unsqueeze(-1), w10)

        new_f0 = self.norm0(msg00 + msg10)

        outer = r_hat.unsqueeze(-1) * f0.unsqueeze(0).unsqueeze(2)
        w01 = self.W01(rbf).view(N, N, self.in_f0, self.out_f1)
        msg01 = torch.einsum("ijcf,ijfg->igc", outer * mask.view(N, N, 1, 1), w01)

        w11 = self.W11(rbf).view(N, N, self.in_f1, self.out_f1)
        msg11 = torch.einsum("jkx,ijkg->igx", f1, w11 * mask.view(N, N, 1, 1))

        new_f1_raw = msg01 + msg11
        norms = new_f1_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = self.norm1(norms.squeeze(-1)).unsqueeze(-1)
        new_f1 = new_f1_raw / norms * scale

        return new_f0, new_f1


def _pairwise_geometry(pos: torch.Tensor, rbf_encoder: RBFExpansion):
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)       # (N, N, 3) — but N is now TFN_N_SUBSAMPLE
    dist = diff.norm(dim=-1)
    r_hat = diff / dist.unsqueeze(-1).clamp(min=1e-8)
    rbf = rbf_encoder(dist)
    N = pos.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=pos.device)
    return rbf, r_hat, mask


class TensorFieldNetwork(nn.Module):
    def __init__(self, num_classes, hidden_f0=64, hidden_f1=16,
                 num_layers=3, num_rbf=16, cutoff=5.0,
                 classifier_dims=(128, 64)):
        super().__init__()
        self.rbf = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)

        layers, in_f0, in_f1 = [], 1, 1
        for _ in range(num_layers):
            layers.append(TFNLayer(in_f0, in_f1, hidden_f0, hidden_f1, num_rbf))
            in_f0, in_f1 = hidden_f0, hidden_f1
        self.layers = nn.ModuleList(layers)

        inv_dim = hidden_f0 + hidden_f1
        rho, in_d = [], inv_dim
        for d in classifier_dims:
            rho += [nn.Linear(in_d, d), nn.ReLU()]
            in_d = d
        rho.append(nn.Linear(in_d, num_classes))
        self.rho = nn.Sequential(*rho)

    def _encode_single(self, pos: torch.Tensor) -> torch.Tensor:
        f0 = pos.norm(dim=-1, keepdim=True)
        f1 = pos.unsqueeze(1)
        rbf, r_hat, mask = _pairwise_geometry(pos, self.rbf)
        for layer in self.layers:
            f0, f1 = layer(pos, f0, f1, rbf, r_hat, mask)
        return torch.cat([f0, f1.norm(dim=-1)], dim=-1).max(dim=0).values

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        if not batch:
            return torch.empty(0, self.rho[-1].out_features, device=next(self.parameters()).device)
        return self.rho(torch.stack([self._encode_single(pc) for pc in batch]))


# --- Memory-efficient training loop ---
def train_model_classification(
    model, optimizer, criterion,
    train_data_np, train_targets,
    val_data_np, val_targets,
    epochs=20, batch_size=32,
    subsample=TFN_N_SUBSAMPLE,
):
    """
    train_data_np / val_data_np : lists of numpy arrays (CPU).
    Tensors are created on-the-fly per batch and immediately released.
    """
    model.to(device)
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    best_model_state = None

    train_targets_t = torch.tensor(train_targets, dtype=torch.long, device=device)
    val_targets_t   = torch.tensor(val_targets,   dtype=torch.long, device=device)
    n_train = len(train_data_np)

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss, correct, total = 0.0, 0, 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]

            # --- Build batch tensors on-the-fly, subsample to save memory ---
            batch_pcs = [
                numpy_to_tensor(subsample_pc(train_data_np[i], subsample))
                for i in idx
            ]
            batch_targets = train_targets_t[idx]

            optimizer.zero_grad()
            outputs = model(batch_pcs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_pcs)
            _, predicted = outputs.max(1)
            correct += (predicted == batch_targets).sum().item()
            total   += len(batch_targets)

            # Explicitly free batch tensors
            del batch_pcs, outputs, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

        epoch_loss /= n_train
        train_acc = 100 * correct / total
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(train_acc)

        # --- Validation: process in batches too ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        n_val = len(val_data_np)

        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                idx = np.arange(start, min(start + batch_size, n_val))
                batch_pcs = [
                    numpy_to_tensor(subsample_pc(val_data_np[i], subsample))
                    for i in idx
                ]
                batch_targets = val_targets_t[idx]
                outputs = model(batch_pcs)
                val_loss    += criterion(outputs, batch_targets).item() * len(batch_pcs)
                _, predicted = outputs.max(1)
                val_correct += (predicted == batch_targets).sum().item()
                val_total   += len(batch_targets)

                del batch_pcs, outputs
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        val_loss /= n_val
        val_acc   = 100 * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train loss {epoch_loss:.4f} acc {train_acc:.1f}% | "
              f"Val loss {val_loss:.4f} acc {val_acc:.1f}%")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    return model, history, best_model_state


def evaluate_model(model, data_np, targets, batch_size=32, subsample=TFN_N_SUBSAMPLE):
    """
    Batched evaluation that avoids loading the entire test set on GPU at once.
    Returns accuracy (float in [0, 1]).
    """
    model.eval()
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    all_preds = []
    n = len(data_np)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = np.arange(start, min(start + batch_size, n))
            batch_pcs = [
                numpy_to_tensor(subsample_pc(data_np[i], subsample))
                for i in idx
            ]
            outputs = model(batch_pcs)
            _, predicted = outputs.max(1)
            all_preds.append(predicted.cpu())

            del batch_pcs, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

    all_preds = torch.cat(all_preds).numpy()
    return accuracy_score(targets, all_preds)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
print("Generating data  (stored as numpy on CPU — no GPU memory used yet)...")
data_train,      label_train      = create_multiple_circles(N_sets_train, N_points, noisy=False, N_noise=N_noise)
clean_data_test, clean_label_test = create_multiple_circles(N_sets_test,  N_points, noisy=False, N_noise=N_noise)
noisy_data_test, noisy_label_test = create_multiple_circles(N_sets_test,  N_points, noisy=True,  N_noise=N_noise)

# Convert to 3-D (still numpy, still CPU)
data_train      = convert_2d_to_3d_numpy(data_train)
clean_data_test = convert_2d_to_3d_numpy(clean_data_test)
noisy_data_test = convert_2d_to_3d_numpy(noisy_data_test)

le = LabelEncoder().fit(label_train)
label_classif_train     = le.transform(label_train)
clean_label_classif_test = le.transform(clean_label_test)
noisy_label_classif_test = le.transform(noisy_label_test)

print(f"Data ready. Each cloud will be subsampled to {TFN_N_SUBSAMPLE} pts inside the training loop.")

# Model
num_classes = len(np.unique(label_classif_train))
model_tfn   = TensorFieldNetwork(num_classes=num_classes)
optimizer   = Adamax(model_tfn.parameters(), lr=5e-4)
criterion   = nn.CrossEntropyLoss()

print(f"TFN initialised ({num_classes} classes). Training...")
model_tfn, history_tfn, best_state_tfn = train_model_classification(
    model_tfn, optimizer, criterion,
    data_train,      label_classif_train,
    clean_data_test, clean_label_classif_test,
    epochs=10000, batch_size=32,
)

# Explicit GC pass before evaluation
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()

clean_acc = evaluate_model(model_tfn, clean_data_test, clean_label_classif_test)
noisy_acc = evaluate_model(model_tfn, noisy_data_test, noisy_label_classif_test)

print(f"\nTFN accuracy — clean test: {clean_acc:.4f} | noisy test: {noisy_acc:.4f}")
