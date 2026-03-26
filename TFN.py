import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adamax
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import List, Tuple
import gc

# ---------------------------------------------------------------------------
# Global Parameters
# ---------------------------------------------------------------------------
N_sets_train    = 900
N_sets_test     = 300
N_points        = 600
N_noise         = 200

# Each cloud is subsampled to this many points before the TFN sees it.
# The pairwise geometry is O(N²), so keeping N small is the single biggest
# memory lever.  64 → 64×64 = 4 096 pairs (vs 360 000 at N=600).
TFN_N_SUBSAMPLE = 64

# Process one cloud at a time: avoids stacking multiple N×N tensors.
BATCH_SIZE      = 1

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"   # mixed-precision only on GPU

# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------
def create_circle(N, r, x0, y0):
    theta = np.random.uniform(0, 2 * np.pi, N)
    return np.stack([r * np.cos(theta) + x0, r * np.sin(theta) + y0], axis=1)

def create_1_circle_clean(N):
    r = 2; x0,y0 = 10*np.random.rand()-5, 10*np.random.rand()-5
    return create_circle(N, r, x0, y0)

def create_2_circle_clean(N):
    r1,r2 = 5,3
    x0,y0 = 30*np.random.rand()-15, 30*np.random.rand()-15
    x1,y1 = 30*np.random.rand()-15, 30*np.random.rand()-15
    while np.sqrt((x0-x1)**2+(y0-y1)**2) <= r1+r2:
        x1,y1 = 30*np.random.rand()-15, 30*np.random.rand()-15
    X = np.concatenate([create_circle(N//2,r1,x0,y0),
                        create_circle(N-N//2,r2,x1,y1)])
    np.random.shuffle(X); return X

def create_3_circle_clean(N):
    r0,r1,r2 = 5,3,2
    x0,y0 = 30*np.random.rand()-15, 30*np.random.rand()-15
    x1,y1 = 30*np.random.rand()-15, 30*np.random.rand()-15
    while np.sqrt((x0-x1)**2+(y0-y1)**2) <= r0+r1:
        x1,y1 = 30*np.random.rand()-15, 30*np.random.rand()-15
    x2,y2 = 30*np.random.rand()-15, 30*np.random.rand()-15
    while (np.sqrt((x0-x2)**2+(y0-y2)**2)<=r0+r2 or
           np.sqrt((x1-x2)**2+(y1-y2)**2)<=r1+r2):
        x2,y2 = 30*np.random.rand()-15, 30*np.random.rand()-15
    n = N//3
    X = np.concatenate([create_circle(n,r0,x0,y0),
                        create_circle(n,r1,x1,y1),
                        create_circle(n,r2,x2,y2)])
    np.random.shuffle(X); return X

def _inject_noise(X, N_noise, xlo, xhi, ylo, yhi):
    noise = np.stack([np.random.uniform(xlo,xhi,N_noise),
                      np.random.uniform(ylo,yhi,N_noise)], axis=1)
    X[np.random.choice(len(X), N_noise, replace=False)] = noise
    return X

def create_1_circle_noisy(N, N_noise):
    r=2; x0,y0=10*np.random.rand()-5,10*np.random.rand()-5
    return _inject_noise(create_circle(N,r,x0,y0),N_noise,x0-r,x0+r,y0-r,y0+r)

def create_2_circle_noisy(N, N_noise):
    r1,r2=5,3
    x0,y0=30*np.random.rand()-15,30*np.random.rand()-15
    x1,y1=30*np.random.rand()-15,30*np.random.rand()-15
    while np.sqrt((x0-x1)**2+(y0-y1)**2)<=r1+r2:
        x1,y1=30*np.random.rand()-15,30*np.random.rand()-15
    X=np.concatenate([create_circle(N//2,r1,x0,y0),
                      create_circle(N-N//2,r2,x1,y1)])
    np.random.shuffle(X)
    return _inject_noise(X,N_noise,
                         min(x0-r1,x1-r2),max(x0+r1,x1+r2),
                         min(y0-r1,y1-r2),max(y0+r1,y1+r2))

def create_3_circle_noisy(N, N_noise):
    r0,r1,r2=5,3,2
    x0,y0=30*np.random.rand()-15,30*np.random.rand()-15
    x1,y1=30*np.random.rand()-15,30*np.random.rand()-15
    while np.sqrt((x0-x1)**2+(y0-y1)**2)<=r0+r1:
        x1,y1=30*np.random.rand()-15,30*np.random.rand()-15
    x2,y2=30*np.random.rand()-15,30*np.random.rand()-15
    while (np.sqrt((x0-x2)**2+(y0-y2)**2)<=r0+r2 or
           np.sqrt((x1-x2)**2+(y1-y2)**2)<=r1+r2):
        x2,y2=30*np.random.rand()-15,30*np.random.rand()-15
    n=N//3
    X=np.concatenate([create_circle(n,r0,x0,y0),
                      create_circle(n,r1,x1,y1),
                      create_circle(n,r2,x2,y2)])
    np.random.shuffle(X)
    return _inject_noise(X,N_noise,
                         min(x0-r0,x1-r1,x2-r2),max(x0+r0,x1+r1,x2+r2),
                         min(y0-r0,y1-r1,y2-r2),max(y0+r0,y1+r1,y2+r2))

def create_multiple_circles(N_sets, N_points, noisy=False, N_noise=0):
    """All data stays as CPU numpy arrays — zero GPU memory used here."""
    data, labels = [], np.zeros(N_sets, dtype=np.int64)
    fns = (
        [create_1_circle_noisy, create_2_circle_noisy, create_3_circle_noisy]
        if noisy else
        [create_1_circle_clean, create_2_circle_clean, create_3_circle_clean]
    )
    third = N_sets // 3
    for k, (start, end, fn) in enumerate(
            zip([0,third,2*third],[third,2*third,N_sets],fns), 1):
        for i in tqdm(range(start, end), desc=f"class {k}"):
            data.append(fn(N_points, N_noise) if noisy else fn(N_points))
            labels[i] = k
    perm = np.random.permutation(N_sets)
    return [data[p] for p in perm], labels[perm]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_3d_numpy(pc_list: List[np.ndarray]) -> List[np.ndarray]:
    out = []
    for pc in pc_list:
        if pc.shape[1] == 2:
            pc = np.concatenate([pc, np.zeros((len(pc),1), dtype=pc.dtype)], axis=1)
        out.append(pc)
    return out

def subsample_and_to_tensor(pc: np.ndarray, n: int) -> torch.Tensor:
    if len(pc) > n:
        pc = pc[np.random.choice(len(pc), n, replace=False)]
    return torch.tensor(pc, dtype=torch.float32, device=device)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RBFExpansion(nn.Module):
    def __init__(self, num_rbf=16, cutoff=5.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, num_rbf))
        self.width = (cutoff / num_rbf) ** 2

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(d.unsqueeze(-1) - self.centers)**2 / self.width)


class TFNLayer(nn.Module):
    def __init__(self, in_f0, in_f1, out_f0, out_f1, num_rbf=16):
        super().__init__()
        self.W00 = nn.Sequential(nn.Linear(num_rbf,32),nn.SiLU(),nn.Linear(32,in_f0*out_f0))
        self.W10 = nn.Sequential(nn.Linear(num_rbf,32),nn.SiLU(),nn.Linear(32,in_f1*out_f0))
        self.W01 = nn.Sequential(nn.Linear(num_rbf,32),nn.SiLU(),nn.Linear(32,in_f0*out_f1))
        self.W11 = nn.Sequential(nn.Linear(num_rbf,32),nn.SiLU(),nn.Linear(32,in_f1*out_f1))
        self.norm0 = nn.LayerNorm(out_f0)
        self.norm1 = nn.LayerNorm(out_f1)
        self.in_f0,self.in_f1   = in_f0,in_f1
        self.out_f0,self.out_f1 = out_f0,out_f1

    def forward(self, pos, f0, f1, rbf, r_hat, mask):
        N = pos.shape[0]
        # Use float mask to avoid dtype issues under autocast
        mask_f = mask.float()

        w00   = self.W00(rbf).view(N,N,self.in_f0,self.out_f0)
        msg00 = torch.einsum("ijd,ijdo->io",
                              f0.unsqueeze(0).expand(N,-1,-1) * mask_f.unsqueeze(-1), w00)

        dot   = torch.einsum("ijk,jlk->ijl", r_hat, f1)
        w10   = self.W10(rbf).view(N,N,self.in_f1,self.out_f0)
        msg10 = torch.einsum("ijc,ijco->io", dot * mask_f.unsqueeze(-1), w10)

        new_f0 = self.norm0(msg00 + msg10)

        outer  = r_hat.unsqueeze(-1) * f0.unsqueeze(0).unsqueeze(2)
        w01    = self.W01(rbf).view(N,N,self.in_f0,self.out_f1)
        msg01  = torch.einsum("ijcf,ijfg->igc",
                               outer * mask_f.view(N,N,1,1), w01)

        w11    = self.W11(rbf).view(N,N,self.in_f1,self.out_f1)
        msg11  = torch.einsum("jkx,ijkg->igx", f1, w11 * mask_f.view(N,N,1,1))

        new_f1_raw = msg01 + msg11
        norms      = new_f1_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        new_f1     = new_f1_raw / norms * self.norm1(norms.squeeze(-1)).unsqueeze(-1)
        return new_f0, new_f1


def _pairwise_geometry(pos: torch.Tensor, rbf_encoder: RBFExpansion):
    diff  = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist  = diff.norm(dim=-1)
    r_hat = diff / dist.unsqueeze(-1).clamp(min=1e-8)
    rbf   = rbf_encoder(dist)
    mask  = ~torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)
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

        rho, in_d = [], hidden_f0 + hidden_f1
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
            # gradient checkpointing during training:
            # PyTorch discards intermediate activations and recomputes them
            # on the backward pass → big activation-memory saving per layer.
            if self.training:
                f0, f1 = grad_checkpoint(
                    layer, pos, f0, f1, rbf, r_hat, mask,
                    use_reentrant=False,
                )
            else:
                f0, f1 = layer(pos, f0, f1, rbf, r_hat, mask)

        return torch.cat([f0, f1.norm(dim=-1)], dim=-1).max(dim=0).values

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        if not batch:
            return torch.empty(0, self.rho[-1].out_features,
                               device=next(self.parameters()).device)
        return self.rho(torch.stack([self._encode_single(pc) for pc in batch]))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model_classification(
    model, optimizer, criterion,
    train_data_np, train_targets,
    val_data_np,   val_targets,
    epochs=20, batch_size=BATCH_SIZE, subsample=TFN_N_SUBSAMPLE,
):
    model.to(device)
    # GradScaler: keeps fp16 gradients from underflowing
    scaler           = GradScaler(enabled=USE_AMP)
    patience         = 10
    best_val_loss    = float('inf')
    patience_counter = 0
    history = {'train_loss':[],'val_loss':[],'train_accuracy':[],'val_accuracy':[]}
    best_model_state = None

    train_targets_t = torch.tensor(train_targets, dtype=torch.long, device=device)
    val_targets_t   = torch.tensor(val_targets,   dtype=torch.long, device=device)
    n_train, n_val  = len(train_data_np), len(val_data_np)

    for epoch in range(epochs):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss, correct, total = 0.0, 0, 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start+batch_size]
            # Build tensors just-in-time; released after the step
            batch_pcs = [subsample_and_to_tensor(train_data_np[i], subsample)
                         for i in idx]
            batch_tgt = train_targets_t[idx]

            # set_to_none frees gradient buffers immediately (saves memory vs zero_grad)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=USE_AMP):        # fp16 halves tensor sizes
                outputs = model(batch_pcs)
                loss    = criterion(outputs, batch_tgt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * len(batch_pcs)
            correct    += outputs.argmax(1).eq(batch_tgt).sum().item()
            total      += len(batch_tgt)

            del batch_pcs, outputs, loss, batch_tgt
            if device.type == "cuda":
                torch.cuda.empty_cache()

        history['train_loss'].append(epoch_loss / n_train)
        history['train_accuracy'].append(100 * correct / total)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                idx = np.arange(start, min(start+batch_size, n_val))
                batch_pcs = [subsample_and_to_tensor(val_data_np[i], subsample)
                             for i in idx]
                batch_tgt = val_targets_t[idx]
                with autocast(enabled=USE_AMP):
                    outputs   = model(batch_pcs)
                    val_loss += criterion(outputs, batch_tgt).item() * len(batch_pcs)
                val_correct += outputs.argmax(1).eq(batch_tgt).sum().item()
                val_total   += len(batch_tgt)
                del batch_pcs, outputs, batch_tgt
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        val_loss /= n_val
        val_acc   = 100 * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"train loss {history['train_loss'][-1]:.4f} "
              f"acc {history['train_accuracy'][-1]:.1f}% | "
              f"val loss {val_loss:.4f} acc {val_acc:.1f}%")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss    = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device)
                               for k, v in best_model_state.items()})
    return model, history, best_model_state


def evaluate_model(model, data_np, targets,
                   batch_size=BATCH_SIZE, subsample=TFN_N_SUBSAMPLE):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(data_np), batch_size):
            idx = np.arange(start, min(start+batch_size, len(data_np)))
            batch_pcs = [subsample_and_to_tensor(data_np[i], subsample)
                         for i in idx]
            with autocast(enabled=USE_AMP):
                outputs = model(batch_pcs)
            preds.append(outputs.argmax(1).cpu())
            del batch_pcs, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return accuracy_score(targets, torch.cat(preds).numpy())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Generating data (CPU numpy — no GPU memory used yet)...")
data_train,      label_train       = create_multiple_circles(N_sets_train, N_points, noisy=False, N_noise=N_noise)
clean_data_test, clean_label_test  = create_multiple_circles(N_sets_test,  N_points, noisy=False, N_noise=N_noise)
noisy_data_test, noisy_label_test  = create_multiple_circles(N_sets_test,  N_points, noisy=True,  N_noise=N_noise)

data_train      = to_3d_numpy(data_train)
clean_data_test = to_3d_numpy(clean_data_test)
noisy_data_test = to_3d_numpy(noisy_data_test)

le = LabelEncoder().fit(label_train)
label_classif_train      = le.transform(label_train)
clean_label_classif_test = le.transform(clean_label_test)
noisy_label_classif_test = le.transform(noisy_label_test)

print(f"Data ready — subsample={TFN_N_SUBSAMPLE} pts/cloud | "
      f"batch_size={BATCH_SIZE} | AMP={'on' if USE_AMP else 'off'}")

num_classes = len(np.unique(label_classif_train))
model_tfn   = TensorFieldNetwork(num_classes=num_classes)
optimizer   = Adamax(model_tfn.parameters(), lr=5e-4)
criterion   = nn.CrossEntropyLoss()

model_tfn, history_tfn, _ = train_model_classification(
    model_tfn, optimizer, criterion,
    data_train,      label_classif_train,
    clean_data_test, clean_label_classif_test,
    epochs=10,
)

gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()

clean_acc = evaluate_model(model_tfn, clean_data_test, clean_label_classif_test)
noisy_acc = evaluate_model(model_tfn, noisy_data_test, noisy_label_classif_test)
print(f"\nTFN accuracy — clean: {clean_acc:.4f} | noisy: {noisy_acc:.4f}")
