"""
Comprehensive Model Testing Script

Combines experiment_augmentation.py and TFN.py functionality to test all models
in the repository on the circle classification task.

Uses circle generation from datasets/utils.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adamax
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import List, Tuple
import gc
import gudhi as gd

# Import all models from centralized models.py
from models import (
    # TFN models
    TensorFieldNetwork,
    GTTensorFieldNetwork,
    HierarchicalGTTFN,
    # Notebook models
    ScalarDistanceDeepSet,
    PointNetTutorial,
    ScalarInputMLP,
    MultiInputModel,
    DenseRagged,
    PermopRagged,
    RaggedPersistenceModel,
    DistanceMatrixRaggedModel,
)

# Import data generation from datasets/utils.py
from datasets.utils import (
    create_multiple_circles,
    create_1_circle_clean,
    create_2_circle_clean,
    create_3_circle_clean,
    create_1_circle_noisy,
    create_2_circle_noisy,
    create_3_circle_noisy,
    compute_PD,
)

# ---------------------------------------------------------------------------
# Global Parameters
# ---------------------------------------------------------------------------
N_sets_train    = 900
N_sets_test     = 300
N_points        = 600
N_noise         = 200

# TFN specific parameters
TFN_N_SUBSAMPLE = 64
BATCH_SIZE      = 1

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"

# ---------------------------------------------------------------------------
# Helper Functions
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

def distance_matrix(point_cloud):
    """
    Compute pairwise Euclidean distance matrix for a point cloud.
    """
    pc = np.asarray(point_cloud, dtype=float)
    if pc.ndim == 1:
        pc = pc.reshape(-1, 1)
    diff = pc[:, None, :] - pc[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

# ---------------------------------------------------------------------------
# Model Wrappers for Classification
# ---------------------------------------------------------------------------
class ClassificationWrapper(nn.Module):
    """Wrapper to add classification head to feature-extracting models"""
    def __init__(self, feature_model, num_classes, feature_dim=None):
        super().__init__()
        self.feature_model = feature_model
        self.classifier = nn.Linear(feature_dim or feature_model.rho[-1].out_features, num_classes)

    def forward(self, x):
        features = self.feature_model(x)
        return self.classifier(features)

class TFNWrapper(nn.Module):
    """Wrapper for TFN models that already output classification"""
    def __init__(self, tfn_model):
        super().__init__()
        self.tfn_model = tfn_model

    def forward(self, x):
        return self.tfn_model(x)

class PersistenceModelWrapper(nn.Module):
    """Wrapper for persistence diagram models"""
    def __init__(self, persistence_model, num_classes):
        super().__init__()
        self.persistence_model = persistence_model
        # Assume output_dim is set appropriately in the model
        self.classifier = nn.Linear(persistence_model.fc[-1].out_features, num_classes)

    def forward(self, x):
        features = self.persistence_model(x)
        return self.classifier(features)

# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------
def create_model(model_name, num_classes):
    """Create and return the specified model wrapped for classification"""
    if model_name == 'TensorFieldNetwork':
        base_model = TensorFieldNetwork(num_classes=num_classes)
        return TFNWrapper(base_model)

    elif model_name == 'GTTensorFieldNetwork':
        base_model = GTTensorFieldNetwork(n=2, num_classes=num_classes)  # 2D point clouds
        return TFNWrapper(base_model)

    elif model_name == 'HierarchicalGTTFN':
        base_model = HierarchicalGTTFN(n=2, num_classes=num_classes)  # 2D point clouds
        return TFNWrapper(base_model)

    elif model_name == 'ScalarDistanceDeepSet':
        # This model expects flattened upper triangular distances
        def flatten_upper_triangular(dm):
            n = dm.shape[0]
            upper_indices = torch.triu_indices(n, n, offset=1)
            return dm[upper_indices[0], upper_indices[1]]
        
        base_model = ScalarDistanceDeepSet(output_dim=128)
        return ClassificationWrapper(base_model, num_classes, feature_dim=128)

    elif model_name == 'PointNetTutorial':
        base_model = PointNetTutorial(output_dim=128)  # Feature dimension
        return ClassificationWrapper(base_model, num_classes, feature_dim=128)

    elif model_name == 'DistanceMatrixRaggedModel':
        base_model = DistanceMatrixRaggedModel(output_dim=128, num_points=600)  # Fixed size for our dataset
        return ClassificationWrapper(base_model, num_classes, feature_dim=128)

    elif model_name == 'RaggedPersistenceModel':
        base_model = RaggedPersistenceModel(output_dim=128)  # Feature dimension
        return PersistenceModelWrapper(base_model, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

# ---------------------------------------------------------------------------
# Data Preparation Functions
# ---------------------------------------------------------------------------
def prepare_data_for_model(model_name, data_list):
    """Prepare data in the format expected by each model"""
    if model_name in ['TensorFieldNetwork', 'GTTensorFieldNetwork', 'HierarchicalGTTFN', 'PointNetTutorial']:
        # These models expect 3D point clouds
        return to_3d_numpy(data_list)

    elif model_name == 'PointNetTutorial':
        # This model expects 2D point clouds
        return [pc[:, :2] for pc in data_list]  # Remove z-coordinate

    elif model_name == 'ScalarDistanceDeepSet':
        # This model expects flattened upper triangular distances
        def flatten_upper_triangular(dm):
            n = dm.shape[0]
            upper_indices = torch.triu_indices(n, n, offset=1)
            return dm[upper_indices[0], upper_indices[1]]
        
        return [flatten_upper_triangular(torch.tensor(distance_matrix(pc), dtype=torch.float32)) for pc in data_list]

    elif model_name == 'RaggedPersistenceModel':
        # This model expects persistence diagrams, but we'll use distance matrices for now
        # In a real scenario, you'd compute actual persistence diagrams
        return [distance_matrix(pc) for pc in data_list]

    else:
        return data_list

# ---------------------------------------------------------------------------
# Training Function (adapted from TFN.py)
# ---------------------------------------------------------------------------
def train_model_classification(
    model, optimizer, criterion,
    train_data_np, train_targets,
    val_data_np, val_targets,
    model_name, epochs=20, batch_size=BATCH_SIZE, subsample=TFN_N_SUBSAMPLE,
):
    model.to(device)
    scaler = GradScaler(enabled=USE_AMP)
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    best_model_state = None

    train_targets_t = torch.tensor(train_targets, dtype=torch.long, device=device)
    val_targets_t = torch.tensor(val_targets, dtype=torch.long, device=device)
    n_train, n_val = len(train_data_np), len(val_data_np)

    for epoch in range(epochs):
        # Training
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss, correct, total = 0.0, 0, 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start+batch_size]

            # Prepare batch data
            if model_name in ['TensorFieldNetwork', 'GTTensorFieldNetwork', 'HierarchicalGTTFN']:
                batch_pcs = [subsample_and_to_tensor(train_data_np[i], subsample) for i in idx]
            else:
                batch_pcs = [torch.tensor(train_data_np[i], dtype=torch.float32, device=device) for i in idx]

            batch_tgt = train_targets_t[idx]

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=USE_AMP):
                outputs = model(batch_pcs)
                loss = criterion(outputs, batch_tgt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * len(batch_pcs)
            correct += outputs.argmax(1).eq(batch_tgt).sum().item()
            total += len(batch_tgt)

            del batch_pcs, outputs, loss, batch_tgt
            if device.type == "cuda":
                torch.cuda.empty_cache()

        history['train_loss'].append(epoch_loss / n_train)
        history['train_accuracy'].append(100 * correct / total)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                idx = np.arange(start, min(start+batch_size, n_val))

                if model_name in ['TensorFieldNetwork', 'GTTensorFieldNetwork', 'HierarchicalGTTFN']:
                    batch_pcs = [subsample_and_to_tensor(val_data_np[i], subsample) for i in idx]
                else:
                    batch_pcs = [torch.tensor(val_data_np[i], dtype=torch.float32, device=device) for i in idx]

                batch_tgt = val_targets_t[idx]
                with autocast(enabled=USE_AMP):
                    outputs = model(batch_pcs)
                    val_loss += criterion(outputs, batch_tgt).item() * len(batch_pcs)
                val_correct += outputs.argmax(1).eq(batch_tgt).sum().item()
                val_total += len(batch_tgt)
                del batch_pcs, outputs, batch_tgt
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        val_loss /= n_val
        val_acc = 100 * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"train loss {history['train_loss'][-1]:.4f} "
              f"acc {history['train_accuracy'][-1]:.1f}% | "
              f"val loss {val_loss:.4f} acc {val_acc:.1f}%")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
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

# ---------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------
def evaluate_model(model, data_np, targets, model_name,
                   batch_size=BATCH_SIZE, subsample=TFN_N_SUBSAMPLE):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(data_np), batch_size):
            idx = np.arange(start, min(start+batch_size, len(data_np)))

            if model_name in ['TensorFieldNetwork', 'GTTensorFieldNetwork', 'HierarchicalGTTFN']:
                batch_pcs = [subsample_and_to_tensor(data_np[i], subsample) for i in idx]
            else:
                batch_pcs = [torch.tensor(data_np[i], dtype=torch.float32, device=device) for i in idx]

            with autocast(enabled=USE_AMP):
                outputs = model(batch_pcs)
            preds.append(outputs.argmax(1).cpu())
            del batch_pcs, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return accuracy_score(targets, torch.cat(preds).numpy())

# ---------------------------------------------------------------------------
# Main Testing Function
# ---------------------------------------------------------------------------
def test_all_models():
    """Test all available models on the circle classification task"""
    print("Generating data...")
    data_train, label_train = create_multiple_circles(N_sets_train, N_points, noisy=False, N_noise=N_noise)
    clean_data_test, clean_label_test = create_multiple_circles(N_sets_test, N_points, noisy=False, N_noise=N_noise)
    noisy_data_test, noisy_label_test = create_multiple_circles(N_sets_test, N_points, noisy=True, N_noise=N_noise)

    # Convert labels to 0-based indexing for PyTorch
    le = LabelEncoder().fit(label_train)
    label_classif_train = le.transform(label_train)
    clean_label_classif_test = le.transform(clean_label_test)
    noisy_label_classif_test = le.transform(noisy_label_test)

    num_classes = len(np.unique(label_classif_train))
    print(f"Data ready — {num_classes} classes")

    # List of models to test
    model_names = [
        'TensorFieldNetwork',
        'GTTensorFieldNetwork',
        'HierarchicalGTTFN',
        'ScalarDistanceDeepSet',
        'PointNetTutorial',
        'DistanceMatrixRaggedModel',
        'RaggedPersistenceModel',
    ]

    results = {}

    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Testing {model_name}")
        print(f"{'='*50}")

        try:
            # Prepare data for this model
            train_data_prepared = prepare_data_for_model(model_name, data_train)
            clean_test_data_prepared = prepare_data_for_model(model_name, clean_data_test)
            noisy_test_data_prepared = prepare_data_for_model(model_name, noisy_data_test)

            # Create model
            model = create_model(model_name, num_classes)
            optimizer = Adamax(model.parameters(), lr=5e-4)
            criterion = nn.CrossEntropyLoss()

            # Train model
            print(f"Training {model_name}...")
            model, history, _ = train_model_classification(
                model, optimizer, criterion,
                train_data_prepared, label_classif_train,
                clean_test_data_prepared, clean_label_classif_test,
                model_name, epochs=10,
            )

            # Evaluate on clean test data
            clean_acc = evaluate_model(model, clean_test_data_prepared, clean_label_classif_test, model_name)

            # Evaluate on noisy test data
            noisy_acc = evaluate_model(model, noisy_test_data_prepared, noisy_label_classif_test, model_name)

            results[model_name] = {
                'clean_accuracy': clean_acc,
                'noisy_accuracy': noisy_acc,
                'history': history,
            }

            print(f"\n{model_name} accuracy — clean: {clean_acc:.4f} | noisy: {noisy_acc:.4f}")
            # Clean up memory
            del model, optimizer, criterion
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Clean Acc':<12} {'Noisy Acc':<12}")
    print("-" * 60)

    for model_name, result in results.items():
        if 'error' in result:
            print(f"{model_name:<25} {'ERROR':<12} {'ERROR':<12}")
        else:
            clean_acc = result['clean_accuracy']
            noisy_acc = result['noisy_accuracy']
            print(f"{model_name:<25} {clean_acc:<12.4f} {noisy_acc:<12.4f}")

    return results

if __name__ == "__main__":
    results = test_all_models()