#!/usr/bin/env python3
"""
NNTrainer_1hidden_145.py

Train a single-hidden-layer network to predict ln(FM coefficients) relative to 294K.
Architecture: 578 -> 145 -> 83521
This follows your working trainer structure but uses HIDDEN_SIZE=145 and saves a new checkpoint.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
class Config:
    # Data paths
    INPUT_TEMPS_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/input_temps_final.npy")
    FM_NORMALIZED_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/output_fm_normalized_final.npy")
    FM_294K_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/output_fm_normalized294k.npy")
    
    # Architecture
    N_ROWS = 17
    N_COLS = 17
    N_CELLS = N_ROWS * N_COLS  # 289
    INPUT_SIZE = N_CELLS * 2    # 578 (fuel temps + other temps concatenated)
    HIDDEN_SIZE = 145           # <--- changed to 145 neurons
    OUTPUT_SIZE = N_CELLS * N_CELLS  # 83521 coefficients

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    MAX_EPOCHS = 10
    VALIDATION_SPLIT = 0.15
    EARLY_STOP_PATIENCE = 15
    LR_REDUCE_PATIENCE = 15
    
    # Output paths
    RESULTS_DIR = Path("fm_coefficient_results")
    MODEL_SAVE_PATH = "fm_coefficient_model_1hidden_145.pth"  # new name
    LOG_FILE = "training_log_1hidden_145.json"
    
    # Regularization for coefficient calculation
    EPSILON = 1e-8  # Small value to avoid log(0) or division by zero

# ---------------- DATA LOADING ----------------
def load_and_prepare_data(cfg: Config):
    """Robust loading + target calculation that tolerates object arrays / odd elements."""
    print("Loading data files...")
    input_temps = np.load(cfg.INPUT_TEMPS_PATH, allow_pickle=True)
    print(f"  Loaded {len(input_temps)} temperature samples")

    fm_samples = np.load(cfg.FM_NORMALIZED_PATH, allow_pickle=True)
    print(f"  Loaded {len(fm_samples)} fission matrix samples")

    fm_294k_data = np.load(cfg.FM_294K_PATH, allow_pickle=True)
    # Normalize fm_294k handling like you had
    if isinstance(fm_294k_data, np.ndarray):
        if fm_294k_data.ndim == 0:
            fm_294k = fm_294k_data.item()
        elif len(fm_294k_data) == 1:
            fm_294k = fm_294k_data[0]
        else:
            fm_294k = fm_294k_data
    else:
        fm_294k = fm_294k_data

    fm_294k = np.asarray(fm_294k, dtype=float)
    if fm_294k.ndim == 1:
        fm_294k = fm_294k.reshape(cfg.N_CELLS, cfg.N_CELLS)

    print(f"  Loaded 294K reference FM with shape {fm_294k.shape}")
    print(f"  294K FM statistics: mean={fm_294k.mean():.4f}, sum={fm_294k.sum():.4f}")

    # NOTE: we will build lists of GOOD feature/target pairs to keep alignment.
    if len(input_temps) != len(fm_samples):
        print("  [WARN] inputs and fm sample counts differ. Will iterate up to min(len(inputs), len(fm_samples)).")

    n_pairs = min(len(input_temps), len(fm_samples))

    # Process inputs into fixed-size feature vectors (same logic as your original code)
    print("\nProcessing input temperatures...")
    features_all = []
    for idx in range(n_pairs):
        temp_tuple = input_temps[idx]
        try:
            fuel_vec, other_vec = temp_tuple
        except Exception:
            # If the structure is different, try to coerce or skip
            print(f"  [WARN] input_temps[{idx}] has unexpected structure: {type(temp_tuple)} -- skipping")
            features_all.append(None)
            continue

        fuel_grid = np.zeros(cfg.N_CELLS, dtype=np.float32)
        fuel_idx = 0
        # Assign sequentially as in original code (best-effort)
        for i in range(cfg.N_CELLS):
            if fuel_idx < len(fuel_vec) and getattr(fuel_vec, 'shape', None) is not None:
                # If fuel_vec element is >0, assign; otherwise keep zero
                try:
                    val = float(fuel_vec[fuel_idx])
                except Exception:
                    val = 0.0
                if val > 0:
                    fuel_grid[i] = val
                    fuel_idx += 1
                else:
                    # If this entry is zero, it's likely a gap — do not advance fuel_idx
                    # advance fuel_idx only when we consumed a fuel value
                    pass
            else:
                # fallback: stop trying to assign
                break

        other_grid = np.asarray(other_vec, dtype=np.float32).flatten()
        if other_grid.size != cfg.N_CELLS:
            # try reshape if possible, otherwise warn and mark None
            if other_grid.size == cfg.N_CELLS:
                other_grid = other_grid
            else:
                print(f"  [WARN] other_vec size != N_CELLS for input index {idx} (size={other_grid.size}) -> skipping")
                features_all.append(None)
                continue

        combined = np.concatenate([fuel_grid, other_grid])
        features_all.append(combined)

    features_all = np.array(features_all, dtype=object)

    print("  Finished building candidate features (including None for skipped inputs).")

    # Now compute targets, only for GOOD fm samples and matching GOOD features
    print("\nComputing coefficient targets...")
    fm_ref_safe = fm_294k + cfg.EPSILON

    good_features = []
    good_targets = []
    skipped_indices = []

    for idx in range(n_pairs):
        feat = features_all[idx]
        raw_fm = fm_samples[idx]

        if feat is None:
            skipped_indices.append((idx, "bad_input_feature"))
            continue

        # Coerce raw_fm into a float ndarray
        try:
            fm_arr = np.asarray(raw_fm, dtype=float)
        except Exception as e:
            skipped_indices.append((idx, f"fm_coercion_error:{repr(e)}"))
            continue

        # Accept either flattened or 2D matrix
        if fm_arr.ndim == 0:
            # scalar -> invalid
            skipped_indices.append((idx, "fm_scalar"))
            continue
        elif fm_arr.ndim == 1:
            if fm_arr.size == cfg.N_CELLS * cfg.N_CELLS:
                fm_arr = fm_arr.reshape(cfg.N_CELLS, cfg.N_CELLS)
            else:
                skipped_indices.append((idx, f"fm_1d_wrong_size:{fm_arr.size}"))
                continue
        elif fm_arr.ndim == 2:
            if fm_arr.shape != (cfg.N_CELLS, cfg.N_CELLS):
                # try to flatten/reshape if total size matches
                print("FM input array is flat")
                if fm_arr.size == cfg.N_CELLS * cfg.N_CELLS:
                    fm_arr = fm_arr.reshape(cfg.N_CELLS, cfg.N_CELLS)
                else:
                    skipped_indices.append((idx, f"fm_2d_wrong_shape:{fm_arr.shape}"))
                    continue
        else:
            skipped_indices.append((idx, f"fm_ndim={fm_arr.ndim}"))
            continue

        # Safe division and log
        try:
            coeff = (fm_arr + cfg.EPSILON) / fm_ref_safe
            ln_coeff = np.log(coeff)
            ln_coeff_flat = np.nan_to_num(ln_coeff.flatten(), nan=0.0, posinf=10.0, neginf=-10.0)
        except Exception as e:
            skipped_indices.append((idx, f"log_error:{repr(e)}"))
            continue

        good_features.append(feat.astype(np.float32))
        good_targets.append(ln_coeff_flat.astype(np.float32))

    if len(skipped_indices) > 0:
        print("\n[WARN] Skipped the following sample indices due to issues:")
        for idx, reason in skipped_indices[:20]:
            print(f"  idx={idx}: {reason}")
        if len(skipped_indices) > 20:
            print(f"  ... and {len(skipped_indices)-20} more")

    if len(good_features) == 0:
        raise RuntimeError("No valid feature/FM pairs found after cleaning. Aborting.")

    features_final = np.stack(good_features, axis=0)
    targets_final = np.stack(good_targets, axis=0)

    print(f"\nFinal dataset sizes: features={features_final.shape}, targets={targets_final.shape}")
    print(f"  (skipped {len(skipped_indices)} of {n_pairs} candidate pairs)")

    return features_final, targets_final, fm_294k


# ---------------- NEURAL NETWORK ----------------
class FMCoefficientPredictor(nn.Module):
    """Single hidden layer network: 578 -> 145 -> 83521"""
    def __init__(self, cfg: Config):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(cfg.INPUT_SIZE, cfg.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_SIZE, cfg.OUTPUT_SIZE)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 578) - concatenated [fuel_temps (289), other_temps (289)]
        Returns:
            (batch, 83521) - ln(coefficients) for each FM position
        """
        return self.network(x)

# ---------------- TRAINING UTILITIES ----------------
class TrainingLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.metrics = {
            "start_time": datetime.now().isoformat(),
            "epochs": [],
            "best_epoch": None,
            "final_stats": {}
        }
    
    def log_epoch(self, epoch, train_loss, val_loss, lr):
        entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "learning_rate": float(lr),
            "timestamp": datetime.now().isoformat()
        }
        self.metrics["epochs"].append(entry)
    
    def log_final_stats(self, stats):
        self.metrics["final_stats"] = stats
        self.metrics["end_time"] = datetime.now().isoformat()
    
    def save(self):
        with open(self.log_dir / Config.LOG_FILE, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def print_summary(self):
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        if self.metrics["epochs"]:
            train_losses = [e["train_loss"] for e in self.metrics["epochs"]]
            val_losses = [e["val_loss"] for e in self.metrics["epochs"]]
            print(f"Total epochs: {len(self.metrics['epochs'])}")
            print(f"Best validation loss: {min(val_losses):.6f} at epoch {val_losses.index(min(val_losses)) + 1}")
            print(f"Final train loss: {train_losses[-1]:.6f}")
            print(f"Final val loss: {val_losses[-1]:.6f}")
        print("="*80 + "\n")

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches

def plot_training_history(logger, save_path):
    """Plot training and validation loss curves"""
    epochs = [e["epoch"] for e in logger.metrics["epochs"]]
    train_losses = [e["train_loss"] for e in logger.metrics["epochs"]]
    val_losses = [e["val_loss"] for e in logger.metrics["epochs"]]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training History', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Saved training curve to {save_path}")
    plt.close()

# ---------------- MAIN TRAINING LOOP ----------------
def main():
    cfg = Config()
    cfg.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("FISSION MATRIX COEFFICIENT PREDICTOR")
    print("="*80)
    print(f"Architecture: {cfg.INPUT_SIZE} -> {cfg.HIDDEN_SIZE} -> {cfg.OUTPUT_SIZE}")
    print(f"Max epochs: {cfg.MAX_EPOCHS}")
    print(f"Batch size: {cfg.BATCH_SIZE}")
    print("="*80 + "\n")
    
    # Load and prepare data
    features, targets, fm_294k = load_and_prepare_data(cfg)
    n_samples = features.shape[0]
    
    # Split into train/validation
    n_val = int(n_samples * cfg.VALIDATION_SPLIT)
    n_train = n_samples - n_val
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_X, train_y = features[train_indices], targets[train_indices]
    val_X, val_y = features[val_indices], targets[val_indices]
    
    print(f"\nDataset split:")
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {n_val}")
    
    # Normalize inputs
    train_mean = train_X.mean(axis=0)
    train_std = train_X.std(axis=0) + 1e-8
    
    train_X_norm = (train_X - train_mean) / train_std
    val_X_norm = (val_X - train_mean) / train_std
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_X_norm),
        torch.FloatTensor(train_y)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_X_norm),
        torch.FloatTensor(val_y)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model = FMCoefficientPredictor(cfg).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=cfg.LR_REDUCE_PATIENCE, verbose=True
    )
    
    # Training tracking
    logger = TrainingLogger(cfg.RESULTS_DIR)
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    # Training loop
    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Log
        logger.log_epoch(epoch, train_loss, val_loss, current_lr)
        
        # Print progress
        print(f"Epoch {epoch:3d}/{cfg.MAX_EPOCHS} [{epoch_time:.1f}s] | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.6f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ New best validation loss!")
        else:
            epochs_no_improve += 1
        
        # Early stopping check
        if epochs_no_improve >= cfg.EARLY_STOP_PATIENCE:
            print(f"\n{'='*80}")
            print(f"Early stopping: No improvement for {cfg.EARLY_STOP_PATIENCE} epochs")
            print(f"{'='*80}")
            break
        
        # Save checkpoint periodically
        if epoch % 10 == 0:
            logger.save()
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Restored best model from epoch {best_epoch}")
    
    # Final validation
    final_val_loss = validate(model, val_loader, criterion, device)
    print(f"\nFinal validation loss: {final_val_loss:.6f}")
    
    # Test prediction on one sample
    print("\n" + "="*80)
    print("SAMPLE PREDICTION TEST")
    print("="*80)
    model.eval()
    with torch.no_grad():
        test_input = torch.FloatTensor(val_X_norm[0:1]).to(device)
        test_output = model(test_input).cpu().numpy().reshape(cfg.N_CELLS, cfg.N_CELLS)
        
        # Convert back to coefficients
        test_coefficients = np.exp(test_output)
        
        print(f"Predicted ln(coefficient) stats:")
        print(f"  Mean: {test_output.mean():.4f}")
        print(f"  Std:  {test_output.std():.4f}")
        print(f"  Range: [{test_output.min():.4f}, {test_output.max():.4f}]")
        print(f"\nPredicted coefficient stats:")
        print(f"  Mean: {test_coefficients.mean():.4f}")
        print(f"  Std:  {test_coefficients.std():.4f}")
        print(f"  Range: [{test_coefficients.min():.4f}, {test_coefficients.max():.4f}]")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_mean': train_mean,
        'train_std': train_std,
        'fm_294k': fm_294k,
        'config': vars(cfg),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_val_loss': final_val_loss
    }, cfg.RESULTS_DIR / cfg.MODEL_SAVE_PATH)
    print(f"\n✓ Saved model to {cfg.RESULTS_DIR / cfg.MODEL_SAVE_PATH}")
    
    # Log final stats
    final_stats = {
        'total_epochs': epoch,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'final_val_loss': float(final_val_loss),
        'n_train_samples': int(n_train),
        'n_val_samples': int(n_val),
        'n_parameters': int(n_params)
    }
    logger.log_final_stats(final_stats)
    logger.save()
    
    # Plot training history
    plot_training_history(logger, cfg.RESULTS_DIR / "training_curve_1hidden_145.png")
    
    # Print summary
    logger.print_summary()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
