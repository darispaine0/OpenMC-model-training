#!/usr/bin/env python3
"""
NNTrainer3Hidden.py

Train network with architecture:
  578 -> 1024 -> 512 -> 1024 -> 83521
(using nn.Sequential as `network`, with optional BatchNorm + Dropout)

Saves checkpoint to fm_coefficient_results/fm_coefficient_3_hidden.pth
with keys: model_state_dict, train_mean, train_std, fm_294k, config, best_epoch, best_val_loss, final_val_loss
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
import sys

# ---------------- CONFIG ----------------
class Config:
    # Data paths - adjust if necessary
    INPUT_TEMPS_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/input_temps_final.npy")
    FM_NORMALIZED_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/output_fm_normalized_final.npy")
    FM_294K_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/output_fm_normalized294k.npy")

    # Grid / network sizes
    N_ROWS = 17
    N_COLS = 17
    N_CELLS = N_ROWS * N_COLS             # 289
    INPUT_SIZE = N_CELLS * 2              # 578 (fuel + other)
    HIDDEN_SIZES = [1024, 512, 1024]      # explicit hidden sizes
    OUTPUT_SIZE = N_CELLS * N_CELLS       # 83521

    DROPOUT_RATE = 0.2
    USE_BATCH_NORM = True

    # Training hyperparams
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    MAX_EPOCHS = 10
    VALIDATION_SPLIT = 0.15
    EARLY_STOP_PATIENCE = 15
    LR_REDUCE_PATIENCE = 10

    # Output
    RESULTS_DIR = Path("fm_coefficient_results")
    MODEL_SAVE_PATH = "fm_coefficient_3_hidden.pth"
    LOG_FILE = "training_log_3hidden.json"

    EPSILON = 1e-8  # numeric safety

# ---------------- utilities ----------------
def load_keff_values(path):
    arr = np.load(path, allow_pickle=True)
    keffs = []
    for val in arr:
        try:
            if hasattr(val, "nominal_value"):
                keffs.append(float(val.nominal_value))
            else:
                keffs.append(float(val))
        except Exception:
            keffs.append(np.nan)
    return np.array(keffs, dtype=float)

def load_fm_294k(path, n_cells):
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        if raw.ndim == 0:
            raw = raw.item()
        elif raw.size == 1:
            raw = raw.flatten()[0]
    fm = np.asarray(raw, dtype=float)
    while fm.ndim > 2:
        fm = np.squeeze(fm)
    if fm.ndim == 1:
        side = int(np.sqrt(fm.size))
        if side * side != fm.size:
            raise ValueError(f"FM size {fm.size} is not a perfect square")
        fm = fm.reshape(side, side)
    if fm.ndim != 2 or fm.shape[0] != fm.shape[1]:
        raise ValueError(f"Invalid FM shape: {fm.shape}")
    if fm.shape[0] != n_cells:
        raise ValueError(f"FM shape {fm.shape} doesn't match expected n_cells {n_cells}")
    return fm

def make_features(temp_tuple, n_cells):
    fuel_vec, other_vec = temp_tuple
    fuel_grid = np.zeros(n_cells, dtype=np.float32)
    fuel_idx = 0
    for i in range(n_cells):
        if fuel_idx < len(fuel_vec) and getattr(fuel_vec, 'shape', None) is not None:
            try:
                val = float(fuel_vec[fuel_idx])
            except Exception:
                val = 0.0
            if val > 0:
                fuel_grid[i] = val
                fuel_idx += 1
            else:
                pass
        else:
            break
    other_grid = np.asarray(other_vec, dtype=np.float32).flatten()
    if other_grid.size != n_cells:
        raise ValueError(f"other_vec size {other_grid.size} != {n_cells}")
    return np.concatenate([fuel_grid, other_grid])

# ---------------- data loading & processing ----------------
def load_and_prepare_data(cfg: Config):
    print("Loading data files...")
    input_temps = np.load(cfg.INPUT_TEMPS_PATH, allow_pickle=True)
    print(f"  Loaded {len(input_temps)} temperature samples")
    fm_samples = np.load(cfg.FM_NORMALIZED_PATH, allow_pickle=True)
    print(f"  Loaded {len(fm_samples)} fission matrix samples")
    fm_294k = load_fm_294k(cfg.FM_294K_PATH, cfg.N_CELLS)
    print(f"  Loaded 294K reference FM shape: {fm_294k.shape}")

    n_pairs = min(len(input_temps), len(fm_samples))
    print(f"  Preparing up to {n_pairs} pairs")

    features = []
    targets = []
    skipped = []
    fm_ref_safe = fm_294k + cfg.EPSILON

    for idx in range(n_pairs):
        try:
            feat = make_features(input_temps[idx], cfg.N_CELLS)
        except Exception as e:
            skipped.append((idx, f"feature_error:{e}"))
            continue

        raw_fm = fm_samples[idx]
        try:
            fm_arr = np.asarray(raw_fm, dtype=float)
        except Exception as e:
            skipped.append((idx, f"fm_coercion_error:{e}"))
            continue

        # reshape if necessary
        if fm_arr.ndim == 1:
            if fm_arr.size == cfg.N_CELLS * cfg.N_CELLS:
                fm_arr = fm_arr.reshape(cfg.N_CELLS, cfg.N_CELLS)
            else:
                skipped.append((idx, f"fm_1d_wrong_size:{fm_arr.size}"))
                continue
        elif fm_arr.ndim == 2:
            if fm_arr.shape != (cfg.N_CELLS, cfg.N_CELLS):
                if fm_arr.size == cfg.N_CELLS * cfg.N_CELLS:
                    fm_arr = fm_arr.reshape(cfg.N_CELLS, cfg.N_CELLS)
                else:
                    skipped.append((idx, f"fm_2d_wrong_shape:{fm_arr.shape}"))
                    continue
        else:
            skipped.append((idx, f"fm_ndim={fm_arr.ndim}"))
            continue

        try:
            coeff = (fm_arr + cfg.EPSILON) / fm_ref_safe
            ln_coeff = np.log(coeff)
            ln_flat = np.nan_to_num(ln_coeff.flatten(), nan=0.0, posinf=10.0, neginf=-10.0)
        except Exception as e:
            skipped.append((idx, f"log_error:{e}"))
            continue

        features.append(feat.astype(np.float32))
        targets.append(ln_flat.astype(np.float32))

    if skipped:
        print(f"[WARN] Skipped {len(skipped)} samples. Examples:")
        for i, reason in skipped[:10]:
            print(f"  idx={i}: {reason}")

    if len(features) == 0:
        raise RuntimeError("No valid data after cleaning")

    X = np.stack(features, axis=0)
    Y = np.stack(targets, axis=0)
    print(f"Final dataset shapes: X={X.shape}, Y={Y.shape}")
    return X, Y, fm_294k

# ---------------- explicit sequential network (network.* keys) ----------------
class FMCoefficientPredictor(nn.Module):
    """
    Sequential architecture that will produce keys like 'network.0.weight', etc.
    578 -> 1024 -> BN? -> ReLU -> Dropout -> 512 -> BN? -> ReLU -> Dropout -> 1024 -> BN? -> ReLU -> Dropout -> 83521
    """
    def __init__(self, cfg: Config):
        super().__init__()
        layers = []

        # block 1: Linear(578 -> 1024), BN?, ReLU, Dropout
        layers.append(nn.Linear(cfg.INPUT_SIZE, cfg.HIDDEN_SIZES[0]))
        if cfg.USE_BATCH_NORM:
            layers.append(nn.BatchNorm1d(cfg.HIDDEN_SIZES[0]))
        layers.append(nn.ReLU())
        if cfg.DROPOUT_RATE > 0:
            layers.append(nn.Dropout(cfg.DROPOUT_RATE))

        # block 2: Linear(1024 -> 512), BN?, ReLU, Dropout
        layers.append(nn.Linear(cfg.HIDDEN_SIZES[0], cfg.HIDDEN_SIZES[1]))
        if cfg.USE_BATCH_NORM:
            layers.append(nn.BatchNorm1d(cfg.HIDDEN_SIZES[1]))
        layers.append(nn.ReLU())
        if cfg.DROPOUT_RATE > 0:
            layers.append(nn.Dropout(cfg.DROPOUT_RATE))

        # block 3: Linear(512 -> 1024), BN?, ReLU, Dropout
        layers.append(nn.Linear(cfg.HIDDEN_SIZES[1], cfg.HIDDEN_SIZES[2]))
        if cfg.USE_BATCH_NORM:
            layers.append(nn.BatchNorm1d(cfg.HIDDEN_SIZES[2]))
        layers.append(nn.ReLU())
        if cfg.DROPOUT_RATE > 0:
            layers.append(nn.Dropout(cfg.DROPOUT_RATE))

        # output
        layers.append(nn.Linear(cfg.HIDDEN_SIZES[2], cfg.OUTPUT_SIZE))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ---------------- training utilities (same pattern as working trainer) ----------------
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

    def save(self, cfg: Config):
        with open(self.log_dir / cfg.LOG_FILE, 'w') as f:
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
    return total_loss / max(1, n_batches)

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
    return total_loss / max(1, n_batches)

def plot_training_history(logger, save_path):
    epochs = [e["epoch"] for e in logger.metrics["epochs"]]
    train_losses = [e["train_loss"] for e in logger.metrics["epochs"]]
    val_losses = [e["val_loss"] for e in logger.metrics["epochs"]]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved training curve to {save_path}")

# ---------------- main training loop ----------------
def main():
    cfg = Config()
    cfg.RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    print("="*80)
    print("TRAIN: FISSION MATRIX COEFFICIENT PREDICTOR (3-hidden explicit sequential)")
    print("="*80)
    print(f"Architecture: {cfg.INPUT_SIZE} -> {' -> '.join(map(str, cfg.HIDDEN_SIZES))} -> {cfg.OUTPUT_SIZE}")
    print(f"Batch norm: {cfg.USE_BATCH_NORM}, Dropout: {cfg.DROPOUT_RATE}")
    print("="*80 + "\n")

    X, Y, fm_294k = load_and_prepare_data(cfg)
    n_samples = X.shape[0]
    n_val = int(n_samples * cfg.VALIDATION_SPLIT)
    n_train = n_samples - n_val

    idx = np.random.permutation(n_samples)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_X = X[train_idx]; train_Y = Y[train_idx]
    val_X = X[val_idx]; val_Y = Y[val_idx]

    print(f"Train/Val split: {n_train} / {n_val}")

    # normalize using train stats
    train_mean = train_X.mean(axis=0)
    train_std = train_X.std(axis=0) + 1e-8
    train_X_norm = (train_X - train_mean) / train_std
    val_X_norm = (val_X - train_mean) / train_std

    train_ds = TensorDataset(torch.from_numpy(train_X_norm).float(), torch.from_numpy(train_Y).float())
    val_ds = TensorDataset(torch.from_numpy(val_X_norm).float(), torch.from_numpy(val_Y).float())
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FMCoefficientPredictor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}, device: {device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=cfg.LR_REDUCE_PATIENCE, verbose=True)

    logger = TrainingLogger(cfg.RESULTS_DIR)
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        logger.log_epoch(epoch, train_loss, val_loss, current_lr)

        print(f"Epoch {epoch:3d}/{cfg.MAX_EPOCHS} [{epoch_time:.1f}s] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ New best validation loss!")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.EARLY_STOP_PATIENCE:
            print(f"\n{'='*80}\nEarly stopping: No improvement for {cfg.EARLY_STOP_PATIENCE} epochs\n{'='*80}")
            break

        if epoch % 10 == 0:
            logger.save(cfg)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Restored best model from epoch {best_epoch}")

    final_val_loss = validate(model, val_loader, criterion, device)
    print(f"\nFinal validation loss: {final_val_loss:.6f}")

    # sample prediction test
    print("\n" + "="*80)
    print("SAMPLE PREDICTION TEST")
    print("="*80)
    model.eval()
    with torch.no_grad():
        test_input = torch.FloatTensor(val_X_norm[0:1]).to(device)
        test_output = model(test_input).cpu().numpy().reshape(cfg.N_CELLS, cfg.N_CELLS)
        test_coefficients = np.exp(test_output)
        print(f"Predicted ln(coefficient) stats: mean {test_output.mean():.4f}, std {test_output.std():.4f}, range [{test_output.min():.4f}, {test_output.max():.4f}]")
        print(f"Predicted coefficient stats: mean {test_coefficients.mean():.4f}, std {test_coefficients.std():.4f}, range [{test_coefficients.min():.4f}, {test_coefficients.max():.4f}]")

    # Save final model and metadata
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
    logger.save(cfg)
    plot_training_history(logger, cfg.RESULTS_DIR / "training_curve_3hidden.png")
    logger.print_summary()
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
