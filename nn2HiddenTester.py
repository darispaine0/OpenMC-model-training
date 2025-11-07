#!/usr/bin/env python3
"""
sweep_two_hidden_random_grid.py

Randomized grid sweep of two-hidden-layer networks:
  input (578) -> hidden1 (h1) -> hidden2 (h2) -> output (83521)

- Sample n1 distinct integers in [min_neurons,max_neurons] for hidden1
- Sample n2 distinct integers in [min_neurons,max_neurons] for hidden2
- Evaluate all pairs (n1 * n2) networks (default 100x100 = 10,000)
- Train each model once (seeded), compute mean-absolute PCM vs OpenMC keff (if available)
- Save intermediate results to allow resume, and produce a heatmap at the end

WARNING: This can be extremely expensive. Use --quick to test/preview.
"""
import argparse
import json
from pathlib import Path
import time
from datetime import datetime
import math
import traceback
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

# ---------------- User-configurable paths / defaults ----------------
INPUT_TEMPS_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/input_temps_final.npy")
FM_NORMALIZED_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/output_fm_normalized_final.npy")
FM_294K_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/output_fm_normalized294k.npy")
KEFF_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/output_keff_final.npy")

# default results root
DEFAULT_RESULTS_ROOT = Path("fm_coefficient_results")

# grid / problem sizes
N_ROWS = 17
N_COLS = 17
N_CELLS = N_ROWS * N_COLS
INPUT_SIZE = N_CELLS * 2
OUTPUT_SIZE = N_CELLS * N_CELLS

# training defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_MAX_EPOCHS = 10
DEFAULT_VALIDATION_SPLIT = 0.15
DEFAULT_EARLY_STOP = 3
DEFAULT_LR_REDUCE_PATIENCE = 8

EPSILON = 1e-8
LN_CLIP_MIN = -10.0
LN_CLIP_MAX = 10.0

# ---------------- Utility functions (robust loading / preprocessing) ----------------
def robust_load_fm294(path: Path):
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        if raw.ndim == 0:
            raw = raw.item()
        elif raw.size == 1:
            raw = raw.flatten()[0]
    fm = np.asarray(raw, dtype=float)
    # squeeze extras
    while fm.ndim > 2:
        fm = np.squeeze(fm)
    if fm.ndim == 1:
        side = int(math.isqrt(fm.size))
        if side * side != fm.size:
            raise ValueError(f"fm_294k 1D size {fm.size} not perfect square")
        fm = fm.reshape(side, side)
    if fm.ndim != 2 or fm.shape[0] != fm.shape[1]:
        raise ValueError(f"fm_294k invalid shape: {fm.shape}")
    return fm

def load_keff_values(path: Path):
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=True)
    keffs = []
    for v in arr:
        try:
            if hasattr(v, "nominal_value"):
                keffs.append(float(v.nominal_value))
            else:
                keffs.append(float(v))
        except Exception:
            keffs.append(np.nan)
    return np.array(keffs, dtype=float)

def make_feature_from_input(temp_tuple, n_cells=N_CELLS):
    try:
        fuel_vec, other_vec = temp_tuple
    except Exception:
        raise ValueError("input sample not in expected (fuel_vec, other_vec) form")
    fuel_grid = np.zeros(n_cells, dtype=np.float32)
    fuel_idx = 0
    for i in range(n_cells):
        if fuel_idx < len(fuel_vec):
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
        raise ValueError(f"other_vec size {other_grid.size} != expected {n_cells}")
    return np.concatenate([fuel_grid, other_grid])

def prepare_features_and_targets():
    """Load inputs & FMs, build features and ln(coefficient) targets (for training)."""
    print("Loading raw arrays...")
    inputs = np.load(INPUT_TEMPS_PATH, allow_pickle=True)
    fms = np.load(FM_NORMALIZED_PATH, allow_pickle=True)
    fm_294k = robust_load_fm294(FM_294K_PATH)
    print(f"Loaded: inputs={len(inputs)}, fms={len(fms)}, fm_294k shape={fm_294k.shape}")

    n_pairs = min(len(inputs), len(fms))
    print(f"Preparing up to {n_pairs} samples...")

    fm_ref_safe = fm_294k + EPSILON
    features = []
    targets = []
    valid_indices = []
    skipped = []

    for i in range(n_pairs):
        try:
            feat = make_feature_from_input(inputs[i], N_CELLS)
        except Exception as e:
            skipped.append((i, f"feature_err:{e}"))
            continue
        raw_fm = fms[i]
        try:
            fm_arr = np.asarray(raw_fm, dtype=float)
        except Exception as e:
            skipped.append((i, f"fm_cast_err:{e}"))
            continue
        if fm_arr.ndim == 1:
            if fm_arr.size == N_CELLS * N_CELLS:
                fm_arr = fm_arr.reshape(N_CELLS, N_CELLS)
            else:
                skipped.append((i, f"fm_1d_size:{fm_arr.size}"))
                continue
        elif fm_arr.ndim == 2:
            if fm_arr.shape != (N_CELLS, N_CELLS):
                if fm_arr.size == N_CELLS * N_CELLS:
                    fm_arr = fm_arr.reshape(N_CELLS, N_CELLS)
                else:
                    skipped.append((i, f"fm_2d_shape:{fm_arr.shape}"))
                    continue
        else:
            skipped.append((i, f"fm_ndim:{fm_arr.ndim}"))
            continue

        try:
            coeff = (fm_arr + EPSILON) / fm_ref_safe
            ln_coeff = np.log(coeff)
            ln_flat = np.nan_to_num(ln_coeff.flatten(), nan=0.0, posinf=10.0, neginf=-10.0)
        except Exception as e:
            skipped.append((i, f"log_err:{e}"))
            continue

        features.append(feat.astype(np.float32))
        targets.append(ln_flat.astype(np.float32))
        valid_indices.append(i)

    if skipped:
        print(f"[WARN] Skipped {len(skipped)} samples. Example skips: {skipped[:8]}")
    if len(features) == 0:
        raise RuntimeError("No valid samples after cleaning.")

    X = np.stack(features, axis=0)
    Y = np.stack(targets, axis=0)
    print(f"Prepared dataset: X.shape={X.shape}, Y.shape={Y.shape}")
    return X, Y, fm_294k, valid_indices

# ---------------- Two-hidden-layer model ----------------
class TwoHiddenFMPredictor(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, h1=256, h2=256, output_size=OUTPUT_SIZE):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_size)
        )
    def forward(self, x):
        return self.network(x)

# ---------------- Power iteration ----------------
def power_iteration_user(fm, tol=1e-6, max_iter=500):
    fm = np.asarray(fm, dtype=float)
    n = fm.shape[0]
    v = np.ones((n,1))
    eig_val = 1.0
    for i in range(max_iter):
        v_new = fm.dot(v)
        eig_new = float(np.max(np.abs(v_new)))
        if eig_new == 0:
            eig_val = 0.0
            v = v_new
            break
        v = v_new / eig_new
        if i>0 and abs(eig_new - eig_val) < tol:
            eig_val = eig_new
            break
        eig_val = eig_new
    v = v.ravel()
    s = v.sum()
    if s != 0:
        v = v / s
    return float(eig_val), v

# ---------------- Train / evaluate helpers ----------------
def train_model_two_hidden(X, Y, h1, h2, seed, device,
                           batch_size=DEFAULT_BATCH_SIZE,
                           lr=DEFAULT_LR,
                           weight_decay=DEFAULT_WEIGHT_DECAY,
                           max_epochs=DEFAULT_MAX_EPOCHS,
                           val_split=DEFAULT_VALIDATION_SPLIT,
                           early_stop=DEFAULT_EARLY_STOP,
                           lr_reduce_patience=DEFAULT_LR_REDUCE_PATIENCE,
                           verbose=False):
    """Train a two-hidden-layer model and return (model, train_mean, train_std, best_val_loss, best_epoch)."""
    # deterministic-ish
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    n_samples = X.shape[0]
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    perm = np.random.permutation(n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_X = X[train_idx]; train_Y = Y[train_idx]
    val_X = X[val_idx]; val_Y = Y[val_idx]

    train_mean = train_X.mean(axis=0)
    train_std = train_X.std(axis=0) + 1e-8
    train_X_norm = (train_X - train_mean) / train_std
    val_X_norm = (val_X - train_mean) / train_std

    train_ds = TensorDataset(torch.from_numpy(train_X_norm).float(), torch.from_numpy(train_Y).float())
    val_ds = TensorDataset(torch.from_numpy(val_X_norm).float(), torch.from_numpy(val_Y).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TwoHiddenFMPredictor(INPUT_SIZE, h1, h2, OUTPUT_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_reduce_patience, verbose=False)

    best_val = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs+1):
        # train
        model.train()
        total_loss = 0.0
        nb = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()); nb += 1
        train_loss = total_loss / max(1, nb)

        # val
        model.eval()
        val_loss_sum = 0.0
        nbv = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss_sum += float(loss.item()); nbv += 1
        val_loss = val_loss_sum / max(1, nbv)

        scheduler.step(val_loss)

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_mean, train_std, best_val, best_epoch

def evaluate_model_pcm(model, train_mean, train_std, X_all, fm_294k, valid_indices, keff_array, device, batch_size=DEFAULT_BATCH_SIZE):
    """Run batched inference on X_all, reconstruct FMs, compute predicted keff and PCM errors."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X_all.shape[0], batch_size):
            end = min(start + batch_size, X_all.shape[0])
            Xb = X_all[start:end]
            Xn = (Xb - train_mean) / train_std
            Xt = torch.from_numpy(Xn).float().to(device)
            out = model(Xt).cpu().numpy()
            preds.append(out)
    preds = np.vstack(preds)

    M = preds.shape[0]
    pcm_arr = np.full(M, np.nan, dtype=float)
    kfm_arr = np.full(M, np.nan, dtype=float)

    for local_idx, global_idx in enumerate(valid_indices):
        try:
            ln_flat = preds[local_idx].reshape(N_CELLS, N_CELLS)
        except Exception:
            pcm_arr[local_idx] = np.nan
            continue
        ln_flat = np.clip(ln_flat, LN_CLIP_MIN, LN_CLIP_MAX)
        coeff = np.exp(ln_flat)
        fm_pred = fm_294k * coeff
        k_pred, _ = power_iteration_user(fm_pred)
        kfm_arr[local_idx] = k_pred

        if keff_array is not None and global_idx < keff_array.size:
            k_true = keff_array[global_idx]
            if np.isfinite(k_pred) and np.isfinite(k_true) and k_true != 0:
                pcm_arr[local_idx] = (k_pred / k_true - 1.0) * 1e5
            else:
                pcm_arr[local_idx] = np.nan
        else:
            pcm_arr[local_idx] = np.nan

    finite_pcm = pcm_arr[np.isfinite(pcm_arr)]
    if finite_pcm.size > 0:
        mean_abs_pcm = float(np.mean(np.abs(finite_pcm)))
    else:
        mean_abs_pcm = float('nan')
    return pcm_arr, kfm_arr, mean_abs_pcm

# ---------------- Main sweep orchestration ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Randomized grid sweep for two-hidden-layer architectures.")
    p.add_argument("--n1", type=int, default=100, help="How many random values for hidden layer 1 (default 100)")
    p.add_argument("--n2", type=int, default=100, help="How many random values for hidden layer 2 (default 100)")
    p.add_argument("--min-neurons", type=int, default=200, help="Minimum neurons (inclusive)")
    p.add_argument("--max-neurons", type=int, default=500, help="Maximum neurons (inclusive)")
    p.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS, help="Max epochs per training")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    p.add_argument("--outdir", type=str, default=str(DEFAULT_RESULTS_ROOT), help="Results root directory")
    p.add_argument("--seed", type=int, default=12345, help="Random seed for sampling neuron counts and reproducibility")
    p.add_argument("--quick", action="store_true", help="Quick mode: reduce epochs/early-stop and skip saving checkpoints (for smoke test)")
    p.add_argument("--save-checkpoints", action="store_true", default=False, help="Save individual model checkpoints (very large IO for 10k models)")
    p.add_argument("--resume", action="store_true", help="Resume from existing result file (if present)")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = outdir / f"sweep_2hidden_random_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    n1 = args.n1
    n2 = args.n2
    mn = args.min_neurons
    mx = args.max_neurons
    seed = int(args.seed)
    device = torch.device(args.device if args.device in ("cpu","cuda") else ("cuda" if torch.cuda.is_available() else "cpu"))
    batch = args.batch
    max_epochs = args.max_epochs
    save_checkpoints = args.save_checkpoints
    resume = args.resume

    if args.quick:
        print("[INFO] Quick mode enabled: reducing max epochs and early-stop to speed up runs.")
        max_epochs = min(30, max_epochs)
        early_stop = 5
    else:
        early_stop = DEFAULT_EARLY_STOP

    print("Sweep settings:")
    print(f"  n1={n1}, n2={n2}, range=[{mn},{mx}], total models ~ {n1*n2}")
    print(f"  device={device}, batch={batch}, max_epochs={max_epochs}, early_stop={early_stop}")
    print(f"  results -> {sweep_dir}")
    print("WARNING: This sweep can be extremely time-consuming. Use --quick for a smoke test.")

    # prepare dataset once
    X_all, Y_all, fm_294k, valid_indices = prepare_features_and_targets()
    keff_array = load_keff_values(KEFF_PATH)
    if keff_array is None:
        print("[WARN] KEFF file not found; PCM comparisons will be NaN.")

    # sample neuron counts (allow duplicates) - but user asked randomized; sample without replacement if (mx-mn+1)>=n
    rng = np.random.RandomState(seed)
    population = np.arange(mn, mx+1)
    if len(population) >= n1:
        h1_list = rng.choice(population, size=n1, replace=False)
    else:
        h1_list = rng.choice(population, size=n1, replace=True)
    if len(population) >= n2:
        h2_list = rng.choice(population, size=n2, replace=False)
    else:
        h2_list = rng.choice(population, size=n2, replace=True)

    # sort lists for nicer plotting (optional) - keep mapping so matrix axes correspond
    h1_list = np.array(sorted(h1_list))
    h2_list = np.array(sorted(h2_list))

    print(f"Sampled hidden1 values (n={len(h1_list)}): min={h1_list.min()}, max={h1_list.max()}")
    print(f"Sampled hidden2 values (n={len(h2_list)}): min={h2_list.min()}, max={h2_list.max()}")

    # prepare grid matrix to store mean_abs_pcm for each combination
    pcm_grid_path = sweep_dir / "pcm_grid.npy"
    meta_path = sweep_dir / "grid_meta.json"
    if resume and pcm_grid_path.exists():
        print("[INFO] Resuming from existing pcm_grid.npy")
        pcm_grid = np.load(pcm_grid_path)
        if pcm_grid.shape != (len(h2_list), len(h1_list)):
            print("[WARN] Existing pcm_grid shape mismatch -> reinitializing.")
            pcm_grid = np.full((len(h2_list), len(h1_list)), np.nan, dtype=float)
    else:
        pcm_grid = np.full((len(h2_list), len(h1_list)), np.nan, dtype=float)

    # save metadata (so we know mapping)
    meta = {
        "h1_list": h1_list.tolist(),
        "h2_list": h2_list.tolist(),
        "timestamp": timestamp,
        "args": vars(args)
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total = len(h1_list) * len(h2_list)
    print(f"Beginning sweep of {total} models... (this may take a long time)")

    # iterate combinations
    count = 0
    start_time = time.time()
    for j, h1 in enumerate(h1_list):         # columns (x)
        for i, h2 in enumerate(h2_list):     # rows (y)
            count += 1
            if np.isfinite(pcm_grid[i, j]):
                # already computed (resume)
                if count % 200 == 0:
                    elapsed = time.time() - start_time
                    print(f"[{count}/{total}] skipping completed ({h1},{h2}) - elapsed {elapsed/60:.1f} min")
                continue

            seed_model = seed + i * 100000 + j * 1000
            model_dir = sweep_dir / f"h1_{h1}_h2_{h2}"
            model_dir.mkdir(parents=True, exist_ok=True)

            try:
                model, train_mean, train_std, best_val_loss, best_epoch = train_model_two_hidden(
                    X_all, Y_all, int(h1), int(h2), seed_model, device,
                    batch_size=batch,
                    lr=DEFAULT_LR,
                    weight_decay=DEFAULT_WEIGHT_DECAY,
                    max_epochs=max_epochs,
                    val_split=DEFAULT_VALIDATION_SPLIT,
                    early_stop=early_stop,
                    lr_reduce_patience=DEFAULT_LR_REDUCE_PATIENCE,
                    verbose=False
                )
            except Exception as e:
                print(f"[ERROR] training failed for (h1={h1}, h2={h2}) -> recording NaN. Error: {e}")
                traceback.print_exc()
                pcm_grid[i, j] = np.nan
                # save intermediate grid
                np.save(pcm_grid_path, pcm_grid)
                continue

            # evaluate
            try:
                pcm_arr, kfm_arr, mean_abs_pcm = evaluate_model_pcm(
                    model, train_mean, train_std, X_all, fm_294k, valid_indices, keff_array, device, batch_size=batch
                )
            except Exception as e:
                print(f"[ERROR] evaluation failed for (h1={h1}, h2={h2}) -> recording NaN. Error: {e}")
                traceback.print_exc()
                mean_abs_pcm = float('nan')

            pcm_grid[i, j] = mean_abs_pcm

            # optionally save checkpoint and arrays (user warned about IO)
            if save_checkpoints:
                ckpt = {
                    'model_state_dict': model.state_dict(),
                    'train_mean': train_mean,
                    'train_std': train_std,
                    'config': {'h1': int(h1), 'h2': int(h2), 'input_size': INPUT_SIZE, 'output_size': OUTPUT_SIZE},
                    'best_epoch': best_epoch,
                    'best_val_loss': float(best_val_loss) if best_val_loss is not None else None
                }
                torch.save(ckpt, model_dir / f"checkpoint_h1{h1}_h2{h2}.pth")
                np.save(model_dir / "pcm_array.npy", pcm_arr)
                np.save(model_dir / "kfm_array.npy", kfm_arr)
                with open(model_dir / "meta.json", "w") as f:
                    json.dump({'h1': int(h1), 'h2': int(h2), 'seed': seed_model, 'mean_abs_pcm': mean_abs_pcm, 'best_epoch': int(best_epoch)}, f, indent=2)

            # save intermediate grid every few iterations
            if count % 50 == 0 or count == total:
                np.save(pcm_grid_path, pcm_grid)
                elapsed = time.time() - start_time
                avg = elapsed / max(1, count)
                remaining = (total - count) * avg
                print(f"[{count}/{total}] saved partial grid. Elapsed {elapsed/60:.1f} min, est remaining {remaining/60:.1f} min")

    # final save
    np.save(pcm_grid_path, pcm_grid)
    print("Sweep complete. Saved pcm_grid to", pcm_grid_path)

    # Plot heatmap
    try:
        plt.figure(figsize=(12, 10))
        # imshow expects matrix with rows corresponding to y; our pcm_grid already maps h2 (rows) x h1 (cols)
        im = plt.imshow(pcm_grid, origin='lower', aspect='auto', interpolation='nearest')
        plt.colorbar(im, label='Mean absolute PCM (pcm)')
        plt.xlabel('Hidden layer 1 neurons (sampled values)')
        plt.ylabel('Hidden layer 2 neurons (sampled values)')
        # annotate ticks with a subset of neuron values
        nx = len(h1_list)
        ny = len(h2_list)
        xticks = np.linspace(0, nx-1, min(10, nx)).astype(int)
        yticks = np.linspace(0, ny-1, min(10, ny)).astype(int)
        plt.xticks(xticks, [str(h1_list[idx]) for idx in xticks], rotation=45)
        plt.yticks(yticks, [str(h2_list[idx]) for idx in yticks])
        plt.title(f"Sweep mean-abs-PCM (h1 in cols, h2 in rows) - {n1}x{n2} samples")
        plot_path = sweep_dir / "pcm_grid_heatmap.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        print("Saved heatmap to", plot_path)
    except Exception as e:
        print("Failed to create heatmap:", e)
        traceback.print_exc()

    # save final metadata & numpy arrays
    meta['pcm_grid_path'] = str(pcm_grid_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("All done. Results in:", sweep_dir)

if __name__ == "__main__":
    main()
