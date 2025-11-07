#!/usr/bin/env python3
"""
sweep_hidden_neurons.py

Train and evaluate single-hidden-layer models with hidden sizes from 600 down to 200 (step -10).
For each size, train `N_RUNS` times, evaluate PCM errors, compute mean-abs-PCM per run,
then aggregate per-size mean and std and plot mean_abs_pcm vs hidden neurons.

Saves results to fm_coefficient_results/sweep_{timestamp}/
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
import math
import csv
import sys
import traceback
import random

# ---------------- USER ADJUSTABLE CONFIG ----------------
class GlobalConfig:
    # Data paths (adjust if your files are somewhere else)
    INPUT_TEMPS_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/input_temps_final.npy")
    FM_NORMALIZED_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/output_fm_normalized_final.npy")
    FM_294K_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/output_fm_normalized294k.npy")
    KEFF_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/output_keff_final.npy")  # OpenMC keff values (optional)

    # Grid sizes
    N_ROWS = 17
    N_COLS = 17
    N_CELLS = N_ROWS * N_COLS          # 289
    INPUT_SIZE = N_CELLS * 2           # 578
    OUTPUT_SIZE = N_CELLS * N_CELLS    # 83521

    # Training hyperparams (kept same as your working trainer)
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    MAX_EPOCHS = 500
    VALIDATION_SPLIT = 0.15
    EARLY_STOP_PATIENCE = 15
    LR_REDUCE_PATIENCE = 15

    # Sweep settings
    HIDDEN_START = 435
    HIDDEN_STOP = 425   # inclusive lower bound
    HIDDEN_STEP = -1
    N_RUNS = 10          # runs per hidden-size

    # Output / bookkeeping
    RESULTS_ROOT = Path("fm_coefficient_results")
    SWEEP_SUBDIR = None  # filled at runtime
    SAVE_CHECKPOINTS = True

    # numeric safety
    EPSILON = 1e-8
    LN_CLIP_MIN = -10.0
    LN_CLIP_MAX = 10.0

# ---------------- Utilities (data loading / preprocessing) ----------------
def robust_load_fm294(path):
    raw = np.load(path, allow_pickle=True)
    # unwrap single-element object arrays
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        if raw.ndim == 0:
            raw = raw.item()
        elif raw.size == 1:
            raw = raw.flatten()[0]
    fm = np.asarray(raw, dtype=float)
    # squeeze extra dims
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

def make_feature_from_input(temp_tuple, n_cells):
    """Build 578-length feature from (fuel_vec, other_vec) - same logic as your working code"""
    fuel_vec, other_vec = temp_tuple
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
        raise ValueError(f"other_vec size {other_grid.size} != {n_cells}")
    return np.concatenate([fuel_grid, other_grid])

def prepare_dataset(cfg: GlobalConfig):
    """
    Loads input_temps, fm_samples and fm_294k and returns:
      features: (M, 578) numpy array
      targets_ln_flat: (M, 83521) numpy array (ln coeff flattened) -- used only if you want to train
      fm_294k: (289,289) array
      good_indices: list of original sample indices used
    This function follows your robust cleaning logic.
    """
    print("Loading raw arrays...")
    input_temps = np.load(cfg.INPUT_TEMPS_PATH, allow_pickle=True)
    fm_samples = np.load(cfg.FM_NORMALIZED_PATH, allow_pickle=True)
    fm_294k = robust_load_fm294(cfg.FM_294K_PATH)
    print(f"Loaded input_temps {len(input_temps)}, fm_samples {len(fm_samples)}, fm_294k shape {fm_294k.shape}")

    n_pairs = min(len(input_temps), len(fm_samples))
    print(f"Preparing up to {n_pairs} pairs...")

    fm_ref_safe = fm_294k + cfg.EPSILON
    features = []
    targets = []
    good_indices = []
    skipped = []

    for idx in range(n_pairs):
        # build feature
        try:
            feat = make_feature_from_input(input_temps[idx], cfg.N_CELLS)
        except Exception as e:
            skipped.append((idx, f"feature_err:{e}"))
            continue

        # load fm and shape-check
        raw_fm = fm_samples[idx]
        try:
            fm_arr = np.asarray(raw_fm, dtype=float)
        except Exception as e:
            skipped.append((idx, f"fm_coercion_err:{e}"))
            continue

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

        # compute ln(coeff) target
        try:
            coeff = (fm_arr + cfg.EPSILON) / fm_ref_safe
            ln_coeff = np.log(coeff)
            ln_flat = np.nan_to_num(ln_coeff.flatten(), nan=0.0, posinf=10.0, neginf=-10.0)
        except Exception as e:
            skipped.append((idx, f"log_err:{e}"))
            continue

        features.append(feat.astype(np.float32))
        targets.append(ln_flat.astype(np.float32))
        good_indices.append(idx)

    if len(skipped) > 0:
        print(f"[WARN] skipped {len(skipped)} samples (examples up to 10): {skipped[:10]}")

    if len(features) == 0:
        raise RuntimeError("No valid samples found after cleaning.")

    X = np.stack(features, axis=0)
    Y = np.stack(targets, axis=0)
    print(f"Prepared dataset: features {X.shape}, targets {Y.shape}, good_indices len {len(good_indices)}")
    return X, Y, fm_294k, good_indices

# ---------------- Model class (single-hidden Sequential to produce network.* keys) ----------------
class FMCoefficientPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.network(x)

# ---------------- Training / validation functions ----------------
def train_one_run(cfg: GlobalConfig, X, Y, seed, hidden_size, run_dir):
    """
    Train a single-model run and return:
      model (trained on best state), train_mean, train_std, best_val_loss, best_epoch, pcm_mean_abs_for_run
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # train/val split
    n_samples = X.shape[0]
    n_val = int(n_samples * cfg.VALIDATION_SPLIT)
    n_train = n_samples - n_val
    perm = np.random.permutation(n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_X = X[train_idx]; train_y = Y[train_idx]
    val_X = X[val_idx]; val_y = Y[val_idx]

    # normalization
    train_mean = train_X.mean(axis=0)
    train_std = train_X.std(axis=0) + 1e-8
    train_X_norm = (train_X - train_mean) / train_std
    val_X_norm = (val_X - train_mean) / train_std

    # dataloaders
    train_ds = TensorDataset(torch.from_numpy(train_X_norm).float(), torch.from_numpy(train_y).float())
    val_ds = TensorDataset(torch.from_numpy(val_X_norm).float(), torch.from_numpy(val_y).float())
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # model, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FMCoefficientPredictor(cfg.INPUT_SIZE, hidden_size, cfg.OUTPUT_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=cfg.LR_REDUCE_PATIENCE, verbose=False)

    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    best_state = None

    # training loop
    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        t0 = time.time()
        # train
        model.train()
        running_loss = 0.0
        nb = 0
        for bx, by in train_loader:
            bx = bx.to(device); by = by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()); nb += 1
        train_loss = running_loss / max(1, nb)

        # validate
        model.eval()
        running_val_loss = 0.0
        nbv = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device); by = by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                running_val_loss += float(loss.item()); nbv += 1
        val_loss = running_val_loss / max(1, nbv)

        scheduler.step(val_loss)
        # early stopping logic
        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.EARLY_STOP_PATIENCE:
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # optionally save checkpoint
    if cfg.SAVE_CHECKPOINTS:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "train_mean": train_mean,
            "train_std": train_std,
            "fm_294k": None,  # we'll save fm_294k elsewhere globally
            "config": {"INPUT_SIZE": cfg.INPUT_SIZE, "HIDDEN_SIZE": hidden_size, "OUTPUT_SIZE": cfg.OUTPUT_SIZE},
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss)
        }
        (run_dir / f"checkpoint_hidden{hidden_size}_seed{seed}.pth").parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, run_dir / f"checkpoint_hidden{hidden_size}_seed{seed}.pth")

    return model, train_mean, train_std, best_val_loss, best_epoch

# ---------------- Evaluation (compute PCM per sample) ----------------
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

def evaluate_model_pcm(cfg: GlobalConfig, model, train_mean, train_std, features_all, fm_294k, good_indices, keff_array, batch_size=32):
    """
    Run batched inference on features_all (shape M x 578), reconstruct FMs and compute PCM errors.
    Returns:
      pcm_array (length M) (nan where keff comparison not possible)
      mean_abs_pcm_value (float) computed over finite pcm entries (np.nan if none)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, features_all.shape[0], batch_size):
            end = min(start + batch_size, features_all.shape[0])
            Xb = features_all[start:end]
            Xn = (Xb - train_mean) / train_std
            Xt = torch.from_numpy(Xn).float().to(device)
            out = model(Xt).cpu().numpy()
            preds.append(out)
    preds = np.vstack(preds)  # (M, OUTPUT_SIZE)

    M = preds.shape[0]
    pcm_list = np.full(M, np.nan, dtype=float)
    kfm_list = np.full(M, np.nan, dtype=float)

    for local_i, global_idx in enumerate(good_indices):
        try:
            ln_flat = preds[local_i].reshape(cfg.N_CELLS, cfg.N_CELLS)
        except Exception as e:
            pcm_list[local_i] = np.nan
            continue
        ln_flat = np.clip(ln_flat, cfg.LN_CLIP_MIN, cfg.LN_CLIP_MAX)
        coeff = np.exp(ln_flat)
        fm_pred = fm_294k * coeff
        k_pred, _ = power_iteration_user(fm_pred)
        kfm_list[local_i] = k_pred

        if keff_array is not None:
            if global_idx < keff_array.size:
                k_true = keff_array[global_idx]
                if np.isfinite(k_pred) and np.isfinite(k_true) and k_true != 0:
                    pcm = (k_pred / k_true - 1.0) * 1e5
                    pcm_list[local_i] = pcm
                else:
                    pcm_list[local_i] = np.nan
            else:
                pcm_list[local_i] = np.nan
        else:
            pcm_list[local_i] = np.nan

    # compute mean absolute PCM across finite pcm elements
    finite_pcm = pcm_list[np.isfinite(pcm_list)]
    if finite_pcm.size > 0:
        mean_abs_pcm = float(np.mean(np.abs(finite_pcm)))
    else:
        mean_abs_pcm = float("nan")

    return pcm_list, kfm_list, mean_abs_pcm

# ---------------- Main sweep orchestration ----------------
def main():
    cfg = GlobalConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = cfg.RESULTS_ROOT / f"sweep_single_hidden_{timestamp}"
    cfg.SWEEP_SUBDIR = sweep_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print("Sweep output directory:", sweep_dir)

    # Prepare data once
    X_all, Y_all, fm_294k, good_indices = prepare_dataset(cfg)

    # Load keff array if available
    keff_array = None
    if cfg.KEFF_PATH.exists():
        try:
            keff_array = load_keff_values(cfg.KEFF_PATH)
            print(f"Loaded keff array length {keff_array.size}")
        except Exception as e:
            print(f"[WARN] Failed to load KEFF array: {e}")
            keff_array = None
    else:
        print("[WARN] KEFF file not found; PCM comparisons will be NaN.")

    neuron_values = list(range(cfg.HIDDEN_START, cfg.HIDDEN_STOP - 1, cfg.HIDDEN_STEP))
    results_summary = []  # list of dicts for CSV/printing

    for hidden_size in neuron_values:
        print("\n" + "="*80)
        print(f"Hidden size: {hidden_size} — running {cfg.N_RUNS} trainings")
        run_metrics = []   # per-run mean_abs_pcm
        run_pcms = []      # store per-run full pcm arrays (for optional analysis)
        run_kfms = []      # store per-run kfm arrays
        for run_idx in range(cfg.N_RUNS):
            seed = 1000 + hidden_size * 10 + run_idx  # deterministic varied seeds
            print(f"\n--- Training run {run_idx+1}/{cfg.N_RUNS} (seed={seed}) ---")
            run_subdir = sweep_dir / f"hidden_{hidden_size}" / f"run_{run_idx+1}"
            run_subdir.mkdir(parents=True, exist_ok=True)
            try:
                model, train_mean, train_std, best_val_loss, best_epoch = train_one_run(cfg, X_all, Y_all, seed, hidden_size, run_subdir)
            except Exception as e:
                print(f"[ERROR] training failed for hidden {hidden_size} run {run_idx+1}: {e}")
                traceback.print_exc()
                run_metrics.append(float("nan"))
                run_pcms.append(None)
                run_kfms.append(None)
                continue

            # Evaluate model (use features X_all, good_indices, fm_294k, keff_array)
            print("Evaluating trained model for PCM...")
            pcm_arr, kfm_arr, mean_abs_pcm = evaluate_model_pcm(cfg, model, train_mean, train_std, X_all, fm_294k, good_indices, keff_array, batch_size=cfg.BATCH_SIZE)
            print(f"  Run {run_idx+1} mean-abs-PCM: {mean_abs_pcm:.3f} pcm")
            run_metrics.append(mean_abs_pcm)
            run_pcms.append(pcm_arr)
            run_kfms.append(kfm_arr)

            # Save per-run outputs
            np.save(run_subdir / "pcm_array.npy", pcm_arr)
            np.save(run_subdir / "kfm_array.npy", kfm_arr)
            # save metadata
            meta = {
                "hidden_size": hidden_size,
                "run": run_idx+1,
                "seed": seed,
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss)
            }
            with open(run_subdir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

        # aggregate per-hidden-size
        run_metrics_np = np.array([v for v in run_metrics], dtype=float)
        valid = np.isfinite(run_metrics_np)
        if np.any(valid):
            mean_of_mean_abs_pcm = float(np.nanmean(run_metrics_np[valid]))
            std_of_mean_abs_pcm = float(np.nanstd(run_metrics_np[valid]))
        else:
            mean_of_mean_abs_pcm = float("nan")
            std_of_mean_abs_pcm = float("nan")

        print("\nSUMMARY for hidden_size =", hidden_size)
        print(f"  runs mean-abs-pcm values: {run_metrics}")
        print(f"  mean_of_mean_abs_pcm = {mean_of_mean_abs_pcm:.3f} pcm, std = {std_of_mean_abs_pcm:.3f} pcm")

        # persist summary row
        results_summary.append({
            "hidden_size": int(hidden_size),
            "mean_of_mean_abs_pcm": mean_of_mean_abs_pcm,
            "std_of_mean_abs_pcm": std_of_mean_abs_pcm,
            "per_run_mean_abs_pcm": run_metrics
        })

        # Save intermediate results CSV after each hidden_size
        csv_path = sweep_dir / "sweep_summary.csv"
        with open(csv_path, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["hidden_size", "mean_of_mean_abs_pcm", "std_of_mean_abs_pcm", "per_run_mean_abs_pcm"])
            for row in results_summary:
                writer.writerow([row["hidden_size"], row["mean_of_mean_abs_pcm"], row["std_of_mean_abs_pcm"], json.dumps(row["per_run_mean_abs_pcm"])])

    # final reporting + plotting
    hs = [r["hidden_size"] for r in results_summary]
    means = [r["mean_of_mean_abs_pcm"] for r in results_summary]
    stds = [r["std_of_mean_abs_pcm"] for r in results_summary]

    # save numeric arrays
    np.save(sweep_dir / "sweep_hidden_sizes.npy", np.array(hs))
    np.save(sweep_dir / "sweep_means.npy", np.array(means))
    np.save(sweep_dir / "sweep_stds.npy", np.array(stds))

    # Print a neat table
    print("\n" + "="*80)
    print("FINAL SWEEP SUMMARY")
    print("="*80)
    print("hidden_size | mean_abs_pcm (mean of runs) | std_of_runs")
    for h, m, s in zip(hs, means, stds):
        print(f"{h:10d} | {np.nan if np.isnan(m) else m:20.3f} | {np.nan if np.isnan(s) else s:10.3f}")

    # Plot
    plt.figure(figsize=(10,6))
    x = np.array(hs)
    y = np.array(means, dtype=float)
    yerr = np.array(stds, dtype=float)
    plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=4)
    plt.xlabel("Hidden layer neurons")
    plt.ylabel("Mean absolute PCM (pcm)")
    plt.title("Sweep: mean_abs_pcm vs hidden layer size (mean over runs)")
    plt.grid(True)
    plt.gca().invert_xaxis()  # optional: show decreasing neurons left->right
    plt.tight_layout()
    plot_path = sweep_dir / "sweep_pcm_vs_hidden.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved plot to {plot_path}")

    # Save full summary JSON
    with open(sweep_dir / "sweep_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\nSweep complete. Results directory:", sweep_dir)

if __name__ == "__main__":
    main()
