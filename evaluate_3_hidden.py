#!/usr/bin/env python3
"""
evaluate_3_hidden.py

Load trained 3-hidden FM coefficient model (nn.Sequential 'network'), run inference
over input_temps, reconstruct FM matrices via 294K reference, compute dominant eigenvalue
(power iteration), and compute PCM errors vs OpenMC keff (if available).
Saves predicted_kfm.npy, pcm_errors.npy, valid_indices.npy into fm_coefficient_results/.
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import math
import sys

# ---------------- Config / Paths ----------------
RESULTS_DIR = Path("fm_coefficient_results")
CHECKPOINT_PATH = RESULTS_DIR / "fm_coefficient_3_hidden.pth"

INPUT_TEMPS_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/input_temps_final.npy")
FM_NORMALIZED_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/output_fm_normalized_final.npy")
FM_294K_CANDIDATES = [
    Path("/home/daris/Monte_Carlo/Test/training_data/output_fm_normalized294k.npy"),
    Path("training_data/output_fm_normalized294k.npy"),
    Path("training_data/output_fm_normalized_294k.npy"),
    Path("training_data/output_fm_normalized294K.npy"),
]
KEFF_CANDIDATES = [
    Path("/home/daris/Monte_Carlo/Test/training_data/output_keff_final.npy"),
    Path("training_data/output_keff_final.npy"),
    Path("results/consolidated_datasets/train_output_keff.npy"),
    Path("results/consolidated_datasets/output_keff.npy")
]

LN_CLIP_MIN = -10.0
LN_CLIP_MAX = 10.0
BATCH = 16

# ---------------- Utilities ----------------
def find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def load_keff_array(path):
    arr = np.load(path, allow_pickle=True)
    vals = []
    for e in arr:
        try:
            if hasattr(e, "nominal_value"):
                kv = float(e.nominal_value)
            else:
                kv = float(e)
            vals.append(kv)
        except Exception:
            vals.append(np.nan)
    return np.array(vals, dtype=float)

def make_features_from_input_tuple(entry, N_CELLS=289):
    fuel_vec, other_vec = entry
    fuel_grid = np.zeros(N_CELLS, dtype=np.float32)
    fuel_idx = 0
    for i in range(N_CELLS):
        if fuel_idx < len(fuel_vec):
            try:
                val = float(fuel_vec[fuel_idx])
            except Exception:
                val = 0.0
            if val > 0:
                fuel_grid[i] = val
                fuel_idx += 1
    other_grid = np.asarray(other_vec, dtype=np.float32).flatten()
    if other_grid.size != N_CELLS:
        raise ValueError(f"other_vec size {other_grid.size} != N_CELLS {N_CELLS}")
    return np.concatenate([fuel_grid, other_grid])

def power_iteration_user(fm):
    fm = np.asarray(fm, dtype=float)
    size = fm.shape[0]
    eig_value = 1.0
    eig_vector = np.ones((size, 1))
    tolerance, max_iter = 1e-6, 500
    for i_iter in range(max_iter):
        eig_vector_new = np.dot(fm, eig_vector)
        eig_value_new = np.max(np.abs(eig_vector_new))
        if eig_value_new == 0:
            eig_vector = eig_vector_new
            eig_value = eig_value_new
            break
        eig_vector = eig_vector_new / eig_value_new
        if i_iter > 0 and abs(eig_value_new - eig_value) < tolerance:
            eig_value = eig_value_new
            break
        eig_value = eig_value_new
    s = eig_vector.sum()
    if s != 0:
        eig_vector = eig_vector / s
    return float(eig_value), eig_vector.ravel()

# ---------------- Model class matching trainer (nn.Sequential network) ----------------
class FMCoefficientPredictor(nn.Module):
    def __init__(self, input_size=578, hidden_sizes=(1024,512,1024), output_size=289*289,
                 dropout_rate=0.2, use_batch_norm=True):
        super().__init__()
        layers = []
        # block1
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        # block2
        layers.append(nn.Linear(hidden_sizes[0], hidden_sizes[1]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[1]))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        # block3
        layers.append(nn.Linear(hidden_sizes[1], hidden_sizes[2]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[2]))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        # output
        layers.append(nn.Linear(hidden_sizes[2], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ---------------- Main evaluation ----------------
def main():
    if not CHECKPOINT_PATH.exists():
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)
    if not INPUT_TEMPS_PATH.exists():
        print(f"[ERROR] Input temps not found: {INPUT_TEMPS_PATH}")
        sys.exit(1)

    print("Loading checkpoint:", CHECKPOINT_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    # Extract saved items
    if "model_state_dict" not in ckpt:
        print("[ERROR] checkpoint missing 'model_state_dict'")
        sys.exit(1)
    state = ckpt["model_state_dict"]
    train_mean = np.asarray(ckpt.get("train_mean"), dtype=float) if "train_mean" in ckpt else None
    train_std = np.asarray(ckpt.get("train_std"), dtype=float) if "train_std" in ckpt else None
    fm_294k_saved = ckpt.get("fm_294k", None)
    cfg_saved = ckpt.get("config", None)

    # Determine architecture from config if present, else assume default layout
    if cfg_saved and isinstance(cfg_saved, dict):
        N_CELLS = int(cfg_saved.get("N_CELLS", 289))
        INPUT_SIZE = int(cfg_saved.get("INPUT_SIZE", N_CELLS * 2))
        HIDDEN_SIZES = tuple(cfg_saved.get("HIDDEN_SIZES", [1024, 512, 1024]))
        OUTPUT_SIZE = int(cfg_saved.get("OUTPUT_SIZE", N_CELLS * N_CELLS))
        DROPOUT_RATE = float(cfg_saved.get("DROPOUT_RATE", 0.2))
        USE_BATCH_NORM = bool(cfg_saved.get("USE_BATCH_NORM", True))
        print("Using architecture from checkpoint config.")
    else:
        N_CELLS = 289
        INPUT_SIZE = 578
        HIDDEN_SIZES = (1024, 512, 1024)
        OUTPUT_SIZE = N_CELLS * N_CELLS
        DROPOUT_RATE = 0.2
        USE_BATCH_NORM = True
        print("No config in checkpoint; using default 3-hidden architecture.")

    # Load or resolve fm_294k
    if fm_294k_saved is not None:
        fm_294k = np.asarray(fm_294k_saved, dtype=float)
    else:
        fm294k_path = find_first_existing(FM_294K_CANDIDATES)
        if fm294k_path is None:
            print("[ERROR] Could not find fm_294k reference file in candidates.")
            sys.exit(1)
        raw = np.load(fm294k_path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.dtype == object:
            if raw.ndim == 0:
                raw = raw.item()
            elif raw.size == 1:
                raw = raw.flatten()[0]
        fm_294k = np.asarray(raw, dtype=float)

    # Normalize fm_294k shape
    if fm_294k.ndim > 2:
        fm_294k = np.squeeze(fm_294k)
    if fm_294k.ndim == 1:
        side = int(math.isqrt(fm_294k.size))
        if side * side != fm_294k.size:
            raise ValueError("fm_294k 1D size not perfect square")
        fm_294k = fm_294k.reshape(side, side)
    if fm_294k.ndim != 2 or fm_294k.shape[0] != fm_294k.shape[1]:
        raise ValueError(f"fm_294k shape invalid: {fm_294k.shape}")
    if fm_294k.shape[0] != N_CELLS:
        print(f"[WARN] fm_294k size {fm_294k.shape[0]} != expected N_CELLS {N_CELLS} -- continuing")

    print(f"fm_294k resolved shape: {fm_294k.shape}")

    # Build model, load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FMCoefficientPredictor(input_size=INPUT_SIZE, hidden_sizes=HIDDEN_SIZES, output_size=OUTPUT_SIZE,
                                   dropout_rate=DROPOUT_RATE, use_batch_norm=USE_BATCH_NORM).to(device)
    try:
        model.load_state_dict(state)
        print("✓ Successfully loaded model_state_dict (strict=True)")
    except Exception as e:
        print("[WARN] strict load failed:", e)
        try:
            model.load_state_dict(state, strict=False)
            print("✓ Loaded model_state_dict with strict=False (some params may be uninitialized).")
        except Exception as e2:
            print("[ERROR] Failed to load model_state_dict even with strict=False:", e2)
            sys.exit(1)

    model.eval()

    # Load input temps and optional OpenMC keff
    inputs = np.load(INPUT_TEMPS_PATH, allow_pickle=True)
    keff_path = find_first_existing(KEFF_CANDIDATES)
    openmc_keffs = None
    if keff_path:
        try:
            openmc_keffs = load_keff_array(keff_path)
            print(f"Loaded OpenMC keff from {keff_path} ({len(openmc_keffs)} entries)")
        except Exception as e:
            print(f"[WARN] Failed to load OpenMC keff: {e}")
            openmc_keffs = None
    else:
        print("No OpenMC keff file found; pcm comparisons will be NaN.")

    n_samples = len(inputs)
    print(f"Found {n_samples} input samples; building features...")

    features = []
    valid_idx = []
    skipped = []
    for i in range(n_samples):
        try:
            feat = make_features_from_input_tuple(inputs[i], N_CELLS=N_CELLS)
            if feat.shape[0] != INPUT_SIZE:
                raise ValueError(f"feature length {feat.shape[0]} != expected input_size {INPUT_SIZE}")
            features.append(feat.astype(np.float32))
            valid_idx.append(i)
        except Exception as e:
            skipped.append((i, str(e)))
    print(f"Built {len(features)} features; skipped {len(skipped)} examples")

    if len(features) == 0:
        print("[ERROR] No valid features to evaluate")
        sys.exit(1)

    X_all = np.stack(features, axis=0)
    print(f"Feature matrix shape: {X_all.shape}")

    # Decide normalization (prefer saved mean/std)
    if train_mean is not None and train_std is not None and train_mean.size == X_all.shape[1] and train_std.size == X_all.shape[1]:
        print("Using saved train_mean/train_std from checkpoint.")
        t_mean = train_mean
        t_std = train_std
    else:
        print("train_mean/std missing or mismatched, computing from features (fallback).")
        t_mean = X_all.mean(axis=0)
        t_std = X_all.std(axis=0) + 1e-8

    # Run batched inference
    preds = []
    with torch.no_grad():
        for start in range(0, X_all.shape[0], BATCH):
            end = min(start + BATCH, X_all.shape[0])
            Xb = X_all[start:end]
            Xn = (Xb - t_mean) / t_std
            Xt = torch.from_numpy(Xn).float().to(device)
            out = model(Xt).cpu().numpy()
            preds.append(out)
    preds = np.vstack(preds)
    print(f"Predictions shape: {preds.shape}")

    # Evaluate: reconstruct FMs and compute keff + pcm
    predicted_kfm = []
    pcm_list = []
    valid_global_indices = []
    for local_idx, global_idx in enumerate(valid_idx):
        ln_flat = preds[local_idx].reshape(N_CELLS, N_CELLS)
        ln_flat = np.clip(ln_flat, LN_CLIP_MIN, LN_CLIP_MAX)
        coeff = np.exp(ln_flat)
        fm_pred = fm_294k * coeff
        kfm, _ = power_iteration_user(fm_pred)
        predicted_kfm.append(kfm)

        # compute pcm if openmc available
        if openmc_keffs is not None and global_idx < openmc_keffs.size:
            k_true = openmc_keffs[global_idx]
            if np.isfinite(kfm) and np.isfinite(k_true) and k_true != 0:
                pcm = (kfm / k_true - 1.0) * 1e5
            else:
                pcm = np.nan
        else:
            pcm = np.nan
        pcm_list.append(pcm)
        valid_global_indices.append(global_idx)

        if (local_idx + 1) % 200 == 0 or (local_idx + 1) == len(valid_idx):
            print(f"  Processed {local_idx+1}/{len(valid_idx)} (global idx {global_idx})")

    predicted_kfm = np.array(predicted_kfm, dtype=float)
    pcm_arr = np.array(pcm_list, dtype=float)
    valid_global_indices = np.array(valid_global_indices, dtype=int)

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    np.save(RESULTS_DIR / "predicted_kfm_3hidden.npy", predicted_kfm)
    np.save(RESULTS_DIR / "pcm_errors_3hidden.npy", pcm_arr)
    np.save(RESULTS_DIR / "valid_indices_3hidden.npy", valid_global_indices)
    print(f"\nSaved predicted_kfm_3hidden.npy, pcm_errors_3hidden.npy, valid_indices_3hidden.npy to {RESULTS_DIR}")

    # Summaries
    finite_kfm = predicted_kfm[np.isfinite(predicted_kfm)]
    print("\nPredicted kfm summary:")
    if finite_kfm.size:
        print(f"  n = {finite_kfm.size}, mean = {finite_kfm.mean():.6f}, std = {finite_kfm.std():.6f}")
    else:
        print("  No finite kfm values.")

    if np.any(np.isfinite(pcm_arr)):
        pcm_finite = pcm_arr[np.isfinite(pcm_arr)]
        print("\nPCM error summary:")
        print(f"  mean abs pcm = {np.mean(np.abs(pcm_finite)):.2f} pcm")
        print(f"  median abs pcm = {np.median(np.abs(pcm_finite)):.2f} pcm")
    else:
        print("\nNo PCM comparisons available.")

    print("\nDone.")

if __name__ == "__main__":
    main()
