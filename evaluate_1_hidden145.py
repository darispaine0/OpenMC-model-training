#!/usr/bin/env python3
"""
evaluate_fm_1hidden_145.py

Evaluate the single-hidden-layer (145) model:
 - loads checkpoint fm_coefficient_model_1hidden_145.pth
 - reconstructs predicted FMs using fm_294k
 - computes dominant eigenvalue via power iteration
 - computes PCM error vs OpenMC keff (if available)
Saves predicted_kfm.npy, pcm_errors.npy, valid_indices.npy in fm_coefficient_results/
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import math
import sys

# -------------- Config / Paths --------------
RESULTS_DIR = Path("fm_coefficient_results")
CHECKPOINT_PATH = RESULTS_DIR / "fm_coefficient_model_1hidden_145.pth"

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
BATCH = 32

# -------------- Utilities --------------
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

# -------------- Power iteration (user's implementation) --------------
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

# -------------- Model class (matching trainer) --------------
class FMCoefficientPredictor(nn.Module):
    def __init__(self, input_size=578, hidden_size=145, output_size=289*289):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.network(x)

# -------------- Main --------------
def main():
    if not CHECKPOINT_PATH.exists():
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)
    if not INPUT_TEMPS_PATH.exists() or not FM_NORMALIZED_PATH.exists():
        print("[ERROR] Required consolidated data files missing.")
        sys.exit(1)

    print("Loading inputs and fission matrices...")
    inputs = np.load(INPUT_TEMPS_PATH, allow_pickle=True)
    fms = np.load(FM_NORMALIZED_PATH, allow_pickle=True)
    n_samples = min(len(inputs), len(fms))
    print(f"  Found {len(inputs)} input samples and {len(fms)} fm samples (using {n_samples})")

    # find and robustly load fm_294k
    fm294k_path = find_first_existing(FM_294K_CANDIDATES)
    if fm294k_path is None:
        print("[ERROR] Could not find fm_294k reference file.")
        sys.exit(1)

    print(f"Loading fm_294k from {fm294k_path} ...")
    raw = np.load(fm294k_path, allow_pickle=True)

    # handle various saved formats robustly
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        if raw.ndim == 0:
            raw = raw.item()
        elif raw.ndim == 1 and raw.size == 1:
            raw = raw[0]

    fm_294k = np.asarray(raw, dtype=float)
    if fm_294k.ndim > 2:
        fm_294k = np.squeeze(fm_294k)
    if fm_294k.ndim == 1:
        N_CELLS = int(math.isqrt(fm_294k.size))
        if N_CELLS * N_CELLS != fm_294k.size:
            raise ValueError("fm_294k flattened size is not a perfect square.")
        fm_294k = fm_294k.reshape(N_CELLS, N_CELLS)
    elif fm_294k.ndim == 2:
        if fm_294k.shape[0] != fm_294k.shape[1]:
            raise ValueError(f"fm_294k is not square: shape={fm_294k.shape}")
        N_CELLS = fm_294k.shape[0]
    else:
        raise ValueError(f"Unexpected fm_294k ndim: {fm_294k.ndim}")

    print(f"  fm_294k shape resolved to {fm_294k.shape} (N_CELLS={N_CELLS})")

    # load checkpoint
    print("Loading model checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if "train_mean" not in ckpt or "train_std" not in ckpt:
        raise RuntimeError("Checkpoint must contain 'train_mean' and 'train_std'")

    train_mean = np.asarray(ckpt["train_mean"], dtype=float)
    train_std = np.asarray(ckpt["train_std"], dtype=float)

    # derive sizes
    input_size = int(train_mean.size)
    output_size = N_CELLS * N_CELLS

    cfg = ckpt.get("config", None)
    if cfg and isinstance(cfg, dict):
        hidden_size = int(cfg.get("HIDDEN_SIZE", 145))
    else:
        hidden_size = 145

    print(f"Model sizes -> input: {input_size}, hidden: {hidden_size}, output: {output_size}")

    # build, load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FMCoefficientPredictor(input_size, hidden_size, output_size).to(device)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as e:
        print("[ERROR] Failed to load state_dict into model. Details:")
        raise

    model.eval()

    # reconstruct features
    print("Reconstructing features...")
    features = []
    valid_idx = []
    for i in range(n_samples):
        try:
            feat = make_features_from_input_tuple(inputs[i], N_CELLS=N_CELLS)
            features.append(feat)
            valid_idx.append(i)
        except Exception as e:
            print(f"  [WARN] skipping index {i} during feature reconstruction: {e}")

    if not features:
        raise RuntimeError("No valid features reconstructed.")

    X_all = np.stack(features, axis=0).astype(np.float32)
    print(f"  Features stacked shape: {X_all.shape}")

    # prediction loop
    preds_ln = []
    print("Running model predictions...")
    with torch.no_grad():
        for start in range(0, X_all.shape[0], BATCH):
            end = min(start + BATCH, X_all.shape[0])
            Xb = X_all[start:end]
            Xn = (Xb - train_mean) / train_std
            Xt = torch.from_numpy(Xn).float().to(device)
            out = model(Xt).cpu().numpy()
            preds_ln.append(out)
    preds_ln = np.vstack(preds_ln)
    print("Predictions done. preds_ln.shape =", preds_ln.shape)

    # attempt to load openmc keff values for comparison
    keff_path = find_first_existing(KEFF_CANDIDATES)
    openmc_keffs = None
    if keff_path is not None:
        try:
            openmc_keffs = load_keff_array(keff_path)
            print(f"Loaded {len(openmc_keffs)} OpenMC keff values from {keff_path}")
        except Exception as e:
            print(f"[WARN] Failed to load keff array: {e}")
            openmc_keffs = None
    else:
        print("No OpenMC keff file found in candidates; skipping pcm comparisons.")

    # reconstruct predicted FMs, run power iteration
    print("Reconstructing predicted FMs and running power iteration...")
    kfm_list = []
    pcm_list = []
    openmc_list = []

    for local_idx, global_idx in enumerate(valid_idx):
        lnvec = preds_ln[local_idx].reshape(N_CELLS, N_CELLS)
        lnvec = np.clip(lnvec, LN_CLIP_MIN, LN_CLIP_MAX)
        coeff = np.exp(lnvec)
        fm_pred = fm_294k * coeff

        kfm, eigvec = power_iteration_user(fm_pred)
        kfm_list.append(kfm)

        if openmc_keffs is not None and global_idx < openmc_keffs.size:
            k_openmc = openmc_keffs[global_idx]
            openmc_list.append(k_openmc)
            if np.isfinite(kfm) and np.isfinite(k_openmc) and k_openmc != 0:
                pcm = (kfm / k_openmc - 1.0) * 1e5
            else:
                pcm = np.nan
            pcm_list.append(pcm)
        else:
            openmc_list.append(np.nan)
            pcm_list.append(np.nan)

        if (local_idx + 1) % 200 == 0 or (local_idx + 1) == len(valid_idx):
            print(f"  Processed {local_idx+1}/{len(valid_idx)} (global idx {global_idx})")

    kfm_arr = np.array(kfm_list, dtype=float)
    pcm_arr = np.array(pcm_list, dtype=float)
    openmc_arr = np.array(openmc_list, dtype=float)

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    np.save(RESULTS_DIR / "predicted_kfm_1hidden_145.npy", kfm_arr)
    np.save(RESULTS_DIR / "predicted_kfm_indices_1hidden_145.npy", np.array(valid_idx, dtype=int))
    np.save(RESULTS_DIR / "pcm_errors_1hidden_145.npy", pcm_arr)
    if openmc_arr.size > 0 and not np.all(np.isnan(openmc_arr)):
        np.save(RESULTS_DIR / "openmc_keff_for_eval_1hidden_145.npy", openmc_arr)

    # print summaries
    finite_kfm = kfm_arr[np.isfinite(kfm_arr)]
    print("\nPredicted kfm summary:")
    if finite_kfm.size:
        print(f"  n = {finite_kfm.size}, mean = {finite_kfm.mean():.6f}, std = {finite_kfm.std():.6f}")
    else:
        print("  No finite kfm values.")

    if np.any(np.isfinite(pcm_arr)):
        pcm_finite = pcm_arr[np.isfinite(pcm_arr)]
        print("\nPCM error summary (kfm vs OpenMC keff):")
        print(f"  mean abs pcm = {np.mean(np.abs(pcm_finite)):.2f} pcm")
        print(f"  median abs pcm = {np.median(np.abs(pcm_finite)):.2f} pcm")
    else:
        print("\nNo PCM comparisons available.")

    print(f"\nSaved: {RESULTS_DIR / 'predicted_kfm_1hidden_145.npy'} , {RESULTS_DIR / 'pcm_errors_1hidden_145.npy'}")

if __name__ == "__main__":
    main()
