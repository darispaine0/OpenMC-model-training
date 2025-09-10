#!/usr/bin/env python3
"""
view_full_fij.py

Loads:
 - input_temps.npy
 - output_pj.npy
 - output_fij.npy

For each sample shows:
 - input temperature map(s) (17x17 or 17x17x2)
 - p_j reshaped to 17x17
 - reduced f_ij (sum over sources -> destination totals) as 17x17
 - full f_ij matrix as 289x289
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------- auto-detect data dir (adjust if needed) ----------------
if sys.platform.startswith("win"):
    data_dir = os.path.join(os.environ.get("USERPROFILE", ""), "OneDrive", "Documents", "Monte_Carlo", "Test")
else:
    data_dir = os.path.expanduser("~/Monte_Carlo/Test")

input_path = os.path.join(data_dir, "input_temps.npy")
pj_path = os.path.join(data_dir, "output_pj.npy")
fij_path = os.path.join(data_dir, "output_fij.npy")

# ---------------- load files ----------------
for p in (input_path, pj_path, fij_path):
    if not os.path.exists(p):
        print("Missing file:", p)
        sys.exit(1)

input_temps = np.load(input_path, allow_pickle=True)
pj_all = np.load(pj_path, allow_pickle=True)
fij_all = np.load(fij_path, allow_pickle=True)

print("Loaded:")
print(" - input_temps:", input_temps.shape, input_temps.dtype)
print(" - output_pj:", pj_all.shape, pj_all.dtype)
print(" - output_fij:", fij_all.shape, fij_all.dtype)

# Try to stack object arrays into numeric arrays when possible
def try_stack(arr):
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            return np.stack(arr, axis=0)
        except Exception:
            return arr
    return arr

input_temps = try_stack(input_temps)
pj_all = try_stack(pj_all)
fij_all = try_stack(fij_all)

# Determine number of comparable samples
def first_dim(a):
    try:
        return int(a.shape[0])
    except Exception:
        return 0

n_input = first_dim(input_temps)
n_pj = first_dim(pj_all)
n_fij = first_dim(fij_all)
N_candidates = [x for x in (n_input, n_pj, n_fij) if x>0]
if not N_candidates:
    print("Unable to determine sample count. Shapes:", input_temps.shape, pj_all.shape, fij_all.shape)
    sys.exit(1)
N = min(N_candidates)
print(f"Number of matched samples: {N}")

# helpers
def as_17x17(vec):
    vec = np.asarray(vec).ravel()
    if vec.size == 289:
        return vec.reshape((17,17))
    # try square reshape
    side = int(np.sqrt(vec.size))
    if side*side == vec.size:
        return vec.reshape((side, side))
    # fallback: pad/trim to 289
    if vec.size < 289:
        vec = np.pad(vec, (0, 289-vec.size))
    else:
        vec = vec[:289]
    return vec.reshape((17,17))

def full_fij_matrix(fij):
    """Return a 289x289 matrix from various possible shapes of fij."""
    arr = np.asarray(fij)
    if arr.ndim == 2 and arr.shape[0]*arr.shape[1] == 289*289:
        # if already 289x289 (or another square that equals 289^2)
        if arr.shape == (289,289):
            return arr
        else:
            # if it's (289,289) already but shaped weird, reshape anyway
            return arr.reshape((289,289))
    flat = arr.ravel()
    if flat.size >= 289*289:
        return flat[:289*289].reshape((289,289))
    else:
        # pad with zeros
        flat = np.pad(flat, (0, 289*289 - flat.size))
        return flat.reshape((289,289))

# main visualization loop
for i in range(N):
    temp = input_temps[i]
    temp = np.array(temp) if not isinstance(temp, np.ndarray) else temp

    pj = np.array(pj_all[i]).ravel()
    fij = fij_all[i]  # could be array or object

    pj_map = as_17x17(pj)

    fij_mat_full = full_fij_matrix(fij)
    # reduced destination totals (sum over sources)
    fij_dest = fij_mat_full.sum(axis=0)
    fij_map = as_17x17(fij_dest)

    # Plot layout:
    # Top row: Input(s) | p_j (17x17) | reduced f_ij (17x17)
    # Bottom row: full f_ij (289x289) spanning width
    plt.close('all')
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 1.5), width_ratios=(1,1,1), hspace=0.3)

    # Input panels
    if temp.ndim == 3 and temp.shape[-1] == 2:
        ax_in0 = fig.add_subplot(gs[0,0])
        im0 = ax_in0.imshow(temp[...,0], origin='lower', interpolation='none')
        ax_in0.set_title(f"Input temp (fuel) sample {i+1}")
        fig.colorbar(im0, ax=ax_in0, fraction=0.05)

        ax_in1 = fig.add_subplot(gs[1,0])
        im1 = ax_in1.imshow(temp[...,1], origin='lower', interpolation='none')
        ax_in1.set_title(f"Input temp (other) sample {i+1}")
        fig.colorbar(im1, ax=ax_in1, fraction=0.05)
        # leave center top empty for layout consistency
        ax_pj = fig.add_subplot(gs[0,1])
        ax_fij_reduced = fig.add_subplot(gs[0,2])
    else:
        ax_in = fig.add_subplot(gs[0,0])
        if temp.ndim == 1:
            im_in = ax_in.imshow(as_17x17(temp), origin='lower', interpolation='none')
            ax_in.set_title(f"Input (reshaped) sample {i+1}")
            fig.colorbar(im_in, ax=ax_in, fraction=0.05)
        elif temp.ndim == 2:
            im_in = ax_in.imshow(temp, origin='lower', interpolation='none')
            ax_in.set_title(f"Input temp sample {i+1}")
            fig.colorbar(im_in, ax=ax_in, fraction=0.05)
        else:
            ax_in.text(0.5, 0.5, f"Cannot display input shape {temp.shape}", ha='center', va='center')
            ax_in.axis('off')
        ax_pj = fig.add_subplot(gs[0,1])
        ax_fij_reduced = fig.add_subplot(gs[0,2])

    # p_j (17x17)
    im_pj = ax_pj.imshow(pj_map, origin='lower', interpolation='none')
    ax_pj.set_title("p_j (mesh source) 17x17")
    fig.colorbar(im_pj, ax=ax_pj, fraction=0.05)

    # reduced f_ij (destination totals)
    im_fred = ax_fij_reduced.imshow(fij_map, origin='lower', interpolation='none')
    ax_fij_reduced.set_title("f_ij reduced (sum over sources) 17x17")
    fig.colorbar(im_fred, ax=ax_fij_reduced, fraction=0.05)

    # full f_ij 289x289 on second row spanning all columns
    ax_full = fig.add_subplot(gs[1,:])
    im_full = ax_full.imshow(fij_mat_full, origin='lower', interpolation='nearest', aspect='auto')
    ax_full.set_title("Full f_ij matrix (289 x 289)")
    fig.colorbar(im_full, ax=ax_full, fraction=0.04, pad=0.02)
    ax_full.set_xlabel("destination mesh cell index (0..288)")
    ax_full.set_ylabel("source mesh cell index (0..288)")

    plt.suptitle(f"Sample {i+1}/{N}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(f"Showing sample {i+1}/{N} — press any key in the figure (or close it) to continue...")
    plt.waitforbuttonpress(timeout=-1)
    plt.close(fig)

print("Done viewing all samples.")
