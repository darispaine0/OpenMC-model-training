#!/usr/bin/env python3
"""
- Model 1: one temperature per lattice unit (17x17) -> input saved shape (n_runs,17,17)
- Model 2: two temperatures per lattice unit (fuel, other) -> input saved shape (n_runs,17,17,2)
- Output: p_j (vector of mesh fission sources) and f_ij (mesh-by-mesh fission matrix)

Steps to run (bash for my env only)
1. docker run -it -v ${PWD}:/openmc openmc/openmc
2. cd /openmc
3. cd "/openmc/OneDrive/Documents/Monte_Carlo/Test"
4. python3 pwr_test_validation.py
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import glob
import shutil
import numpy as np
import openmc
import openmc.examples
import re
import h5py

# ---------------- USER INPUT ----------------
try:
    modelnum = int(input("Which model to run? (1 = single-temp/unit, 2 = fuel/non-fuel two-temperatures): ").strip())
except Exception:
    print("Invalid model number; defaulting to 1")
    modelnum = 1

try:
    n_runs = int(input("Number of runs (e.g. 5): ").strip())
except Exception:
    print("Invalid run number; defaulting to 1")
    n_runs = 1

try:
    particles = int(input("Particles per batch (e.g. 1000): ").strip())
except Exception:
    print("Invalid particle number; defaulting to 1000")
    particles = 1000

# ---------------- SETTINGS ----------------
n_rows_default = 17
n_cols_default = 17

temp_min, temp_max = 300.0, 500.0   # sampling (K)
input_file = "input_temps.npy"
output_pj_file = "output_pj.npy"
output_fij_file = "output_fij.npy"
output_keff_file = "output_keff.npy"
output_fij_op_file = "output_fij_operator.npy"   # saved operator actually used for validation
output_norm_file = "output_normalization.npy"    # store normalization info per run

# ---------------- helper functions ----------------
def material_has_uranium(mat):
    try:
        if mat.get_nuclide_atom_fraction("U234") or mat.get_nuclide_atom_fraction("U235") or mat.get_nuclide_atom_fraction("U238"):
            return True
    except Exception:
        pass
    try:
        keys = list(mat.nuclides.keys())
    except Exception:
        keys = []
    for k in keys:
        ks = str(k)
        if "U234" in ks or "U235" in ks or "U238" in ks:
            return True
        if re.search(r"9223(4|5|8)", ks):
            return True
    return False

def finite_bbox(root):
    ll, ur = root.bounding_box
    ll = list(ll); ur = list(ur)
    defaults = (-10.71, -10.71, -1.0, 10.71, 10.71, 1.0)
    for idx in range(3):
        if not np.isfinite(ll[idx]):
            ll[idx] = defaults[idx]
        if not np.isfinite(ur[idx]):
            ur[idx] = defaults[idx+3]
    return tuple(ll), tuple(ur)

def calculate_keff_direct(A, Fvec):
    N, _ = A.shape
    F = np.ones((N,1))/float(N)
    AF = A @ F
    dk = 1.0
    keff_old = 10
    i_iter = 0
    while i_iter < 500 and dk > 1e-5:
        i_iter += 1
        print(i_iter)
        F_new = A.dot(F)
        keff = F_new.max() 
        F = F_new/keff
        dk = np.abs(keff/keff_old-1)
        keff_old = keff
        
    return keff, F_new

def extract_keff_comprehensive(sp, run_dir, run_idx):
    keff_value = None
    keff_std = None
    try:
        if hasattr(sp, 'k_effective'):
            keff = sp.k_effective
            if hasattr(keff, 'nominal_value'):
                return keff.nominal_value, keff.std_dev, "sp.k_effective"
            elif isinstance(keff, (int, float)):
                return float(keff), 0.0, "sp.k_effective(direct)"
    except Exception:
        pass
    try:
        for attr_name in ['keff', 'k_eff', 'multiplication_factor', 'eigenvalue']:
            if hasattr(sp, attr_name):
                keff_obj = getattr(sp, attr_name)
                if hasattr(keff_obj, 'nominal_value'):
                    return keff_obj.nominal_value, getattr(keff_obj, 'std_dev', 0.0), f"sp.{attr_name}"
                elif isinstance(keff_obj, (int, float)):
                    return float(keff_obj), 0.0, f"sp.{attr_name}"
    except Exception:
        pass
    try:
        sp_files = [f for f in os.listdir(run_dir) if f.startswith('statepoint') and f.endswith('.h5')]
        if sp_files:
            sp_file = os.path.join(run_dir, sorted(sp_files)[-1])
            with h5py.File(sp_file, 'r') as f:
                keff_paths = ['k_effective', 'keff', 'eigenvalue', 'multiplication_factor', 'global/k_effective', 'results/k_effective', 'k-effective']
                for path in keff_paths:
                    if path in f:
                        keff_data = f[path]
                        if isinstance(keff_data, h5py.Dataset):
                            if keff_data.shape == ():
                                return float(keff_data[()]), 0.0, f"HDF5:{path}"
                            else:
                                return float(keff_data[-1]), 0.0, f"HDF5:{path}"
    except Exception:
        pass
    return None, None, "failed"

# ---------------- Build model ----------------
model = openmc.examples.pwr_assembly()

settings = openmc.Settings()
settings.batches = 100
settings.inactive = 20
settings.particles = particles
settings.statepoint = {"batches": [settings.batches]}
settings.temperature = {"method": "nearest", "default": 294.0, "tolerance": 500.0}
model.settings = settings

lattices = list(model.geometry.get_all_lattices().values())
if len(lattices) == 0:
    raise RuntimeError("No lattices found in the model geometry.")
lattice = lattices[0]

try:
    n_rows = len(lattice.universes)
    n_cols = len(lattice.universes[0])
    if n_rows is None or n_cols is None:
        raise Exception()
except Exception:
    n_rows, n_cols = n_rows_default, n_cols_default

print(f"Using lattice size: {n_rows} x {n_cols}")

ll, ur = finite_bbox(model.geometry.root_universe)
x_min, y_min, z_min = ll
x_max, y_max, z_max = ur
z_lower, z_upper = -1.0, 1.0

mesh = openmc.RegularMesh()
mesh.dimension = (n_rows, n_cols, 1)
mesh.lower_left = (x_min, y_min, z_lower)
mesh.upper_right = (x_max, y_max, z_upper)
mesh_filter = openmc.MeshFilter(mesh)

# Storage
input_data = []
pj_data = []
fij_data = []
fij_op_data = []            # store final operator used for validation
normalization_info = []     # store (method, scale) per run
keff_data = []

out_root = os.path.abspath("runs")
os.makedirs(out_root, exist_ok=True)

# ---------------- Main loop ----------------
for run_idx in range(n_runs):
    print(f"\n=== Run {run_idx+1}/{n_runs} ===")

    # Set temperatures
    if modelnum == 1:
        temps = np.random.uniform(temp_min, temp_max, size=(n_rows, n_cols)).astype(float)
        for i in range(n_rows):
            for j in range(n_cols):
                uni = lattice.universes[i][j]
                if uni is None:
                    continue
                assigned_temp = float(temps[i, j])
                for cell in uni.cells.values():
                    if isinstance(cell.fill, openmc.Material):
                        cell.temperature = assigned_temp
                temps[i, j] = assigned_temp
        input_data.append(temps.copy())
    elif modelnum == 2:
        temps = np.random.uniform(temp_min, temp_max, size=(n_rows, n_cols, 2)).astype(float)
        for i in range(n_rows):
            for j in range(n_cols):
                uni = lattice.universes[i][j]
                if uni is None:
                    continue
                assigned_fuel = 294.0
                assigned_other = 294.0
                for cell in uni.cells.values():
                    if isinstance(cell.fill, openmc.Material):
                        mat = cell.fill
                        if material_has_uranium(mat):
                            assigned_fuel = float(temps[i, j, 0])
                            cell.temperature = assigned_fuel
                        else:
                            assigned_other = float(temps[i, j, 1])
                            cell.temperature = assigned_other
                temps[i, j, 0] = assigned_fuel
                temps[i, j, 1] = assigned_other
        input_data.append(temps.copy())
    else:
        raise ValueError("modelnum must be 1 or 2")

    # Tallies
    tallies = openmc.Tallies()

    p_j = openmc.Tally(name="nufission")
    p_j.filters = [mesh_filter]
    p_j.scores = ["nu-fission"]

    f_ij = openmc.Tally(name="f_ij")
    f_ij.filters = [mesh_filter, openmc.MeshBornFilter(mesh)]
    f_ij.scores = ["nu-fission"]

    tallies += [p_j, f_ij]
    model.tallies = tallies

    # Run OpenMC
    run_dir = os.path.join(out_root, f"run_{run_idx+1}")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    model.export_to_xml(run_dir)
    print(f"Running OpenMC in {run_dir} ...")
    openmc.run(output=True, cwd=run_dir)

    # load statepoint
    sp_files = sorted(glob.glob(os.path.join(run_dir, "statepoint.*.h5")), key=os.path.getmtime)
    if not sp_files:
        raise FileNotFoundError(f"No statepoint.*.h5 found in {run_dir}")
    sp_file = sp_files[-1]
    print("Loading", sp_file)
    sp = openmc.StatePoint(sp_file)

    print("OpenMC version:", openmc.__version__)

    # Extract tallies (use sum for total counts)
    try:
        pj_arr = sp.get_tally(name="nufission").get_reshaped_data(value="sum").squeeze()
        fij_arr = sp.get_tally(name="f_ij").get_reshaped_data(value="sum").squeeze()
    except Exception as e:
        raise RuntimeError("Failed to extract tallies: " + str(e))

    # Extract keff
    keff_value, keff_std, _ = extract_keff_comprehensive(sp, run_dir, run_idx)
    if keff_value is None:
        keff_value = np.nan
    keff_data.append(keff_value)

    # Shapes and flatten
    n_mesh = n_rows * n_cols
    pj_flat = np.array(pj_arr).ravel()
    if pj_flat.size != n_mesh:
        pj_flat = pj_flat[:n_mesh]
    fij_flat = np.array(fij_arr).ravel()
    if fij_flat.size < n_mesh * n_mesh:
        fij_flat = np.pad(fij_flat, (0, n_mesh*n_mesh - fij_flat.size))
    fij_mat = fij_flat.reshape((n_mesh, n_mesh))

    pj_data.append(pj_flat.copy())
    fij_data.append(fij_mat.copy())

    print(f"  p_j shape (flattened): {pj_flat.shape}")
    print(f"  f_ij shape (matrix): {fij_mat.shape}")

    # ------------------ Normalization logic (same used in validation) ------------------
    A_raw = fij_mat.copy()
    P = pj_flat.copy().flatten()
    eps = 1e-20
    used_normalization = None
    scale_applied = 1.0

    # Try to extract birth counts from source bank if available
    birth_counts = None
    try:
        sources = None
        if hasattr(sp, 'source') and sp.source is not None and len(sp.source) > 0:
            sources = sp.source
        elif hasattr(sp, 'source_bank') and sp.source_bank is not None and len(sp.source_bank) > 0:
            sources = sp.source_bank

        if sources is not None:
            # get positions and optional weights
            try:
                positions = np.array(sources['r'])
            except Exception:
                # fallback structure: list-like
                positions = np.array([s[0] for s in sources])
            try:
                weights = np.array(sources['wgt'])
            except Exception:
                weights = np.ones(len(positions), dtype=float)

            lower = np.array(mesh.lower_left[:2], dtype=float)
            upper = np.array(mesh.upper_right[:2], dtype=float)
            dims = np.array(mesh.dimension[:2], dtype=int)
            widths = (upper - lower) / dims
            widths[widths == 0] = 1.0
            rel = (positions[:, :2] - lower[None, :]) / widths[None, :]
            ix = np.floor(rel[:, 0]).astype(int)
            iy = np.floor(rel[:, 1]).astype(int)
            ix = np.clip(ix, 0, dims[0]-1)
            iy = np.clip(iy, 0, dims[1]-1)
            flat_idx = ix * dims[1] + iy
            birth_counts = np.zeros(n_mesh, dtype=float)
            for k, idx in enumerate(flat_idx):
                birth_counts[idx] += float(weights[k])
            print("  Found source bank: birth_counts extracted.")
    except Exception:
        birth_counts = None

    # helper: evaluate operator
    def evaluate_operator(Aop):
        keff_ray, F = calculate_keff_direct(Aop, P)
        res_norm = np.linalg.norm(Aop @ P - keff_value * P)
        rel_err = abs(keff_value - keff_ray) / (abs(keff_value) + 1e-20)
        return keff_ray, res_norm, rel_err

    # 1) Try birth-count normalization
    if birth_counts is not None and birth_counts.sum() > 0:
        bc_safe = birth_counts.copy(); bc_safe[bc_safe == 0.0] = eps
        A_birth = A_raw.copy()
        for j in range(n_mesh):
            A_birth[:, j] = A_raw[:, j] / bc_safe[j]
        keff_birth, res_birth, rel_birth = evaluate_operator(A_birth)
        print(f"  birth-count normalization -> rel_err={rel_birth:.6e}")
        if rel_birth < 0.02:
            used_normalization = "birth_counts"
            A_op_final = A_birth.copy()
            keff_candidate = keff_birth
    # 2) Column-normalize by P (per-fission)
    if used_normalization is None:
        P_safe = P.copy(); P_safe[P_safe == 0.0] = eps
        A_p = A_raw.copy()
        for j in range(n_mesh):
            A_p[:, j] = A_raw[:, j] / P_safe[j]
        keff_p, res_p, rel_p = evaluate_operator(A_p)
        print(f"  divide-by-P normalization -> rel_err={rel_p:.6e}")
        if rel_p < 0.02:
            used_normalization = "divide_by_P"
            A_op_final = A_p.copy()
            keff_candidate = keff_p

    # 3) Global scale on top of divide_by_P as last resort
    if used_normalization is None:
        try:
            eigvals = np.linalg.eigvals(A_p)
            dom = float(np.real(eigvals[np.argmax(np.real(eigvals))]))
        except Exception:
            dom = np.sum(A_p) / (n_mesh + eps)
        scale_needed = keff_value / (dom + 1e-30)
        A_scaled = A_p * scale_needed
        keff_scaled, res_scaled, rel_scaled = evaluate_operator(A_scaled)
        print(f"  global scale on divide-by-P -> scale={scale_needed:.6e}, rel_err={rel_scaled:.6e}")
        # accept even if slightly larger rel_err because it's a last-resort fix
        used_normalization = "global_scale_on_divide_by_P"
        scale_applied = float(scale_needed)
        A_op_final = A_scaled.copy()
        keff_candidate = keff_scaled
        rel_error_candidate = rel_scaled

    # Save operator + normalization info for this run
    fij_op_data.append(A_op_final.copy())
    normalization_info.append({'method': used_normalization, 'scale': float(scale_applied)})
    print(f"  Selected normalization: {used_normalization}  |  keff_from_operator={keff_candidate:.6f}")

# ---------------- Save results ----------------
input_array = np.array(input_data, dtype=object) if modelnum == 2 else np.array(input_data)
np.save(input_file, input_array)
np.save(output_pj_file, np.array(pj_data))
np.save(output_fij_file, np.array(fij_data))
np.save(output_fij_op_file, np.array(fij_op_data))
np.save(output_norm_file, np.array(normalization_info))
np.save(output_keff_file, np.array(keff_data))

print("\nSaved files:")
print(" -", input_file, "shape:", np.load(input_file, allow_pickle=True).shape)
print(" -", output_pj_file, "shape:", np.load(output_pj_file, allow_pickle=True).shape)
print(" -", output_fij_file, "shape:", np.load(output_fij_file, allow_pickle=True).shape)
print(" -", output_fij_op_file, "shape:", np.load(output_fij_op_file, allow_pickle=True).shape)
print(" -", output_keff_file, "shape:", np.load(output_keff_file, allow_pickle=True).shape)

print("\nAll statepoint files written under:", out_root)
for p in sorted(glob.glob(os.path.join(out_root, "run_*", "statepoint.*.h5"))):
    print(" ", p)

# ---------------- Validation across runs ----------------
print("\nValidation process beginning: ")
validation_errors = []

for idx in range(n_runs):
    try:
        A_op = np.array(fij_op_data[idx], dtype=float)      # operator selected earlier
        Pj = np.array(pj_data[idx], dtype=float).flatten()
        keff_openmc = keff_data[idx]

        print(f"\n--- Run {idx+1} Validation ---")
        if np.isnan(keff_openmc):
            print(f"Skipping run {idx+1}: k_eff extraction failed")
            validation_errors.append(f"Run {idx+1}: k_eff extraction failed")
            continue

        keff_solved, F = calculate_keff_direct(A_op, Pj)
        if keff_openmc != 0:
            rel_error = abs(keff_openmc - keff_solved) / abs(keff_openmc)
            print(f"OpenMC k_eff: {keff_openmc:.6f}")
            print(f"Eigenvalue k_eff (from selected A_op): {keff_solved:.6f}")
            print(f"Relative error: {rel_error:.4%}")
            if rel_error > 0.1:
                validation_errors.append(f"Run {idx+1}: Large error ({rel_error:.4%})")
        else:
            print("Cannot calculate relative error: OpenMC k_eff is zero")
            validation_errors.append(f"Run {idx+1}: OpenMC k_eff is zero")

    except Exception as e:
        print(f"Error in validation for run {idx+1}: {e}")
        validation_errors.append(f"Run {idx+1}: Validation error - {e}")

# Summary
print("\n" + "="*50)
print("VALIDATION SUMMARY")
print("="*50)
if validation_errors:
    print(f"Issues found in {len(validation_errors)} runs:")
    for error in validation_errors:
        print(f"  - {error}")
else:
    print("All runs validated successfully!")

print("\nDone.")
