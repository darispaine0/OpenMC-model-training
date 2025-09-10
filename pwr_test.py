
"""

- Model 1: one temperature per lattice unit (17x17) -> input saved shape (n_runs,17,17)
- Model 2: two temperatures per lattice unit (fuel, other) -> input saved shape (n_runs,17,17,2)
- Output: p_j (vector of mesh fission sources) and f_ij (mesh-by-mesh fission matrix)
"""

# steps to run simulation 
# #1: docker run -it -v ${PWD}:/openmc openmc/openmc 
# #2: cd /openmc 
# #3: cd "/openmc/OneDrive/Documents/Monte_Carlo/Test" 
# #4: python3 pwr_test.py
# (these are usage instructions for my environment; not executed by Python)
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # avoid HDF5 locking issues in some Docker/FS setups

import tempfile
import glob
import shutil
import numpy as np
import openmc
import openmc.examples
import re

# ---------------- USER INPUT ----------------
try:
    modelnum = int(input("Which model to run? (1 = single-temp/unit, 2 = fuel/non-fuel two-temperatures): ").strip())
except Exception:
    print("Invalid model number; defaulting to 1")
    modelnum = 1

try:
    n_runs = int(input("Number of runs (e.g. 5): ").strip())
except Exception:
    print("Invalid run number; defaulting to 5")
    n_runs = 5

try:
    particles = int(input("Particles per batch (e.g. 1000): ").strip())
except Exception:
    print("Invalid particle number; defaulting to 1000")
    particles = 1000

# ---------------- SETTINGS ----------------
n_rows_default = 17
n_cols_default = 17

temp_min, temp_max = 300.0, 500.0   # sampling range (K) for randomization
input_file = "input_temps.npy"
output_pj_file = "output_pj.npy"
output_fij_file = "output_fij.npy"

# ---------------- helper functions ----------------
def material_has_uranium(mat):
    """Robust check whether material contains U234/U235/U238 (works across openmc versions)."""
    try:
        # Preferred API: get_nuclide_atom_fraction returns 0/None if absent
        if mat.get_nuclide_atom_fraction("U234") or mat.get_nuclide_atom_fraction("U235") or mat.get_nuclide_atom_fraction("U238"):
            return True
    except Exception:
        pass
    # Fallback: inspect nuclides keys
    try:
        keys = list(mat.nuclides.keys())
    except Exception:
        keys = []
    for k in keys:
        ks = str(k)
        if "U234" in ks or "U235" in ks or "U238" in ks:
            return True
        # catch ZAID-like strings
        if re.search(r"9223(4|5|8)", ks):
            return True
    return False

def finite_bbox(root):
    """Return finite bounding box ((xmin,ymin,zmin),(xmax,ymax,zmax)) with safe defaults."""
    ll, ur = root.bounding_box
    ll = list(ll); ur = list(ur)
    # sensible assembly fallbacks
    defaults = (-10.71, -10.71, -1.0, 10.71, 10.71, 1.0)
    for idx in range(3):
        if not np.isfinite(ll[idx]):
            ll[idx] = defaults[idx]
        if not np.isfinite(ur[idx]):
            ur[idx] = defaults[idx+3]
    return tuple(ll), tuple(ur)

# ---------------- Build model ----------------
model = openmc.examples.pwr_assembly()  # example PWR assembly

# Settings: ensure a statepoint is written and use nearest temperature fallback
settings = openmc.Settings()
settings.batches = 10
settings.inactive = 0
settings.particles = particles
settings.statepoint = {"batches": [settings.batches]}
# Use nearest/default to avoid fail if some nuclides only have 294 K data
settings.temperature = {"method": "nearest", "default": 294.0, "tolerance": 500.0}
model.settings = settings

# Get lattice (assume first lattice is the assembly)
lattices = list(model.geometry.get_all_lattices().values())
if len(lattices) == 0:
    raise RuntimeError("No lattices found in the model geometry.")
lattice = lattices[0]

# Determine lattice dims robustly
try:
    n_rows = len(lattice.universes)
    n_cols = len(lattice.universes[0])
    if n_rows is None or n_cols is None:
        raise Exception()
except Exception:
    n_rows, n_cols = n_rows_default, n_cols_default

print(f"Using lattice size: {n_rows} x {n_cols}")

# Build mesh covering assembly
ll, ur = finite_bbox(model.geometry.root_universe)
x_min, y_min, z_min = ll
x_max, y_max, z_max = ur
# z-limits will be a small slab
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

# Make a persistent runs folder so outputs remain after completion
out_root = os.path.abspath("runs")
os.makedirs(out_root, exist_ok=True)

# ---------------- Main loop ----------------
for run_idx in range(n_runs):
    print(f"\n=== Run {run_idx+1}/{n_runs} ===")
    # Model 1: single temperature per lattice unit
    if modelnum == 1:
        temps = np.random.uniform(temp_min, temp_max, size=(n_rows, n_cols)).astype(float)
        # assign same temperature to all cells in that lattice universe
        for i in range(n_rows):
            for j in range(n_cols):
                uni = lattice.universes[i][j]
                if uni is None:
                    continue
                assigned_temp = float(temps[i, j])
                # set temperature for all cells that are materials in the universe
                for cell in uni.cells.values():
                    # if cell.fill is a Material, set it; if it's a Universe, this is nested and we skip here
                    if isinstance(cell.fill, openmc.Material):
                        cell.temperature = assigned_temp
                    else:
                        # if fill is Universe or None, we don't set here
                        pass
                # record actual assigned (no change expected)
                temps[i, j] = assigned_temp
        input_data.append(temps.copy())

    # Model 2: two temperatures per lattice unit (fuel, other)
    elif modelnum == 2:
        # temps[...,0] = fuel temp, temps[...,1] = other cell temp
        temps = np.random.uniform(temp_min, temp_max, size=(n_rows, n_cols, 2)).astype(float)
        # default second-channel (non-fuel) to 294 K if you prefer; here we keep randomized second channel
        for i in range(n_rows):
            for j in range(n_cols):
                uni = lattice.universes[i][j]
                if uni is None:
                    continue
                assigned_fuel = 294.0
                assigned_other = 294.0
                # iterate universe cells: set fuel cells to fuel temp; others to other temp
                for cell in uni.cells.values():
                    if isinstance(cell.fill, openmc.Material):
                        mat = cell.fill
                        if material_has_uranium(mat):
                            assigned_fuel = float(temps[i, j, 0])
                            cell.temperature = assigned_fuel
                        else:
                            assigned_other = float(temps[i, j, 1])
                            cell.temperature = assigned_other
                    else:
                        # nested Universe or void: skip
                        pass
                temps[i, j, 0] = assigned_fuel
                temps[i, j, 1] = assigned_other
        input_data.append(temps.copy())

    else:
        raise ValueError("modelnum must be 1 or 2")

    # Create tallies: p_j (mesh nu-fission) and f_ij (mesh nu-fission with MeshBornFilter)
    tallies = openmc.Tallies()

    p_j = openmc.Tally(name="nufission")
    p_j.filters = [mesh_filter]
    p_j.scores = ["nu-fission"]

    f_ij = openmc.Tally(name="f_ij")
    f_ij.filters = [mesh_filter, openmc.MeshBornFilter(mesh)]
    f_ij.scores = ["nu-fission"]

    tallies += [p_j, f_ij]
    model.tallies = tallies

    # Run in per-run directory
    run_dir = os.path.join(out_root, f"run_{run_idx+1}")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    model.export_to_xml(run_dir)
    print(f"Running OpenMC in {run_dir} (this may take a bit)...")
    openmc.run(output=True, cwd=run_dir)

    # locate statepoint
    sp_files = sorted(glob.glob(os.path.join(run_dir, "statepoint.*.h5")), key=os.path.getmtime)
    if not sp_files:
        raise FileNotFoundError(f"No statepoint.*.h5 found in {run_dir}")
    sp_file = sp_files[-1]
    print("Loading", sp_file)
    sp = openmc.StatePoint(sp_file)

    # Extract tallies
    try:
        pj_arr = sp.get_tally(name="nufission").get_reshaped_data(value="mean").squeeze()
    except Exception as e:
        raise RuntimeError("Failed to extract p_j tally: " + str(e))
    try:
        fij_arr = sp.get_tally(name="f_ij").get_reshaped_data(value="mean").squeeze()
    except Exception as e:
        raise RuntimeError("Failed to extract f_ij tally: " + str(e))

    # Post-process shapes: convert to 1D p_j and 2D f_ij (mesh_cells x mesh_cells)
    n_mesh = n_rows * n_cols

    # Flatten p_j to vector length n_mesh
    pj_flat = np.array(pj_arr).ravel()
    if pj_flat.size != n_mesh:
        # sometimes reshaped array has extra dims; try to reduce
        pj_flat = pj_flat[:n_mesh]
    pj_data.append(pj_flat.copy())

    # Convert fij to (n_mesh, n_mesh)
    fij_flat = np.array(fij_arr).ravel()
    if fij_flat.size < n_mesh * n_mesh:
        # If insufficient size, pad with zeros (shouldn't normally happen)
        fij_flat = np.pad(fij_flat, (0, n_mesh * n_mesh - fij_flat.size))
    fij_mat = fij_flat.reshape((n_mesh, n_mesh))
    fij_data.append(fij_mat.copy())

    print(f"  p_j shape (flattened): {pj_flat.shape}")
    print(f"  f_ij shape (matrix): {fij_mat.shape}")

# ---------------- Save results ----------------
input_array = np.array(input_data, dtype=object) if modelnum == 2 else np.array(input_data)
# for consistency, save using allow_pickle only if dtype=object
np.save(input_file, input_array)
np.save(output_pj_file, np.array(pj_data))
np.save(output_fij_file, np.array(fij_data))

print("\nSaved files:")
print(" -", input_file, "shape:", np.load(input_file, allow_pickle=True).shape)
print(" -", output_pj_file, "shape:", np.load(output_pj_file, allow_pickle=True).shape)
print(" -", output_fij_file, "shape:", np.load(output_fij_file, allow_pickle=True).shape)

print("\nAll statepoint files written under:", out_root)
for p in sorted(glob.glob(os.path.join(out_root, "run_*", "statepoint.*.h5"))):
    print(" ", p)

print("\nDone.")
