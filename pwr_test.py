import os                             # interact with the operating system (env vars, paths)
import tempfile                       # create temporary directories
import shutil                         # high-level file operations (not used actively here, kept from original)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # avoid HDF5 file-locking issues inside some containers/filesystems

import openmc                         # main OpenMC Python API
import openmc.examples                # convenience examples (pwr_assembly)
import numpy as np                    # numerical library for arrays and random numbers
import glob                           # filename pattern matching (to find statepoint files)
import re                             # regular expressions for robust nuclide matching

# steps to run simulation 
# #1: docker run -it -v ${PWD}:/openmc openmc/openmc 
# #2: cd /openmc 
# #3: cd "/openmc/OneDrive/Documents/Monte_Carlo/Test" 
# #4: python3 pwr_test.py
# (these are usage instructions for your environment; not executed by Python)

# -----------------------------
# USER SETTINGS
# -----------------------------
n_runs = 5                            # number of independent random-temperature runs to perform
temp_min, temp_max = 300, 1000        # lower / upper bounds for randomized fuel temperatures (K)
input_file = "input_temps.npy"       # file to save the input 17x17 temperature matrices (per run)
output_file = "output_flux.npy"      # file to save the output flux matrices (per run)
# -----------------------------

# --- Load PWR assembly materials first ---
model = openmc.examples.pwr_assembly()  # load example PWR assembly geometry/materials/settings as an OpenMC Model

# --- Robustly replace U234 with U235 in all materials (handles variants like 'U234.70c') ---
u234_pattern = re.compile(r'(^|[^A-Za-z0-9])U?234([^A-Za-z0-9]|$)', re.IGNORECASE)
# explanation: compile a regex that matches 'U234', 'U234.70c', '234U' variants (robust & case-insensitive)

for mat in model.materials:            # iterate every Material object in the model
    # collect keys to remove (exact keys as they appear in the material)
    keys = []
    try:
        # mat.nuclides is usually a mapping; get its keys (nuclide name strings)
        nucl_keys = list(mat.nuclides.keys())
    except Exception:
        # fallback: some OpenMC versions store nuclides differently; try private attribute
        try:
            nucl_keys = list(mat._nuclides.keys())
        except Exception:
            nucl_keys = []

    for nk in nucl_keys:               # inspect each nuclide key in this material
        if u234_pattern.search(nk):     # if the regex matches this key (looks like U234)
            keys.append(nk)             # mark it for removal

    for nk in keys:                     # for each found U234-like key
        try:
            n_atoms = mat.get_nuclide_atom_fraction(nk)  # read atom fraction for this nuclide (if possible)
        except Exception:
            # fallback: if we can't read the atom fraction, set to None (we'll skip adding)
            n_atoms = None

        # remove the U234 variant from the material (safe even if it fails)
        try:
            mat.remove_nuclide(nk)
        except Exception:
            pass

        # add the same atom fraction as U235 if we successfully read it
        if n_atoms is not None:
            try:
                mat.add_nuclide('U235', n_atoms)
            except Exception:
                # if adding U235 fails, we silently continue (rare)
                pass

# --- Settings ---
settings = openmc.Settings()            # create a Settings object for the simulation
settings.batches = 10                   # number of batches (generations) to run
settings.inactive = 0                   # number of inactive batches (0 here)
settings.particles = 100000               # particles per batch (low for quick tests)
# interpolation method tells OpenMC how to handle non-exact temperature cross sections
settings.temperature = {"method": "interpolation"}
model.settings = settings               # attach settings to the model

# --- Lattice info ---
lattices = list(model.geometry.get_all_lattices().values())  # get list of lattice objects in geometry
if len(lattices) == 0:
    raise RuntimeError("No lattices found in the model!")     # fail early if no lattice exists
lattice = lattices[0]                  # assume the first lattice is the 17x17 assembly lattice

n_rows, n_cols = 17, 17                 # lattice dimensions (17x17 assembly)
print(f"Lattice size: {n_rows}x{n_cols}")

input_data = []                         # container to store per-run input matrices
output_data = []                        # container to store per-run output flux matrices

# helper to check whether a material contains any uranium (robust against suffixes)
def material_has_uranium(mat):
    try:
        nucl_keys = list(mat.nuclides.keys())
    except Exception:
        try:
            nucl_keys = list(mat._nuclides.keys())
        except Exception:
            nucl_keys = []
    for nk in nucl_keys:
        # look for strings like 'U235', 'U234', 'U238' or variants like 'U235.70c'
        if re.search(r'U\s*\d{3}', nk, re.IGNORECASE) or re.search(r'9223[4-8]', nk):
            # ensure it mentions a common uranium mass number (234, 235, 238)
            if re.search(r'234|235|238', nk):
                return True
    return False

for run in range(n_runs):               # outer loop over independent randomized runs
    print(f"\n=== Simulation {run+1}/{n_runs} ===")

    # Randomize temperatures for each lattice square (one scalar per 17x17 position)
    temps = np.random.uniform(temp_min, temp_max, size=(n_rows, n_cols))
    # store a copy of this matrix for later saving (we'll update temps in-place with forced-294s)
    input_data.append(temps.copy())

    # Assign temperature to fuel cells only (fuel cell gets temps[i,j]; others fixed at 294 K)
    for i in range(n_rows):
        for j in range(n_cols):
            universe = lattice.universes[i][j]   # get the Universe placed at lattice position (i,j)
            assigned_temp = 294.0                # default stored temperature if no fuel present

            # iterate all cells in the universe (fuel + cladding + moderator if present)
            for cell in universe.cells.values():
                if isinstance(cell.fill, openmc.Material):  # ensure the cell is filled by a Material
                    mat = cell.fill

                    # If the material contains uranium (makes it a fuel material), randomize it
                    if material_has_uranium(mat):
                        # check whether any U234-like nuclide remains in this material
                        try:
                            nuclist = list(mat.nuclides.keys())
                        except Exception:
                            nuclist = []
                        has_u234 = any(u234_pattern.search(nk) for nk in nuclist)

                        if has_u234:
                            # safety: if U-234 remains, force the cell to 294 K to avoid runtime error
                            cell.temperature = 294.0
                            assigned_temp = 294.0
                        else:
                            # normal case: assign randomized temp to the fuel cell
                            assigned_temp = temps[i, j]
                            cell.temperature = assigned_temp
                    else:
                        # cladding / moderator / guide-tube materials: keep at 294 K
                        cell.temperature = 294.0
                else:
                    # cell.fill is None or another Universe; nothing to set here
                    pass

            # store a single scalar per lattice square: fuel temp if present, else 294 K
            temps[i, j] = assigned_temp

    # --- Define tally ---
    tallies = openmc.Tallies()           # fresh Tallies container for this run
    root = model.geometry.root_universe   # top-level universe (used for bounding box)
    lower_left, upper_right = root.bounding_box  # get geometry bbox corners
    z_lower, z_upper = 0.0, 1.0           # vertical bounds for the mesh (single z-slab)

    mesh = openmc.RegularMesh()           # make a regular Cartesian mesh for tallying
    mesh.dimension = (n_rows, n_cols, 1)  # mesh cells correspond to assembly pins in XY and single Z
    mesh.lower_left = (lower_left[0], lower_left[1], z_lower)  # mesh lower-left coordinate
    mesh.upper_right = (upper_right[0], upper_right[1], z_upper)  # mesh upper-right coordinate

    mesh_filter = openmc.MeshFilter(mesh) # create a filter that selects each mesh cell
    tally = openmc.Tally(name="flux_tally")  # create a tally object named "flux_tally"
    tally.filters = [mesh_filter]          # apply the mesh filter so tally is per-mesh-cell
    tally.scores = ["flux"]                # score neutron flux
    tallies.append(tally)                  # add tally to the tallies collection
    model.tallies = tallies                # attach tallies to the model

    # --- Run in temporary directory ---
    with tempfile.TemporaryDirectory() as tmpdir:  # create and auto-delete a temp directory
        model.export_to_xml(tmpdir)        # write settings, geometry, materials, tallies to XML in tmpdir
        openmc.run(output=True, cwd=tmpdir)  # execute OpenMC in that tmpdir

        # --- Load latest statepoint ---
        statepoints = sorted(glob.glob(os.path.join(tmpdir, "statepoint.*.h5")),
                             key=os.path.getmtime)   # find produced statepoint files, sort by mod time
        if not statepoints:
            raise FileNotFoundError("No statepoint files found!")
        latest_sp = statepoints[-1]         # pick the most recent statepoint file
        print(f"Loading {latest_sp}")

        sp = openmc.StatePoint(latest_sp)   # open the statepoint HDF5 file
        flux_tally = sp.get_tally(name="flux_tally")  # get the tally by name
        flux_matrix = flux_tally.get_reshaped_data(value="mean").squeeze()  # retrieve mean flux as 2D array
        output_data.append(flux_matrix)     # append flux matrix to outputs list

# Save dataset
np.save(input_file, np.array(input_data))   # write all input matrices (n_runs x 17 x 17) to .npy file
np.save(output_file, np.array(output_data)) # write all output flux matrices similarly
print("\n✅ Done!")
