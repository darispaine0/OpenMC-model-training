
import os
import tempfile
import shutil
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import openmc
import openmc.examples
import numpy as np
import glob
# steps to run simulation 
# #1: docker run -it -v ${PWD}:/openmc openmc/openmc 
# #2: cd /openmc 
# #3: cd "/openmc/OneDrive/Documents/Monte_Carlo/Test" 
# #4: python3 pwr_test.py

# -----------------------------
# USER SETTINGS
# -----------------------------
n_runs = 5
temp_min, temp_max = 294, 294
input_file = "input_temps.npy"
output_file = "output_flux.npy"
# -----------------------------

# --- Load PWR assembly materials first ---
model = openmc.examples.pwr_assembly()

# --- Replace U234 in fuel with U235 ---
for mat in model.materials:
    if "U234" in mat.nuclides:
        n_atoms = mat.get_nuclide_atom_fraction("U234")
        print(f"Replacing U234 with U235 in {mat.name} ({n_atoms} atom fraction)")
        mat.remove_nuclide("U234")
        mat.add_nuclide("U235", n_atoms)

# --- Settings ---
settings = openmc.Settings()
settings.batches = 10
settings.inactive = 0
settings.particles = 1000
settings.temperature = {"method": "interpolation"}
model.settings = settings

# --- Lattice info ---
lattices = list(model.geometry.get_all_lattices().values())
if len(lattices) == 0:
    raise RuntimeError("No lattices found in the model!")
lattice = lattices[0]

n_rows, n_cols = 17, 17
print(f"Lattice size: {n_rows}x{n_cols}")

input_data = []
output_data = []

for run in range(n_runs):
    print(f"\n=== Simulation {run+1}/{n_runs} ===")

    # Randomize temperatures for each cell in lattice
    temps = np.random.uniform(temp_min, temp_max, size=(n_rows, n_cols))
    input_data.append(temps)

    # Assign temperature to each fuel pin
    for i in range(n_rows):
        for j in range(n_cols):
            universe = lattice.universes[i][j]
            cell = list(universe.cells.values())[0]  # assume 1 cell per universe
            cell.temperature = temps[i, j]

    # --- Define tally ---
    tallies = openmc.Tallies()
    root = model.geometry.root_universe
    lower_left, upper_right = root.bounding_box
    z_lower, z_upper = 0.0, 1.0

    mesh = openmc.RegularMesh()
    mesh.dimension = (n_rows, n_cols, 1)
    mesh.lower_left = (lower_left[0], lower_left[1], z_lower)
    mesh.upper_right = (upper_right[0], upper_right[1], z_upper)

    mesh_filter = openmc.MeshFilter(mesh)
    tally = openmc.Tally(name="flux_tally")
    tally.filters = [mesh_filter]
    tally.scores = ["flux"]
    tallies.append(tally)
    model.tallies = tallies

    # --- Run in temporary directory ---
    with tempfile.TemporaryDirectory() as tmpdir:
        model.export_to_xml(tmpdir)
        openmc.run(output=True, cwd=tmpdir)

        # --- Load latest statepoint ---
        statepoints = sorted(glob.glob(os.path.join(tmpdir, "statepoint.*.h5")),
                             key=os.path.getmtime)
        if not statepoints:
            raise FileNotFoundError("No statepoint files found!")
        latest_sp = statepoints[-1]
        print(f"Loading {latest_sp}")

        sp = openmc.StatePoint(latest_sp)
        flux_tally = sp.get_tally(name="flux_tally")
        flux_matrix = flux_tally.get_reshaped_data(value="mean").squeeze()
        output_data.append(flux_matrix)

# Save dataset
np.save(input_file, np.array(input_data))
np.save(output_file, np.array(output_data))
print("\n✅ Done!")
