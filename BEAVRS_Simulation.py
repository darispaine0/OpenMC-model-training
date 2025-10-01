#!/usr/bin/env python3
"""
Fission Matrix Testing with OpenMC BEAVRS Assembly
Compares fission matrix eigenvalue and eigenvector with OpenMC reference
"""

"""
For local environment:
1: conda activate openmc-env
2: cd ~/Monte_Carlo/Test
3: export OPENMC_CROSS_SECTIONS=/mnt/c/Users/Daris/Downloads/endfb80/endfb-viii.0-hdf5/cross_sections.xml
# optional but nice:
export OPENMC_DATA=/mnt/c/Users/Daris/Downloads/endfb80/endfb-viii.0-hdf5
"""

import numpy as np
import openmc
import matplotlib.pyplot as plt
import re

def ask_int(prompt, default):
    try:
        return int(input(prompt).strip())
    except Exception:
        print(f"Invalid input; defaulting to {default}")
        return default

def material_has_uranium(mat):
    """Check if material contains uranium using multiple methods."""
    # Method 1: Check material name
    try:
        if hasattr(mat, 'name') and mat.name:
            name_lower = mat.name.lower()
            if 'fuel' in name_lower or 'uo2' in name_lower:
                return True
    except Exception:
        pass
    
    # Method 2: Check using get_nuclides() 
    try:
        nuclides = mat.get_nuclides()
        for nuclide in nuclides:
            if 'U' in str(nuclide):
                return True
    except Exception:
        pass
    
    # Method 3: Check nuclides list directly
    try:
        if hasattr(mat, 'nuclides') and mat.nuclides:
            for nuclide_tuple in mat.nuclides:
                nuclide_str = str(nuclide_tuple[0])
                if 'U234' in nuclide_str or 'U235' in nuclide_str or 'U238' in nuclide_str:
                    return True
                if re.search(r'U\d+', nuclide_str):  # Any uranium isotope
                    return True
    except Exception:
        pass
    
    # Method 4: Try get_nuclide_atom_fraction
    try:
        for isotope in ['U234', 'U235', 'U238']:
            if mat.get_nuclide_atom_fraction(isotope) > 0:
                return True
    except Exception:
        pass
    
    return False


def main():
    modelnum = ask_int("Which model to run? (1 = single-temp/unit, 2 = fuel/non-fuel two-temperatures): ", 1)
    n_runs   = ask_int("Number of runs (e.g. 5): ", 1)
    particles = ask_int("Particles per batch (e.g. 1000): ", 1000)
    temp_min = ask_int("Min random temp (K): ", 300)
    temp_max = ask_int("Max random temp (K): ", 900)
    PlotInput = input("Show plots (Y/N): ")

    input_file = "input_temps.npy"
    output_source_file = "output_source.npy"   # per-run mesh fission source (pj)
    output_fm_raw_file = "output_fm_raw.npy"   # raw fm arrays from tallies (flattened)
    output_fm_file = "output_fm_normalized.npy" # normalized FM used for validation
    output_keff_file = "output_keff.npy"
    output_keff_uncertainty = "output_keff_uncertainty.npy"
    output_source_uncertainty = "output_source_uncertainty.npy"

    input_data = []
    source_data = []
    fm_raw_data = []
    fm_normalized_data = []
    keff_data = []
    source_uncertainty_data = []
    keff_uncertainty_data = []
    
    n_rows = 17
    n_cols = 17
    
    for run_idx in range(n_runs):
        print(f"\n=== Run {run_idx+1}/{n_runs} ===")

        # Create PWR assembly geometry FIRST
        print("Creating PWR assembly geometry...")
        assembly = openmc.examples.pwr_assembly()
        geometry = assembly.geometry
        
        # Get the lattice from the assembly
        # Find the lattice in the geometry
        lattice = None
        for cell in geometry.get_all_cells().values():
            if isinstance(cell.fill, openmc.RectLattice):
                lattice = cell.fill
                break
        
        if lattice is None:
            raise RuntimeError("Could not find lattice in assembly geometry")

        # Set temperatures AFTER creating the geometry and finding lattice
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
                                print("Uranium Detected")
                                cell.temperature = assigned_fuel
                            else:
                                assigned_other = float(temps[i, j, 1])
                                cell.temperature = assigned_other
                    temps[i, j, 0] = assigned_fuel
                    temps[i, j, 1] = assigned_other
            input_data.append(temps.copy())
        
        # Set up mesh for tallies
        print("Setting up mesh and tallies...")
        mesh = openmc.RegularMesh()
        mesh.dimension = (n_rows, n_cols)
        mesh.lower_left = (-10.71, -10.71)
        mesh.upper_right = (+10.71, +10.71)
        
        # Set up filters
        mesh_filter = openmc.MeshFilter(mesh)
        born_filter = openmc.MeshBornFilter(mesh)
        
        # Set up tallies
        tally_source = openmc.Tally(name='Fission source')
        tally_source.filters = [mesh_filter]
        tally_source.scores = ['nu-fission']
        
        tally_fm = openmc.Tally(name='Fission matrix')
        tally_fm.filters = [mesh_filter, born_filter]
        tally_fm.scores = ['nu-fission']
        
        # Add tallies to model
        tallies = assembly.tallies
        tallies.append(tally_source)
        tallies.append(tally_fm)
        assembly.tallies = tallies
        
        # Configure simulation settings
        print("Configuring simulation settings...")
        settings = assembly.settings
        settings.particles = particles  # Use the user input
        settings.batches = 100
        settings.inactive = 30
        settings.temperature = {"method": "interpolation", "default": 294.0, "tolerance": 400.0}
        
        # Run simulation
        print("Running OpenMC simulation...")
        assembly.run()
        
        # Analyze results
        print("Analyzing results...")
        with openmc.StatePoint(filepath='statepoint.100.h5') as output:
            keff = output.keff
            keff_mean, keff_std = openmc.lib.keff()
            keff_data.append(keff)
            keff_uncertainty_data.append(keff_std/keff_mean)
            source_tally = output.get_tally(name='Fission source')
            fm_tally = output.get_tally(name='Fission matrix')
        
        # Extract values
        source = source_tally.mean.squeeze()
        source_std = source_tally.std_dev.squeeze()
        fm_raw = fm_tally.mean.squeeze()
        fm_std = fm_tally.std_dev.squeeze()
        fm_raw_data.append(fm_raw)
        source_data.append(source)

        #Relative Uncertainty
        Source_rel_un = np.divide(source_std, source, out=np.zeros_like(source_std), where=source!=0).reshape(n_rows, n_cols)

        source_uncertainty_data.append(Source_rel_un)

        # Build fission matrix
        print("Building fission matrix...")
        N_pins = 17
        N_cells = int(N_pins**2)
        fm = np.zeros((N_cells, N_cells))
        
        for i in range(N_cells):
            for j in range(N_cells):
                idx = j * N_cells + i
                if fm_raw[idx] > 0:
                    fm[i][j] = fm_raw[idx] / np.sum(fm_raw[j*N_cells:(j+1)*N_cells]) * np.sum(fm_raw)
        fm_normalized_data.append(fm)
        
        # Power iteration to find dominant eigenvalue and eigenvector
        print("Performing power iteration...")
        size, _ = fm.shape
        eig_value = 1.0
        eig_vector = np.ones((size, 1))
        tolerance, max_iter = 1e-6, 500
        
        for i_iter in range(max_iter):
            eig_vector_new = np.dot(fm, eig_vector)
            eig_value_new = np.max(np.abs(eig_vector_new))
            eig_vector = eig_vector_new / eig_value_new
            if i_iter > 0 and abs(eig_value_new - eig_value) < tolerance:
                break
            eig_value = eig_value_new
        
        eig_vector /= eig_vector.sum()
        
        # Compare results
        print("Comparing results...")
        kfm = eig_value
        source_fm = eig_vector.reshape((N_pins, N_pins))
        komc = keff.nominal_value.real
        source_omc = source.reshape((N_pins, N_pins)) / source.sum()
        
        # keff comparison
        keff_rel_diff = (kfm / komc - 1)
        print(f'keff relative difference: {keff_rel_diff*1e5:.0f} pcm')
        
        # Source comparison
        source_rel_diff = np.zeros_like(source_fm)
        source_rel_diff = np.divide(source_fm, source_omc, out=np.ones_like(source_fm), where=source_omc!=0) - 1.0
        source_rel_diff = np.nan_to_num(source_rel_diff, nan=0.0, posinf=0.0, neginf=0.0)
        max_loc = np.argmax(np.abs(source_rel_diff))
        rdiff_max = source_rel_diff.flatten()[max_loc]
        rdiff_ave = np.abs(source_rel_diff[source_rel_diff>0]).mean()
        
        print(f'Fission source largest relative difference: {rdiff_max*100:.2f} %')
        print(f'Fission source average signless relative difference: {rdiff_ave*100:.2f} %')
        
        # Plot source comparison
        if PlotInput == "Y":
            print("Creating comparison plot...")
            plt.rcParams['font.size'] = 15

            fig, axes = plt.subplots(1, 2, figsize=(20, 9))  # Two side-by-side plots

            # ---- Plot 1: Fission Source Relative Difference ----
            im1 = axes[0].imshow(source_rel_diff * 100, cmap='jet', interpolation='nearest', 
                                origin='lower', aspect='equal')
            
            axes[0].set_xticks(range(N_pins))
            axes[0].set_yticks(range(N_pins))
            axes[0].set_xticklabels(range(1, N_pins + 1))
            axes[0].set_yticklabels(range(1, N_pins + 1))
            axes[0].grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.6)
            axes[0].tick_params(which='minor', bottom=False, left=False)

            cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            cbar1.set_label('Fission source relative difference (%)')
            
            axes[0].set_title(f'Fission Source - Relative Difference (Run {run_idx+1})')

            # ---- Plot 2: Fission Source Relative Uncertainty ----
            # Assuming Source_rel_un is already computed as relative uncertainty (std / mean)
            im2 = axes[1].imshow(Source_rel_un * 100, cmap='jet', interpolation='nearest',
                                origin='lower', aspect='equal')
            
            axes[1].set_xticks(range(N_pins))
            axes[1].set_yticks(range(N_pins))
            axes[1].set_xticklabels(range(1, N_pins + 1))
            axes[1].set_yticklabels(range(1, N_pins + 1))
            axes[1].grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.6)
            axes[1].tick_params(which='minor', bottom=False, left=False)

            cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            cbar2.set_label('Fission source relative uncertainty (%)')
            
            axes[1].set_title(f'Fission Source - Relative Uncertainty (Run {run_idx+1})')

            plt.tight_layout()
            plt.show()

            
        print(f'Analysis {run_idx+1} complete!')
    
    # Save all data
    print("\nSaving data to files...")
    np.save(input_file, np.array(input_data, dtype=object))
    np.save(output_source_file, np.array(source_data, dtype=object))
    np.save(output_fm_raw_file, np.array(fm_raw_data, dtype=object))
    np.save(output_fm_file, np.array(fm_normalized_data, dtype=object))
    np.save(output_keff_file, np.array(keff_data, dtype=object))
    np.save(output_keff_uncertainty, np.array(keff_uncertainty_data, dtype=object))
    np.save(output_source_uncertainty, np.array(source_uncertainty_data, dtype=object))
    print("All runs complete and data saved!")

if __name__ == "__main__":
    main()