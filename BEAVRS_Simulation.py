#!/usr/bin/env python3
"""
Fission Matrix Testing with OpenMC BEAVRS Assembly
Compares fission matrix eigenvalue and eigenvector with OpenMC reference
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

def main():
    modelnum = ask_int("Which model to run? (1 = single-temp/unit, 2 = fuel/non-fuel two-temperatures): ", 1)
    n_runs   = ask_int("Number of runs (e.g. 5): ", 1)
    particles = ask_int("Particles per batch (e.g. 1000): ", 1000)
    temp_min = ask_int("Min random temp (K): ", 300)
    temp_max = ask_int("Max random temp (K): ", 900)
    PlotInput = input("Show FM plots (Y/N): ")

    input_file = "input_temps.npy"
    output_source_file = "output_source.npy"   # per-run mesh fission source (pj)
    output_fm_raw_file = "output_fm_raw.npy"   # raw fm arrays from tallies (flattened)
    output_fm_file = "output_fm_normalized.npy" # normalized FM used for validation
    output_keff_file = "output_keff.npy"

    input_data = []
    source_data = []
    fm_raw_data = []
    fm_normalized_data = []
    keff_data = []
    
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
            keff_data.append(keff)
            source_tally = output.get_tally(name='Fission source')
            fm_tally = output.get_tally(name='Fission matrix')
        
        # Extract values
        source = source_tally.mean.squeeze()
        source_std = source_tally.std_dev.squeeze()
        fm_raw = fm_tally.mean.squeeze()
        fm_std = fm_tally.std_dev.squeeze()
        fm_raw_data.append(fm_raw)
        source_data.append(source)

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
            fig, ax = plt.subplots(figsize=(12, 9))
            im = ax.imshow(source_rel_diff*100, cmap='jet', interpolation='nearest', 
                        origin='lower', aspect='equal')
            
            ax.set_xticks(range(N_pins))
            ax.set_yticks(range(N_pins))
            ax.set_xticklabels(range(1, N_pins+1))
            ax.set_yticklabels(range(1, N_pins+1))
            ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.6)
            ax.tick_params(which='minor', bottom=False, left=False)
            
            # Colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Fission source relative difference (%)')
            
            # Set title
            ax.set_title(f'Fission source in BEAVRS assembly - FM vs. OpenMC (Run {run_idx+1})')
            
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
    
    print("All runs complete and data saved!")

if __name__ == "__main__":
    main()