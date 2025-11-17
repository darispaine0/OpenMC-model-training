#!/usr/bin/env python3
"""
Constant Temperature Fission Matrix Generation with OpenMC BEAVRS Assembly
Creates fission matrices at constant temperatures: 300K, 400K, ..., 900K
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

def ask_int(prompt, default):
    try:
        return int(input(prompt).strip())
    except Exception:
        print(f"Invalid input; defaulting to {default}")
        return default

def main():
    particles = ask_int("Particles per batch (e.g. 1000): ", 1000)
    PlotInput = input("Show plots (Y/N): ")

    # Temperature range: 300K to 900K in 100K increments
    temperatures = range(300, 1000, 100)
    
    n_rows = 17
    n_cols = 17
    N_pins = 17
    N_cells = int(N_pins**2)
    
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"Processing Temperature: {temp}K")
        print(f"{'='*60}")

        # Create PWR assembly geometry
        print("Creating PWR assembly geometry...")
        assembly = openmc.examples.pwr_assembly()
        geometry = assembly.geometry
        
        # Find the lattice in the geometry
        lattice = None
        for cell in geometry.get_all_cells().values():
            if isinstance(cell.fill, openmc.RectLattice):
                lattice = cell.fill
                break
        
        if lattice is None:
            raise RuntimeError("Could not find lattice in assembly geometry")

        # Set constant temperature across all cells
        print(f"Setting all cells to {temp}K...")
        for i in range(n_rows):
            for j in range(n_cols):
                uni = lattice.universes[i][j]
                if uni is None:
                    continue
                for cell in uni.cells.values():
                    if isinstance(cell.fill, openmc.Material):
                        cell.temperature = float(temp)
        
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
        settings.particles = particles
        settings.batches = 100
        settings.inactive = 30
        settings.temperature = {"method": "interpolation", "default": float(temp), "tolerance": 400.0}
     
        # Run simulation
        print("Running OpenMC simulation...")
        assembly.run()
        
        # Analyze results
        print("Analyzing results...")
        with openmc.StatePoint(filepath='statepoint.100.h5') as output:
            keff = output.keff
            keff_mean = keff.nominal_value.real
            keff_std = keff.std_dev.real
            source_tally = output.get_tally(name='Fission source')
            fm_tally = output.get_tally(name='Fission matrix')
        
        # Extract values
        source = source_tally.mean.squeeze()
        source_std = source_tally.std_dev.squeeze()
        fm_raw = fm_tally.mean.squeeze()
        
        # Build normalized fission matrix
        print("Building normalized fission matrix...")
        fm = np.zeros((N_cells, N_cells))
        
        for i in range(N_cells):
            for j in range(N_cells):
                idx = j * N_cells + i
                if fm_raw[idx] > 0:
                    fm[i][j] = fm_raw[idx] / np.sum(fm_raw[j*N_cells:(j+1)*N_cells]) * np.sum(fm_raw)
        
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
        print(f'keff (OpenMC): {komc:.6f} ± {keff_std:.6f}')
        print(f'keff (Fission Matrix): {kfm:.6f}')
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
        
        # Save fission matrix
        output_filename = f"fission_matrix_{temp}k.npy"
        np.save(output_filename, fm)
        print(f"Saved fission matrix to: {output_filename}")
        
        # Save additional data
        np.save(f"keff_{temp}k.npy", np.array([keff_mean, keff_std]))
        np.save(f"source_{temp}k.npy", source)
        
        # Plot source comparison
        if PlotInput == "Y":
            print("Creating comparison plot...")
            plt.rcParams['font.size'] = 15

            # Calculate relative uncertainty
            Source_rel_un = np.divide(source_std, source, out=np.zeros_like(source_std), where=source!=0).reshape(n_rows, n_cols)

            fig, axes = plt.subplots(1, 2, figsize=(20, 9))

            # Plot 1: Fission Source Relative Difference
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
            
            axes[0].set_title(f'Fission Source - Relative Difference ({temp}K)')

            # Plot 2: Fission Source Relative Uncertainty
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
            
            axes[1].set_title(f'Fission Source - Relative Uncertainty ({temp}K)')

            plt.tight_layout()
            plt.savefig(f'fission_source_comparison_{temp}k.png', dpi=150, bbox_inches='tight')
            plt.show()

        print(f'Temperature {temp}K complete!')
    
    print("\n" + "="*60)
    print("All temperatures processed successfully!")
    print("="*60)
    print("\nGenerated files:")
    for temp in temperatures:
        print(f"  - fission_matrix_{temp}k.npy")
        print(f"  - keff_{temp}k.npy")
        print(f"  - source_{temp}k.npy")

if __name__ == "__main__":
    main()