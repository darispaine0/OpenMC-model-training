#!/usr/bin/env python3
"""
Visualize Fission Matrix Training Data
Displays temperature inputs, fission source, and fission matrix

Changes:
- Source colorbar spans only min/max of fuel cells (ignores zeros from guide tubes)
- Fission matrix applies cutoff (default 1e-4) and uses LogNorm so tiny values do not dominate colormap
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

def load_data():
    """Load all saved data files."""
    input_temps = np.load("input_temps.npy", allow_pickle=True)
    output_source = np.load("output_source.npy", allow_pickle=True)
    output_fm_normalized = np.load("output_fm_normalized.npy", allow_pickle=True)
    output_keff = np.load("output_keff.npy", allow_pickle=True)
    
    return input_temps, output_source, output_fm_normalized, output_keff

def _safe_keff_value(k):
    """Return a float-like representation of keff for display."""
    try:
        # openmc-style object with nominal_value
        if hasattr(k, "nominal_value"):
            return float(k.nominal_value)
        # numpy scalar or float
        if np.isscalar(k):
            return float(k)
        # array-like
        arr = np.array(k)
        if arr.size == 1:
            return float(arr.squeeze())
    except Exception:
        pass
    return float("nan")

def visualize_run(run_idx, input_temps, output_source, output_fm, keff, fm_cutoff=1e-4):
    """Visualize data for a specific run.

    fm_cutoff: values below this are masked/omitted in the log plot.
    """
    # Convert to proper numpy arrays if needed
    temps = np.array(input_temps[run_idx], dtype=float)
    source_flat = np.array(output_source[run_idx], dtype=float).ravel()
    # infer mesh size from source length (must be perfect square)
    n_cells = int(np.round(np.sqrt(source_flat.size)))
    if n_cells * n_cells != source_flat.size:
        raise ValueError(f"Source array length {source_flat.size} is not a perfect square.")
    source = source_flat.reshape((n_cells, n_cells))
    
    fm = np.array(output_fm[run_idx], dtype=float)
    # fm can be either (n^2, n^2) or flattened; try to handle both
    try:
        if fm.ndim == 1:
            # flatten to n_mesh*n_mesh expected
            expected = n_cells * n_cells
            if fm.size == expected * expected:
                fm = fm.reshape((expected, expected))
            else:
                # try to make a square from length
                side = int(np.round(np.sqrt(fm.size)))
                if side * side == fm.size:
                    fm = fm.reshape((side, side))
                else:
                    raise ValueError("Unexpected fm shape/size.")
        elif fm.ndim == 2:
            pass
        else:
            raise ValueError("Unexpected fm ndim.")
    except Exception as e:
        raise RuntimeError(f"Cannot interpret fm data: {e}")
    
    k = keff[run_idx]
    k_val = _safe_keff_value(k)
    
    # Determine if single or dual temperature model
    is_dual_temp = (temps.ndim == 3 and temps.shape[2] >= 2)
    
    # Create figure layout (use constrained_layout to avoid overlaps)
    if is_dual_temp:
        fig = plt.figure(figsize=(20, 5), constrained_layout=True)
        gs = GridSpec(1, 4, figure=fig, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 5), constrained_layout=True)
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # Plot temperature(s)
    if is_dual_temp:
        # Fuel temperature
        ax1 = fig.add_subplot(gs[0, 0])
        fuel_temps = temps[:, :, 0]
        im1 = ax1.imshow(fuel_temps, cmap='hot', interpolation='nearest', 
                        origin='lower', aspect='equal')
        ax1.set_title(f'Fuel Temperature (K)\nRun {run_idx+1}\nMin: {fuel_temps.min():.1f}, Max: {fuel_temps.max():.1f}')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Moderator temperature
        ax2 = fig.add_subplot(gs[0, 1])
        mod_temps = temps[:, :, 1]
        im2 = ax2.imshow(mod_temps, cmap='cool', interpolation='nearest',
                        origin='lower', aspect='equal')
        ax2.set_title(f'Moderator Temperature (K)\nRun {run_idx+1}\nMin: {mod_temps.min():.1f}, Max: {mod_temps.max():.1f}')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Row')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Fission source
        ax3 = fig.add_subplot(gs[0, 2])
        source_plot_idx = 2
    else:
        # Single temperature
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(temps, cmap='hot', interpolation='nearest',
                        origin='lower', aspect='equal')
        ax1.set_title(f'Temperature (K)\nRun {run_idx+1}\nMin: {temps.min():.1f}, Max: {temps.max():.1f}')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Fission source
        ax3 = fig.add_subplot(gs[0, 1])
        source_plot_idx = 1
    
    # --- Fission source plotting ---
    # Determine vmin/vmax based on "fuel" cells only (exclude zeros/guide tubes)
    source_positive = source[source > 0.0]
    if source_positive.size == 0:
        # no positive entries (degenerate) -> fallback to full range
        vmin, vmax = float(source.min()), float(source.max())
    else:
        vmin, vmax = float(source_positive.min()), float(source_positive.max())
        # if vmin==vmax, expand a little for color mapping
        if vmin == vmax:
            vmin = vmin * 0.999 if vmin != 0 else -1e-8
            vmax = vmax * 1.001 if vmax != 0 else 1e-8
    
    im3 = ax3.imshow(source, cmap='viridis', interpolation='nearest',
                    origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
    ax3.set_title(f'Fission Source Distribution\nk-eff = {np.nan if np.isnan(k_val) else k_val:.5f}')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Source (fuel-only scaling)')
    
    # --- Fission matrix plotting ---
    ax4 = fig.add_subplot(gs[0, source_plot_idx + 1])
    # Apply cutoff: mask values below cutoff (these will appear blank)
    cutoff = fm_cutoff
    fm_plot = np.array(fm, dtype=float)
    # For log plotting we must have positive vmin; mask values below cutoff
    fm_masked = np.where(fm_plot >= cutoff, fm_plot, np.nan)
    
    # find a vmax for color scale among values >= cutoff
    if np.isfinite(fm_masked).any():
        vmax_fm = np.nanmax(fm_masked)
        # Ensure vmax_fm > cutoff
        if vmax_fm <= cutoff:
            vmax_fm = cutoff * 10.0
        norm = LogNorm(vmin=cutoff, vmax=vmax_fm)
        im4 = ax4.imshow(fm_masked, cmap='plasma', interpolation='nearest',
                         origin='lower', aspect='equal', norm=norm)
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label('Transfer Probability (cutoff {:.1e})'.format(cutoff))
    else:
        # If everything is below cutoff, draw the raw matrix but gray it out
        im4 = ax4.imshow(fm_plot, cmap='plasma', interpolation='nearest',
                         origin='lower', aspect='equal')
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label('Transfer Probability (no values above cutoff)')
    
    ax4.set_title('Fission Matrix (n×n)\nlog scale (masked small values)')
    ax4.set_xlabel('Source Cell Index')
    ax4.set_ylabel('Destination Cell Index')
    
    plt.suptitle(f'BEAVRS Assembly Data - Run {run_idx+1}', fontsize=16)

    
    return fig

def main():
    """Main visualization function."""
    
    print("Loading data...")
    try:
        input_temps, output_source, output_fm, output_keff = load_data()
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Make sure you've run the simulation first.")
        print(f"Missing file: {e.filename}")
        return
    
    n_runs = len(input_temps)
    print(f"Found {n_runs} runs in data files.")
    
    # Interactive loop
    while True:
        try:
            run_choice = input(f"\nEnter run number to visualize (1-{n_runs}, or 'q' to quit): ").strip()
            
            if run_choice.lower() == 'q':
                print("Exiting visualization.")
                break
            
            run_idx = int(run_choice) - 1
            
            if run_idx < 0 or run_idx >= n_runs:
                print(f"Invalid run number. Please choose between 1 and {n_runs}.")
                continue
            
            print(f"Visualizing run {run_idx+1}...")
            fig = visualize_run(run_idx, input_temps, output_source, output_fm, output_keff)
            plt.show()
            
            # Ask if user wants to save
            save_choice = input("Save this figure? (y/n): ").strip().lower()
            if save_choice == 'y':
                filename = f"run_{run_idx+1}_visualization.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved as {filename}")
            
            plt.close(fig)
            
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
        except KeyboardInterrupt:
            print("\nExiting visualization.")
            break
        except Exception as e:
            print(f"Error visualizing run {run_idx+1}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
