#!/usr/bin/env python3
"""
Visualize Fission Matrix Training Data
Displays temperature inputs, fission source, and fission matrix
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_data():
    """Load all saved data files."""
    input_temps = np.load("input_temps.npy", allow_pickle=True)
    output_source = np.load("output_source.npy", allow_pickle=True)
    output_fm_normalized = np.load("output_fm_normalized.npy", allow_pickle=True)
    output_keff = np.load("output_keff.npy", allow_pickle=True)
    
    return input_temps, output_source, output_fm_normalized, output_keff

def visualize_run(run_idx, input_temps, output_source, output_fm, keff):
    """Visualize data for a specific run."""
    
    # Convert to proper numpy arrays if needed
    temps = np.array(input_temps[run_idx], dtype=float)
    source = np.array(output_source[run_idx], dtype=float).reshape(17, 17)
    fm = np.array(output_fm[run_idx], dtype=float)
    k = keff[run_idx]
    
    # Determine if single or dual temperature model
    is_dual_temp = (temps.ndim == 3)
    
    # Create figure layout
    if is_dual_temp:
        fig = plt.figure(figsize=(20, 5))
        gs = GridSpec(1, 4, figure=fig, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 5))
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
    
    # Plot fission source
    im3 = ax3.imshow(source, cmap='viridis', interpolation='nearest',
                    origin='lower', aspect='equal')
    ax3.set_title(f'Fission Source Distribution\nk-eff = {k.nominal_value:.5f}')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Plot fission matrix
    ax4 = fig.add_subplot(gs[0, source_plot_idx + 1])
    # Filter out zeros for log scale
    fm_plot = np.where(fm > 0, fm, np.nan)
    im4 = ax4.imshow(fm_plot, cmap='plasma', interpolation='nearest',
                    origin='lower', aspect='equal', norm='log')
    ax4.set_title('Fission Matrix (289×289)\nlog scale')
    ax4.set_xlabel('Source Cell Index')
    ax4.set_ylabel('Destination Cell Index')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Transfer Probability')
    
    plt.suptitle(f'BEAVRS Assembly Data - Run {run_idx+1}', 
                fontsize=16, y=1.02)
    
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