import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- Auto-detect OS and set default data directory ---
if sys.platform.startswith("win"):
    default_dir = os.path.join(os.environ["USERPROFILE"], "OneDrive", "Documents", "Monte_Carlo", "Test")
else:
    default_dir = os.path.expanduser("~/Monte_Carlo/Test")

data_dir = default_dir
input_file = os.path.join(data_dir, "input_temps.npy")
output_file = os.path.join(data_dir, "output_flux.npy")

# --- Load the .npy files ---
if os.path.exists(input_file):
    input_temps = np.load(input_file)
    print(f"Input data shape (temperatures): {input_temps.shape}")
else:
    input_temps = None
    print("Input temperatures file not found.")

if os.path.exists(output_file):
    output_flux = np.load(output_file)
    print(f"Output data shape (flux matrices): {output_flux.shape}")
else:
    raise FileNotFoundError(f"{output_file} not found.")

num_samples = output_flux.shape[0]

for i in range(num_samples):
    plt.figure(figsize=(12,5))

    # --- Plot input temperature ---
    plt.subplot(1, 2, 1)
    if input_temps is not None and input_temps.size > 0:
        temp = input_temps[i % len(input_temps)]  # Wrap around if fewer inputs
        if temp.ndim == 1:
            side_len = int(np.sqrt(temp.size))
            if side_len * side_len == temp.size:
                temp = temp.reshape((side_len, side_len))
        if temp.ndim == 2:
            plt.imshow(temp, origin='lower', cmap='hot', interpolation='none')
            plt.colorbar(label='Temperature')
        else:
            plt.plot(temp)
        plt.title(f"Input temperature #{i+1}")
    else:
        plt.text(0.5, 0.5, "No input temperatures", ha='center', va='center')
        plt.axis('off')

    # --- Plot output flux ---
    plt.subplot(1, 2, 2)
    flux = output_flux[i]
    side_len = int(np.sqrt(flux.size))
    if side_len * side_len == flux.size:
        flux_matrix = flux.reshape((side_len, side_len))
    else:
        flux_matrix = flux

    if flux_matrix.ndim == 2:
        plt.imshow(flux_matrix, origin='lower', cmap='hot', interpolation='none')
        plt.colorbar(label='Flux')
    else:
        plt.plot(flux_matrix)
    plt.title(f"Output flux #{i+1}")

    plt.tight_layout()
    plt.show(block=True)
