import os
import numpy as np
import mne
from scipy.io import loadmat

# Input folder with .set files and output folder for exported .txt
input_dir = r"EEG-analysis-during-mental-arithmetic-task\Processed data\eeglab files"
output_dir = r"EEG-analysis-during-mental-arithmetic-task\Processed data\exported"
os.makedirs(output_dir, exist_ok=True)

# Collect all .set files in the input folder
files = [f for f in os.listdir(input_dir) if f.endswith(".set")]

for fname in files:
    filepath = os.path.join(input_dir, fname)
    print(f"Processing {fname}...")

    # Load EEG data from .set file
    raw = mne.io.read_raw_eeglab(filepath, preload=True)

    # Export raw EEG data (timepoints × channels)
    data = raw.get_data()
    np.savetxt(os.path.join(output_dir, fname.replace(".set", "_data.txt")),
               data.T, fmt="%.6f")

    # Load ICA info directly from .set file (MATLAB struct inside)
    mat = loadmat(filepath, simplify_cells=True)
    if "EEG" in mat and "icaweights" in mat["EEG"]:
        icaweights = mat["EEG"]["icaweights"]
        icawinv = mat["EEG"]["icawinv"]

        # Save ICA weights and inverse weights
        np.savetxt(os.path.join(output_dir, fname.replace(".set", "_icaweights.txt")),
                   icaweights, fmt="%.6f")
        np.savetxt(os.path.join(output_dir, fname.replace(".set", "_icawinv.txt")),
                   icawinv, fmt="%.6f")

        # Compute ICA activations = icaweights × icasphere × data
        sphere = mat["EEG"]["icasphere"]
        ica_activity = icaweights @ sphere @ data

        # Save ICA activations (timepoints × components)
        np.savetxt(os.path.join(output_dir, fname.replace(".set", "_ica_activity.txt")),
                   ica_activity.T, fmt="%.6f")
    else:
        print(f"No ICA info found in {fname}")
