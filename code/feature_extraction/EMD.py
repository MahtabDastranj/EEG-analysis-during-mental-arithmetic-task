import os
import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
import pandas as pd

base_dir = r'E:\AUT\thesis\files\Processed data\exported'
output_dir = r'E:\AUT\thesis\files\features\EMD'

# EMD parameters (thresholds for stopping criteria)
max_imfs = 10
stopping_threshold = 0.1

freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}


# Function to read EEG data from txt files (same as in STFT)
def read_eeg_data(file_path):
    data = np.loadtxt(file_path).T  # Transpose to (channels, samples)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data


# Function to perform cubic spline interpolation
def cubic_spline_interpolate(x, y):
    cs = CubicSpline(x, y)
    return cs


# Function to find local extrema (minima and maxima)
def find_extrema(signal):
    minima = []
    maxima = []

    # Loop over signal and find local extrema
    for i in range(1, len(signal) - 1):
        if signal[i - 1] < signal[i] and signal[i] > signal[i + 1]:
            maxima.append(i)
        elif signal[i - 1] > signal[i] and signal[i] < signal[i + 1]:
            minima.append(i)

    return minima, maxima


# EMD Decomposition process
def emd_decomposition(signal, max_imfs=max_imfs, stopping_threshold=stopping_threshold):
    imfs = []
    residual = signal
    for i in range(max_imfs):
        # Step 1: Extract local extrema
        minima, maxima = find_extrema(residual)

        # Step 2: Interpolate to form upper and lower envelopes
        lower_envelope = cubic_spline_interpolate(minima, residual[minima])(np.arange(len(residual)))
        upper_envelope = cubic_spline_interpolate(maxima, residual[maxima])(np.arange(len(residual)))

        # Step 3: Compute the mean envelope
        mean_envelope = (lower_envelope + upper_envelope) / 2

        # Step 4: Subtract the mean from the signal
        imf = residual - mean_envelope

        # Step 5: Check stopping criteria
        if np.mean(np.abs(imf)) < stopping_threshold:  # Amplitude threshold check
            break

        # Save the IMF
        imfs.append(imf)

        # Update residual for next IMF extraction
        residual = imf

    return imfs


# Function to process all EEG files (based on STFT file access method)
def process_eeg_files(base_dir, output_dir, max_imfs=max_imfs, stopping_threshold=stopping_threshold):
    # Loop over all subfolders in the base directory
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            data_file = os.path.join(folder_path, 'data.txt')  # Assume 'data.txt' in each folder
            if os.path.exists(data_file):
                print(f"Processing folder: {folder_name} (using {data_file})...")

                # Read the EEG data from the text file
                eeg_data = read_eeg_data(data_file)

                # Process each channel (19 channels)
                imfs_per_file = []
                for channel_data in eeg_data:
                    imfs = emd_decomposition(channel_data, max_imfs, stopping_threshold)
                    imfs_per_file.append(imfs)

                # Determine output directory based on folder name (task vs. rest)
                if 'rest' in folder_name.lower():
                    out_dir = os.path.join(output_dir, 'rest')
                else:
                    out_dir = os.path.join(output_dir, 'task')

                # Ensure the output directory exists
                os.makedirs(out_dir, exist_ok=True)

                # Save IMFs for each channel
                for idx, imfs in enumerate(imfs_per_file):
                    for i, imf in enumerate(imfs):
                        # Convert IMF to DataFrame and save as CSV
                        df = pd.DataFrame(imf)
                        csv_filename = f"{folder_name}_channel_{idx}_imf_{i}.csv"
                        csv_path = os.path.join(out_dir, csv_filename)
                        df.to_csv(csv_path, index=False, header=False)
                        print(f"Saved IMF {i} for channel {idx} in {csv_path}")
            else:
                print(f"Warning: 'data.txt' not found in {folder_name}")
