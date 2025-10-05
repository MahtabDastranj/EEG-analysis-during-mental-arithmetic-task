import os
import numpy as np
from scipy import signal

base_dir = r'E:\AUT\thesis\files\Processed data\exported'
output_dir = r'E:\AUT\thesis\files\features\STFT'

# STFT parameters
fs = 500
window_size = 250  # Window length for STFT (adjust if needed, ~0.5 sec at 250 Hz) frequency resolution = 2
overlap = window_size // 2  # 50% overlap for smoother spectrogram

freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)  # With respect to nyquist frequency
}


def stft_features(file_path, fs, window_size, overlap, freq_bands):
    data = np.loadtxt(file_path).T  # Transpose to (channels, samples)(19, 90001)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    num_channels, num_samples = data.shape  # Should be (19, 90001)

    features = []

    for ch in range(num_channels):
        sig = data[ch, :]

        # Compute STFT (returns frequencies f, times t, complex spectrogram Zxx)
        f, t, Zxx = signal.stft(sig, fs=fs, nperseg=window_size, noverlap=overlap)
        # Zxx: Complex-valued spectrogram, shape (freqs, times), where freqs = nperseg/2 + 1 (due to Nyquist limit)

        # Compute power spectrogram (magnitude squared)
        power = np.abs(Zxx) ** 2  # Shape: (freqs, times)

        # Average power over time for each frequency
        avg_power = np.mean(power, axis=1)  # Shape: (freqs,)

        # Extract band powers
        ch_features = []
        for band, (low, high) in freq_bands.items():
            # Find frequency indices in band
            idx = np.logical_and(f >= low, f < high)
            if np.any(idx):
                band_power = np.mean(avg_power[idx])
            else:
                band_power = 0.0  # If no frequencies in band
            ch_features.append(band_power)

        features.append(ch_features)

    return np.array(features)  # Shape: (19, num_bands)


# Iterate over all subfolders in the base directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        data_file = os.path.join(folder_path, 'data.txt')
        if os.path.exists(data_file):
            print(f"Processing folder: {folder_name} (using {data_file})...")

            # Extract features
            feats = stft_features(data_file, fs, window_size, overlap, freq_bands)

            # Determine output directory based on folder name
            if 'rest' in folder_name.lower():
                out_dir = os.path.join(output_dir, 'rest')
            else:
                out_dir = os.path.join(output_dir, 'task')

            # Save features to CSV with folder name as filename
            csv_filename = f"{folder_name}.csv"
            csv_path = os.path.join(out_dir, csv_filename)

            # Save the features with 19 rows (channels) and 5 columns (frequency bands)
            np.savetxt(csv_path, feats, delimiter=',', fmt='%.6f')
            print(f"Saved features to {csv_path}")
        else:
            print(f"Warning: 'data.txt' not found in {folder_name}")
