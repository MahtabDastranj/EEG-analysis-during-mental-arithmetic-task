import os
import numpy as np
from scipy.signal import hilbert
from PyEMD import EMD

base_dir = r'E:\AUT\thesis\files\Processed data\exported'
output_dir = r'E:\AUT\thesis\files\features\EMD'

fs = 500.0
fmin, fmax = 0.5, 45.0
max_imfs = 10
# CHANGED: Setting normalize to False for Absolute Power
normalize = False

freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}


def emd_decompose(x, max_imfs=10):
    emd = EMD()
    imfs = emd.emd(x)
    if imfs.shape[0] > max_imfs:
        imfs = imfs[:max_imfs]
    return imfs


def hht_band_features(file_path, fs, freq_bands, fmin, fmax, max_imfs, normalize):
    data = np.loadtxt(file_path).T
    if data.ndim == 1:
        data = data[np.newaxis, :]
    num_channels, num_samples = data.shape  # num_samples is 90001

    band_names = list(freq_bands.keys())
    feats = np.zeros((num_channels, len(band_names)), dtype=float)

    for ch in range(num_channels):
        x = data[ch].astype(float)
        x = x - np.mean(x)

        imfs = emd_decompose(x, max_imfs=max_imfs)
        if imfs.size == 0:
            continue

        analytic = hilbert(imfs, axis=1)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic), axis=1)
        inst_freq = (fs / (2.0 * np.pi)) * np.diff(phase, axis=1)

        # This is the instantaneous energy at every time point
        energy_matrix = amp[:, :-1] ** 2

        # Global valid frequency mask
        valid = (inst_freq >= fmin) & (inst_freq <= fmax)
        total_energy_sum = np.sum(energy_matrix[valid])

        row = []
        for (lo, hi) in freq_bands.values():
            mask = (inst_freq >= lo) & (inst_freq < hi) & valid

            if normalize:
                # Relative Power Logic
                band_energy_sum = np.sum(energy_matrix[mask])
                row.append(band_energy_sum / (total_energy_sum + 1e-12))
            else:
                # Absolute Power Logic (Average Power over the whole signal)
                # We sum the energy in the band and divide by total time points
                # to match the 'np.mean' logic in your STFT/CWT scripts.
                band_power_avg = np.sum(energy_matrix[mask]) / (num_samples - 1)
                row.append(band_power_avg)

        feats[ch, :] = row

    return feats


# Execution loop remains the same...
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        data_file = os.path.join(folder_path, 'data.txt')
        if os.path.exists(data_file):
            print(f"Processing folder: {folder_name} (using {data_file})...")
            feats = hht_band_features(data_file, fs, freq_bands, fmin, fmax, max_imfs, normalize)

            if 'rest' in folder_name.lower():
                out_dir = os.path.join(output_dir, 'rest')
            else:
                out_dir = os.path.join(output_dir, 'task')
            os.makedirs(out_dir, exist_ok=True)

            csv_path = os.path.join(out_dir, f"{folder_name}.csv")
            np.savetxt(csv_path, feats, delimiter=',', fmt='%.6f')
            print(f"Saved Absolute Power features to {csv_path}")
