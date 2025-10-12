import os
import numpy as np
from scipy.signal import hilbert
from PyEMD import EMD

base_dir = r'E:\AUT\thesis\files\Processed data\exported'
output_dir = r'E:\AUT\thesis\files\features\EMD'

fs = 500.0
fmin, fmax = 0.5, 45.0
max_imfs = 10
normalize = True  # True: relative band energy within 0.5–45 Hz; False: absolute energy

freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}


def emd_decompose(x, max_imfs=10):
    """EMD using PyEMD; returns IMFs as (n_imfs, N)."""
    emd = EMD()
    imfs = emd.emd(x)
    if imfs.shape[0] > max_imfs:
        imfs = imfs[:max_imfs]
    return imfs


def hht_band_features(file_path, fs, freq_bands, fmin, fmax, max_imfs, normalize):
    """
    compute HHT band energies per channel -> (channels, num_bands).
    """
    data = np.loadtxt(file_path).T  # make (channels, samples); your data: (19, 90001)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    num_channels, _ = data.shape

    band_names = list(freq_bands.keys())
    feats = np.zeros((num_channels, len(band_names)), dtype=float)

    for ch in range(num_channels):
        x = data[ch].astype(float)
        x = x - np.mean(x)  # detrend improves EMD numerics

        imfs = emd_decompose(x, max_imfs=max_imfs)
        if imfs.size == 0:
            continue  # leave zeros if no IMFs

        # Hilbert per IMF (axis=1 because imfs is (M, N))
        analytic = hilbert(imfs, axis=1)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic), axis=1)
        inst_freq = (fs / (2.0 * np.pi)) * np.diff(phase, axis=1)  # (M, N-1)
        energy = amp[:, :-1] ** 2  # align with N-1

        # Global valid frequency mask (respect 0.5–45 Hz passband)
        valid = (inst_freq >= fmin) & (inst_freq <= fmax)
        total_energy = np.sum(energy[valid]) + 1e-12

        row = []
        for (lo, hi) in freq_bands.values():
            mask = (inst_freq >= lo) & (inst_freq < hi) & valid
            band_energy = np.sum(energy[mask])
            row.append(band_energy / total_energy if normalize else band_energy)
        feats[ch, :] = row

    return feats  # shape: (channels, num_bands)


for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        data_file = os.path.join(folder_path, 'data.txt')
        if os.path.exists(data_file):
            print(f"Processing folder: {folder_name} (using {data_file})...")

            # Extract HHT features
            feats = hht_band_features(
                data_file, fs, freq_bands, fmin, fmax, max_imfs, normalize
            )

            if 'rest' in folder_name.lower():
                out_dir = os.path.join(output_dir, 'rest')
            else:
                out_dir = os.path.join(output_dir, 'task')
            os.makedirs(out_dir, exist_ok=True)

            # Save features to CSV with folder name as filename
            csv_filename = f"{folder_name}.csv"
            csv_path = os.path.join(out_dir, csv_filename)

            np.savetxt(csv_path, feats, delimiter=',', fmt='%.6f')
            print(f"Saved features to {csv_path}")
        else:
            print(f"Warning: 'data.txt' not found in {folder_name}")
