import os
import numpy as np
import pywt

base_dir   = r'E:\AUT\thesis\files\Processed data\exported'
output_dir = r'E:\AUT\thesis\files\features\CWT'

freq_bands = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0)
}

fs = 500.0
wavelet_name = 'morl'
num_freqs = 96           # number of analysis frequencies between 0.5–45 Hz (log-spaced)
fmin, fmax = 0.5, 45.0   # target analysis frequency range
normalize = False


def compute_scales_for_freqs(freqs_hz, wavelet_name, fs):
    """
    Convert desired analysis frequencies (Hz) to PyWavelets scales.
    scales = fc / (f * dt), with dt = 1/fs and fc from PyWavelets for the given wavelet.
    """
    wavelet = pywt.ContinuousWavelet(wavelet_name)
    fc = pywt.central_frequency(wavelet)  # ~0.8125 for 'morl'
    dt = 1.0 / fs
    return fc / (freqs_hz * dt)


def cwt_features(file_path, fs, freq_bands):
    data = np.loadtxt(file_path).T
    if data.ndim == 1:
        data = data[np.newaxis, :]
    num_channels, num_samples = data.shape

    # Log-spaced frequency grid and matching scales for 0.5–45 Hz
    freqs = np.geomspace(fmin, fmax, num=num_freqs)
    scales = compute_scales_for_freqs(freqs, wavelet_name, fs)

    features = []  # rows=channels, cols=bands
    for ch in range(num_channels):
        sig = data[ch, :]
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

        # CWT with correct sampling_period so frequency axis is in Hz
        coeff, freq_out = pywt.cwt(sig, scales, wavelet_name, sampling_period=1.0/fs)

        # Power scalogram and average over time (mirrors your STFT avg over time)
        power = np.abs(coeff) ** 2            # shape: (num_freqs, N)
        avg_power = np.mean(power, axis=1)    # shape: (num_freqs,)

        # Band aggregation: mean over frequency bins within each band
        ch_features = []
        for band, (low, high) in freq_bands.items():
            idx = np.logical_and(freq_out >= low, freq_out < high)
            if np.any(idx):
                band_power = float(np.mean(avg_power[idx]))
            else:
                band_power = 0.0
            ch_features.append(band_power)

        # normalization to relative energy per channel across 0.5–45 Hz
        if normalize:
            total = sum(ch_features)
            if total > 0:
                ch_features = [v / total for v in ch_features]

        features.append(ch_features)

    return np.array(features)


for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        data_file = os.path.join(folder_path, 'data.txt')
        if os.path.exists(data_file):
            print(f"Processing folder: {folder_name} (using {data_file})...")

            feats = cwt_features(data_file, fs, freq_bands)

            if 'rest' in folder_name.lower():
                out_dir = os.path.join(output_dir, 'rest')
            else:
                out_dir = os.path.join(output_dir, 'task')
            os.makedirs(out_dir, exist_ok=True)

            csv_filename = f"{folder_name}.csv"
            csv_path = os.path.join(out_dir, csv_filename)

            # rows (channels) and columns (frequency bands)
            np.savetxt(csv_path, feats, delimiter=',', fmt='%.6f')
            print(f"Saved features to {csv_path}")
        else:
            print(f"Warning: 'data.txt' not found in {folder_name}")
