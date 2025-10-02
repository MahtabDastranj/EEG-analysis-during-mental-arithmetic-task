import os
import numpy as np
import pywt

base_dir = r'E:\AUT\thesis\files\Processed data\exported'
output_dir = r'E:\AUT\thesis\files\features\CWT'

fs = 500.0
fmin, fmax = 0.5, 45.0
voices_per_oct = 12            # 8–16 typical; 12 is a good default for EEG
omega0 = 6.0                   # Morlet central angular frequency (cycles ~6)
EPS = 1e-12                    # numerical floor for log

freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}


def logspace_frequencies(fmin, fmax, voices_per_oct):
    """Log-spaced frequency vector with 'voices_per_oct' points per octave."""
    n_oct = np.log2(fmax / fmin)
    n = int(np.floor(n_oct * voices_per_oct)) + 1
    return fmin * (2.0 ** (np.arange(n) / voices_per_oct))


def scales_from_frequencies(freqs_hz, fs, omega0):
    """
    SciPy morlet2 uses 's' in *samples*. For Morlet:
      f_c(Hz) = (omega0 * fs) / (2*pi*s)  =>  s = (omega0 * fs) / (2*pi*f_c)
    """
    return (omega0 * fs) / (2.0 * np.pi * freqs_hz)


def cone_of_influence_mask(n_samples, scales):
    """
    COI mask: True where reliable. For Morlet, e-folding time of energy ~ t_e = sqrt(2)*s (samples).
    We mark as invalid any time indices within t_e of the edges (per frequency row).
    Returns: mask shape (n_freqs, n_samples)
    """
    t = np.arange(n_samples)
    mask = np.ones((len(scales), n_samples), dtype=bool)
    for i, s in enumerate(scales):
        te = np.sqrt(2.0) * s
        left_ok = t >= te
        right_ok = (n_samples - 1 - t) >= te
        mask[i, :] = left_ok & right_ok
    return mask


def cwt_power_db_1d(x, fs, freqs_hz, omega0=6.0):
    """
    Compute Morlet CWT power in dB for one 1D signal using PyWavelets.
    freqs_hz: array of target frequencies in Hz
    fs: sampling rate
    omega0 is kept for API symmetry but not directly used in 'cmor'.
    """
    dt = 1.0 / fs

    # Convert desired frequencies -> scales
    # In PyWavelets: scale = center_freq / (freq * dt)
    # center_freq for cmor1.5-1.0 is ~1.0 (PyWavelets convention)
    wavelet = 'cmor1.5-1.0'
    center_freq = pywt.central_frequency(wavelet)
    scales = center_freq / (freqs_hz * dt)

    coeffs, freqs = pywt.cwt(x, scales, wavelet, sampling_period=dt)
    power = np.abs(coeffs) ** 2
    power_db = 10.0 * np.log10(power + 1e-12)
    return power_db


def bandpower_db_from_cwt(power_db, freqs_hz, coi_mask, band_lo, band_hi):
    fmask = (freqs_hz >= band_lo) & (freqs_hz <= band_hi)
    if not np.any(fmask):
        return 0.0

    # Average over time with COI mask, per frequency
    freq_means = []
    for i, use in enumerate(fmask):
        if not use:
            continue
        row = power_db[i, :]
        valid = coi_mask[i, :]
        vals = row[valid]
        if vals.size == 0:
            continue
        freq_means.append(np.mean(vals))
    if len(freq_means) == 0:
        return 0.0

    # Average across frequencies in band
    return float(np.mean(freq_means))


def cwt_features(file_path, fs, fmin, fmax, voices_per_oct, w0, freq_bands):
    data = np.loadtxt(file_path).T  # -> (channels, samples)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    n_channels, n_samples = data.shape

    # Mean-center per channel (safe, stabilizes low-freq estimates)
    data = data - np.mean(data, axis=1, keepdims=True)

    # Prepare frequency grid, scales, and COI mask
    freqs = logspace_frequencies(fmin, fmax, voices_per_oct)
    scales = scales_from_frequencies(freqs, fs, w0)
    coi_mask = cone_of_influence_mask(n_samples, scales)

    features = []
    for ch in range(n_channels):
        x = data[ch, :]
        power_db = cwt_power_db_1d(x, fs, freqs, w0)

        # For each band, compute COI-aware band power (dB)
        for band_name, (lo, hi) in freq_bands.items():
            bp = bandpower_db_from_cwt(power_db, freqs, coi_mask, lo, hi)
            features.append(bp)

    return np.array(features, dtype=float)


for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    data_file = os.path.join(folder_path, 'data.txt')
    if not os.path.exists(data_file):
        print(f"Warning: 'data.txt' not found in {folder_name}")
        continue

    print(f"Processing folder: {folder_name} (using {data_file})...")

    # Extract CWT-based features
    feats = cwt_features(
        data_file,
        fs=fs,
        fmin=fmin, fmax=fmax,
        voices_per_oct=voices_per_oct,
        w0=omega0,
        freq_bands=freq_bands
    )

    if 'rest' in folder_name.lower():
        out_dir = os.path.join(output_dir, 'rest')
    else:
        out_dir = os.path.join(output_dir, 'task')

    os.makedirs(out_dir, exist_ok=True)

    csv_filename = f"{folder_name}.csv"
    csv_path = os.path.join(out_dir, csv_filename)
    np.savetxt(csv_path, feats, delimiter=',', fmt='%.6f')
    print(f"Saved features to {csv_path}")
