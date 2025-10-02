import os
import numpy as np
from scipy.signal import hilbert, firwin, filtfilt, resample_poly
from PyEMD import CEEMDAN

# ====== I/O paths (same style as your STFT script) ======
base_dir = r'E:\AUT\thesis\files\Processed data\exported'
output_dir = r'E:\AUT\thesis\files\features\HHT'  # change folder name as you like

# ====== Parameters ======
fs_in = 500.0
fs_out = 250.0   # downsample for speed (safe for ≤45 Hz analysis)
freq_bands = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}
# CEEMDAN speed/quality trade-off
ceemdan_kwargs = dict(trials=30, noise_strength=0.30, max_imf=6)  # bump to (50,0.3,8) if you want "balanced"

# Instantaneous-frequency guards
AMP_FLOOR_QUANT = 0.05
FREQ_MIN, FREQ_MAX = 0.0, 60.0

# Make output subfolders
os.makedirs(os.path.join(output_dir, 'rest'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'task'), exist_ok=True)

# ---------- helpers ----------
def anti_alias_and_resample(x, fs_in, fs_out):
    """FIR low-pass + zero-phase filter, then rational resample."""
    if fs_in == fs_out:
        return x.astype(np.float32), fs_in
    cutoff = 0.45 * fs_out
    nyq = fs_in / 2.0
    taps = firwin(numtaps=255, cutoff=cutoff/nyq)
    x_f = filtfilt(taps, [1.0], x.astype(np.float32))
    up, down = int(fs_out), int(fs_in)
    g = np.gcd(up, down)
    x_ds = resample_poly(x_f, up//g, down//g).astype(np.float32)
    return x_ds, float(fs_out)

def ceemdan_imfs(x, ceemdan_kw):
    c = CEEMDAN(**ceemdan_kw)
    imfs = c.ceemdan(x)  # [n_imf, T]
    return imfs.astype(np.float32)

def hilbert_amp_if(imfs, fs):
    amps, ifreqs = [], []
    for k in range(imfs.shape[0]):
        s = imfs[k]
        z = hilbert(s)                               # complex analytic
        a = np.abs(z).astype(np.float32)
        phi = np.unwrap(np.angle(z)).astype(np.float32)
        dphi = np.gradient(phi).astype(np.float32)
        f = (fs / (2.0 * np.pi)) * dphi              # Hz
        thr = np.quantile(a, AMP_FLOOR_QUANT)
        bad = (a < max(thr, 1e-12)) | (f < FREQ_MIN) | (f > FREQ_MAX)
        f[bad] = np.nan
        amps.append(a)
        ifreqs.append(f)
    return amps, ifreqs

def band_energy_from_hht(amps, ifreqs, bands):
    vals = []
    for (low, high) in bands.values():
        energy = 0.0
        for a, f in zip(amps, ifreqs):
            idx = (f >= low) & (f < high)
            if np.any(idx):
                aa = a[idx]
                energy += float(np.nansum(aa**2))
        vals.append(energy)
    return vals

# ---------- feature function (same signature/pattern as your STFT one) ----------
def hht_features(file_path, fs_in, fs_out, freq_bands):
    data = np.loadtxt(file_path).T  # -> (channels, samples)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    num_channels, _ = data.shape
    features = []

    for ch in range(num_channels):
        sig = data[ch, :].astype(np.float32)

        # Downsample 500 -> 250 Hz (anti-aliasing included)
        sig, fs_eff = anti_alias_and_resample(sig, fs_in=fs_in, fs_out=fs_out)

        # CEEMDAN decomposition
        imfs = ceemdan_imfs(sig, ceemdan_kwargs)

        # Hilbert amplitude & instantaneous frequency per IMF
        amps, ifreqs = hilbert_amp_if(imfs, fs=fs_eff)

        # HHT-native band energies (sum of a^2 where IF inside band)
        ch_vals = band_energy_from_hht(amps, ifreqs, freq_bands)

        features.extend(ch_vals)

    return np.array(features, dtype=np.float32)  # (channels * num_bands,)

# ---------- iterate subfolders EXACTLY like your STFT script ----------
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        data_file = os.path.join(folder_path, 'data.txt')
        if os.path.exists(data_file):
            print(f"Processing folder: {folder_name} (using {data_file})...")

            feats = hht_features(data_file, fs_in=fs_in, fs_out=fs_out, freq_bands=freq_bands)

            # choose output subfolder by name (rest vs task)
            if 'rest' in folder_name.lower():
                out_dir = os.path.join(output_dir, 'rest')
            else:
                out_dir = os.path.join(output_dir, 'task')

            csv_filename = f"{folder_name}.csv"
            csv_path = os.path.join(out_dir, csv_filename)
            np.savetxt(csv_path, feats, delimiter=',', fmt='%.6f')
            print(f"Saved features to {csv_path}")
        else:
            print(f"Warning: 'data.txt' not found in {folder_name}")
