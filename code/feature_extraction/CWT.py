import numpy as np
from scipy.signal import cwt, morlet2
import os
import pandas as pd

fs = 500
base_dir = r'E:\AUT\thesis\files\Processed data\exported'
output_dir = r'E:\AUT\thesis\files\features\STFT'
participant_ids = range(1, 37)
conditions = ["task", "rest"]
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

f0, f1, fn = 0.5, 45, 100 
# Convert frequencies to scales for SciPy CWT (Morlet central freq ≈6/(2π)≈0.95)
freqs = np.logspace(np.log10(f0), np.log10(f1), fn)
scales = fs / (freqs * (6 / (2 * np.pi)))  # Adjust for Morlet in SciPy

# Results storage
results = []


for pid in participant_ids:
    for cond in conditions:
        file_path = os.path.join(data_dir, f"participant_{pid}_{cond}.txt")
        signal = np.loadtxt(file_path)
        signal -= np.mean(signal)

        # Compute CWT (SciPy)
        cwtmatr = cwt(signal, morlet2, scales, w=6)  # w=6 for time-freq trade-off

        # Power
        power = np.abs(cwtmatr) ** 2

        # Band powers
        band_powers = {}
        for band, (fmin, fmax) in bands.items():
            band_idx = (freqs >= fmin) & (freqs <= fmax)
            band_powers[band] = np.mean(power[band_idx, :]) if np.any(band_idx) else np.nan

        results.append({'participant': pid, 'condition': cond, **band_powers})

df = pd.DataFrame(results)
df.to_csv('cwt_band_powers.csv', index=False)
