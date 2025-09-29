import os
import numpy as np
import fcwt
import matplotlib.pyplot as plt

base_dir = r'E:\AUT\thesis\files\Processed data\exported'
output_dir = r'E:\AUT\thesis\files\features\fCWT'

fs = 500
onset_time = 10.0  # Trial onset (s)
fmin, fmax = 0.5, 45
n_cycles = 7.0  # Morlet wavelet cycles
time_step = 0.005  # Time resolution (s)
num_freqs = 50  # Number of frequencies (logarithmic)

freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}


def fcwt_features(file_path, fs, onset_time, fmin, fmax, n_cycles, time_step, num_freqs, freq_bands, plot=False):
    data = np.loadtxt(file_path).T  # Shape: (19, 90001)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    num_channels, num_samples = data.shape

    # Crop to -10s to +25s
    onset_idx = int(onset_time * fs)
    start_idx = max(0, onset_idx - int(10 * fs))  # -10s
    end_idx = onset_idx + int(25 * fs) + 1  # +25s
    if end_idx > num_samples:
        raise ValueError(f"Signal too short in {file_path}; needs {end_idx} samples.")
    data = data[:, start_idx:end_idx]  # Shape: (19, samples)

    # Frequencies: Logarithmic spacing
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), num_freqs)

    # Morlet wavelet
    omega0 = n_cycles
    dt = 1.0 / fs

    # Time decimation
    decim = max(1, round(time_step / dt))
    actual_time_step = decim * dt
    times = np.arange((end_idx - start_idx) // decim) * actual_time_step - 10.0

    # Initialize fCWT
    morlet = fcwt.Morlet(omega0)
    fwt = fcwt.FCWT(morlet, fs, fmin, fmax, num_freqs, True)  # Complex output

    features = []
    for ch in range(num_channels):
        sig = data[ch, :].astype(np.float32)  # fcwt requires float32

        # Compute fCWT
        n_samples = len(sig)
        output = np.zeros((num_freqs, n_samples), dtype=np.complex64)
        try:
            fwt.cwt(sig, fcwt.SINGLE, output)
        except Exception as e:
            raise RuntimeError(f"fCWT failed for channel {ch}: {e}")

        # Power spectrogram
        power = np.abs(output) ** 2  # Shape: (num_freqs, n_samples)

        # Downsample time axis
        power_down = power[:, ::decim]  # Shape: (num_freqs, n_times)

        # Average power over time
        avg_power = np.mean(power_down, axis=1)  # Shape: (num_freqs,)

        # Extract band powers
        ch_features = []
        for band, (low, high) in freq_bands.items():
            idx = np.logical_and(freqs >= low, freqs < high)
            if np.any(idx):
                band_power = np.mean(avg_power[idx])
            else:
                band_power = 0.0
            ch_features.append(band_power)

        features.extend(ch_features)

        # Optional plot for first channel
        if plot and ch == 0:
            plt.figure(figsize=(12, 6))
            extent = [times[0], times[-1], fmin, fmax]
            plt.imshow(power_down, extent=extent, cmap='jet', aspect='auto', origin='lower')
            plt.yscale('log')
            plt.colorbar(label='Power')
            plt.xlabel('Time (s) relative to onset')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'Morlet fCWT Scalogram - Channel 0 ({os.path.basename(file_path)})')
            plt.show()

    return np.array(features)  # Shape: (19 * 5,)


# Iterate over subfolders
rest_count, task_count = 0, 0
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        data_file = os.path.join(folder_path, 'data.txt')
        if os.path.exists(data_file):
            print(f"Processing folder: {folder_name} (using {data_file})...")

            # Extract features
            try:
                feats = fcwt_features(data_file, fs, onset_time, fmin, fmax, n_cycles, time_step, num_freqs, freq_bands)
            except Exception as e:
                print(f"Error processing {data_file}: {e}")
                continue

            # Determine output directory
            if 'rest' in folder_name.lower():
                out_dir = os.path.join(output_dir, 'rest')
                rest_count += 1
            else:
                out_dir = os.path.join(output_dir, 'task')
                task_count += 1

            os.makedirs(out_dir, exist_ok=True)

            # Save features
            csv_filename = f"{folder_name}.csv"
            csv_path = os.path.join(out_dir, csv_filename)
            np.savetxt(csv_path, feats, delimiter=',', fmt='%.6f')
            print(f"Saved features to {csv_path}")
        else:
            print(f"Warning: 'data.txt' not found in {folder_name}")

print(f"Processed {rest_count} rest files and {task_count} task files.")
if rest_count != 36 or task_count != 36:
    print(f"Warning: Expected 36 rest and 36 task files; found {rest_count} rest and {task_count} task.")