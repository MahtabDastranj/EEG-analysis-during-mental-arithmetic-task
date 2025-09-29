import numpy as np
import os
from scipy.signal import cwt, morlet2
import matplotlib.pyplot as plt


def extract_morlet_cwt_features(dir_path, fs, onset_time=10.0, fmin=1, fmax=100, n_cycles=7.0, time_step=0.005,
                                average=True, plot_average=False):
    """
    Extracts features using Morlet CWT from multiple .txt EEG files in a directory, following the specified parameters.

    Assumptions:
    - Each .txt file contains a 1D EEG signal (loaded via np.loadtxt(file)).
    - If the file loads as 2D, takes the first column as the signal.
    - All signals have the same length and sampling rate fs.
    - Trial onset is at 'onset_time' seconds from the start of the signal.
    - Signal includes at least 10s before and 25s after onset.
    - The 72 files may represent trials, participants, or task blocks; the code averages across all if average=True.
    - No specific grouping for conditions/participants; modify the code if needed for subgroup averaging.

    Parameters:
    - dir_path: Path to directory containing the .txt files.
    - fs: Sampling frequency in Hz (must be provided, e.g., 200 Hz for 5ms native resolution).
    - onset_time: Time in seconds from signal start to trial onset (default: 10s).
    - fmin, fmax: Frequency range (1-100 Hz).
    - n_cycles: Number of cycles for Morlet wavelet (7 as specified).
    - time_step: Desired time resolution in seconds (0.005s = 5ms).
    - average: If True, average power across all files (as per "averaged for each task condition").
    - plot_average: If True and average=True, plots the average scalogram.

    Returns:
    - If average=True: average_power (freqs x times), freqs, times
    - Else: list of powers (each freqs x times), freqs, times

    Raises:
    - ValueError: If no files found, invalid parameters, or signals too short.

    Example usage:
    >>> average_power, freqs, times = extract_morlet_cwt_features('/path/to/your/directory', fs=200.0, plot_average=True)
    """
    # Input validation
    if not os.path.isdir(dir_path):
        raise ValueError("Provided dir_path is not a valid directory.")
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("Invalid frequency range.")
    if n_cycles < 1:
        raise ValueError("n_cycles should be at least 1 (typically 7 for Morlet).")
    if time_step <= 0:
        raise ValueError("time_step must be positive.")

    # Get all .txt files
    txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
    if len(txt_files) == 0:
        raise ValueError("No .txt files found in the directory.")
    print(f"Found {len(txt_files)} .txt files.")

    # Frequencies: 1 to 100 Hz in 1 Hz steps
    freqs = np.arange(fmin, fmax + 1)

    # Morlet parameter w = n_cycles
    w = n_cycles

    # Central frequency for scale calculation
    central_freq = (w + np.sqrt(2 + w ** 2)) / (4 * np.pi)

    # Scales corresponding to frequencies
    scales = central_freq * fs / freqs

    # Decimation step for desired time resolution
    input_dt = 1.0 / fs
    decim = max(1, round(time_step / input_dt))
    actual_time_step = decim * input_dt
    print(f"Using decim={decim}, actual time step: {actual_time_step:.4f} s")

    # Process each file
    powers = []
    for file in txt_files:
        file_path = os.path.join(dir_path, file)
        try:
            signal = np.loadtxt(file_path)
            if signal.ndim == 2:
                signal = signal[:, 0]  # Assume first column is the signal
            elif signal.ndim != 1:
                raise ValueError(f"Invalid signal dimension in {file}.")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

        # Crop to -10s to +25s around onset
        onset_idx = int(onset_time * fs)
        start_idx = max(0, onset_idx - int(10 * fs))
        end_idx = onset_idx + int(25 * fs) + 1  # Inclusive
        if end_idx > len(signal):
            print(f"Warning: Signal in {file} too short; skipping.")
            continue

        signal_crop = signal[start_idx:end_idx]

        # Perform CWT
        try:
            coef = cwt(signal_crop, lambda M: morlet2(M, s=1.0, w=w), scales)
        except Exception as e:
            print(f"Error during CWT for {file}: {e}")
            continue

        # Power
        power = np.abs(coef) ** 2

        # Downsample time axis
        power_down = power[:, ::decim]

        powers.append(power_down)

    if len(powers) == 0:
        raise ValueError("No valid signals processed.")

    # Times relative to onset (for the downsampled TFR)
    num_times = powers[0].shape[1]  # Assume all same shape
    times = np.arange(num_times) * actual_time_step - 10.0

    if average:
        # Average across files (or trials/participants)
        average_power = np.mean(powers, axis=0)
        if plot_average:
            plt.figure(figsize=(12, 6))
            extent = [times[0], times[-1], freqs[0], freqs[-1]]
            plt.imshow(average_power, extent=extent, cmap='jet', aspect='auto', origin='lower')
            plt.colorbar(label='Average Power')
            plt.xlabel('Time (s) relative to onset')
            plt.ylabel('Frequency (Hz)')
            plt.title('Average Morlet CWT Scalogram across Files')
            plt.show()
        return average_power, freqs, times
    else:
        return powers, freqs, times