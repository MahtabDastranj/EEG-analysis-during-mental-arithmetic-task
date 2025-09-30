"""
Fast CWT (FCWT) EEG Feature Extraction with Morlet
- Walks subfolders in base_dir; each folder contains 'data.txt' shaped (samples, channels)
- Transposes to (channels, samples) to match your STFT code
- Computes complex Morlet CWT via 'fcwt' (fast) with ~log-spaced freq grid (voices/octave)
- Applies a Cone of Influence (COI) mask to avoid edge bias
- Extracts band powers (dB) for delta/theta/alpha/beta/gamma per channel
- Saves one CSV per folder into output_dir/{task|rest}/<folder_name>.csv

Requires:
    pip install fcwt numpy
"""

import os
import inspect
import numpy as np

# ---------------------- Paths (same pattern you used) ----------------------
base_dir   = r'E:\AUT\thesis\files\Processed data\exported'
output_dir = r'E:\AUT\thesis\files\features\fCWT'


# ---------------------- Analysis parameters ----------------------
fs = 500.0
fmin, fmax = 0.5, 45.0
voices_per_oct = 12          # frequency resolution (8–16 typical)
omega0 = 6.0                 # Morlet central angular frequency (# of cycles ~6) for COI calc
EPS = 1e-12                  # to avoid log(0) in dB

freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}

# ---------------------- Helpers ----------------------
def _logspace_freqs_fallback(fmin, fmax, nv, K=None):
    """Log-spaced frequencies with nv voices/octave. If K specified, geomspace to K points."""
    fmin = float(fmin); fmax = float(fmax); nv = int(nv)
    if K is not None:
        return np.geomspace(fmin, fmax, int(K))
    octaves = np.log2(fmax / fmin)
    K = int(np.floor(octaves * nv)) + 1
    return fmin * (2.0 ** (np.arange(K) / nv))

def fcwt_cwt_1d(x, fs, fmin, fmax, voices_per_oct):
    """
    Robust FCWT wrapper returning (W, freqs) with shapes (n_freqs, n_samples), (n_freqs,).
    Adapts to multiple fcwt variants:
      - Scales signatures: (fmin,fmax,nv,fs) or (f0,f1,fn) or (fmin,fmax,nv)
      - Transformer classes: FCWT / CWT, methods: cwt / forward / transform / __call__
    """
    try:
        import fcwt  # local import so the script still starts without fcwt installed
    except ImportError as e:
        raise ImportError(
            "The 'fcwt' package is required for this script.\n"
            "Install it with:  pip install fcwt\n"
            f"Original error: {e}"
        )

    x    = np.asarray(x, dtype=np.float64)
    fs   = float(fs)
    fmin = float(fmin)
    fmax = float(fmax)
    nv   = int(voices_per_oct)

    # -------- Build Scales dynamically (via signature introspection) --------
    Scales = getattr(fcwt, "Scales", None)
    if Scales is None:
        raise RuntimeError("fcwt.Scales not found in your fcwt build.")

    scales = None
    last_scale_errs = []

    # Try to inspect the __init__ signature (works on many wheels)
    try:
        sig = inspect.signature(Scales.__init__)
        params = [p for name, p in sig.parameters.items() if name != 'self']
        name_map = {
            'f0': fmin, 'fmin': fmin, 'f1': fmax, 'fmax': fmax,
            'fn': nv, 'nv': nv, 'nvoices': nv,
            'fs': fs, 'sampling_rate': fs, 'sr': fs
        }
        args = []
        for p in params:
            val = name_map.get(p.name, None)
            if val is None:
                if p.default is not inspect._empty:
                    val = p.default
                else:
                    raise TypeError(f"Don't know how to fill Scales param '{p.name}'")
            args.append(val)
        scales = Scales(*args)
    except Exception as e_inspect:
        last_scale_errs.append(f"introspected call failed: {repr(e_inspect)}")

    # If introspection failed, brute-force common patterns
    if scales is None:
        for args in [
            (fmin, fmax, nv, fs),
            (fmin, fmax, nv),
            (fmin, fmax, nv),  # keep trying (some builds alias names)
        ]:
            try:
                scales = Scales(*args)
                break
            except Exception as e_try:
                last_scale_errs.append(repr(e_try))

    if scales is None:
        raise RuntimeError(
            "Could not construct fcwt.Scales on this build.\n"
            "Tried introspected and common signatures.\n"
            "Errors:\n  - " + "\n  - ".join(last_scale_errs)
        )

    # -------- Wavelet (optional) --------
    wavelet = None
    Morlet  = getattr(fcwt, "Morlet", None)
    if Morlet is not None:
        try:
            wavelet = Morlet()     # default ctor on many builds
        except Exception:
            try:
                wavelet = Morlet(6.0)  # some allow a cycles param
            except Exception:
                wavelet = None

    # -------- Transformer construction (try several) --------
    Transformer = getattr(fcwt, "FCWT", None) or getattr(fcwt, "CWT", None)
    if Transformer is None:
        raise RuntimeError("fcwt.FCWT / fcwt.CWT not found in this build.")

    transformer = None
    ctor_trials = [
        ((scales, fs, wavelet), {}),
        ((scales, wavelet, fs), {}),
        ((scales, fs), {}),
        ((scales,), {}),
        ((wavelet, scales, fs), {}),
        ((), {"scales": scales, "fs": fs, "wavelet": wavelet}),  # if kwargs supported
    ]
    ctor_errs = []
    for args, kwargs in ctor_trials:
        try:
            kw = {k: v for k, v in kwargs.items() if v is not None}
            transformer = Transformer(*args, **kw)
            break
        except Exception as e:
            ctor_errs.append(repr(e))
            transformer = None
    if transformer is None:
        raise RuntimeError("Could not construct fcwt transformer. Constructor errors:\n  - " +
                           "\n  - ".join(ctor_errs))

    # -------- Execute transform (method names vary) --------
    W = None
    call_err = None
    for meth in ("cwt", "forward", "transform", "__call__"):
        fn = getattr(transformer, meth, None)
        if fn is None:
            continue
        try:
            W = fn(x)
            break
        except Exception as e:
            call_err = e
            W = None
    if W is None:
        raise RuntimeError(f"Could not run CWT on this fcwt transformer. Last error: {repr(call_err)}")

    W = np.asarray(W)
    if W.ndim != 2:
        raise RuntimeError(f"fcwt returned shape {W.shape}, expected 2D (freqs x time)")

    # -------- Retrieve/construct frequency vector --------
    freqs = getattr(transformer, "frequencies", None)
    if callable(freqs):
        try:
            freqs = freqs(fs)
        except TypeError:
            freqs = freqs()
    if freqs is None:
        freqs = getattr(scales, "frequencies", None)
        if callable(freqs):
            try:
                freqs = freqs(fs)
            except TypeError:
                freqs = freqs()
    if freqs is None and hasattr(fcwt, "freqs"):
        try:
            freqs = fcwt.freqs(scales, fs)
        except Exception:
            freqs = None
    if freqs is None:
        # Last resort: synthesize a sensible log-spaced Hz vector
        freqs = _logspace_freqs_fallback(fmin, fmax, nv, K=W.shape[0])

    freqs = np.asarray(freqs, dtype=float)
    if freqs.shape[0] != W.shape[0]:
        freqs = _logspace_freqs_fallback(fmin, fmax, nv, K=W.shape[0])

    return W, freqs

def scales_from_freqs(freqs_hz, fs, omega0):
    """
    Morlet scale (in samples) used only for COI calculation:
      f_c(Hz) ≈ (omega0*fs) / (2π s)  =>  s ≈ (omega0*fs) / (2π f_c)
    """
    return (omega0 * fs) / (2.0 * np.pi * np.maximum(freqs_hz, 1e-12))

def cone_of_influence_mask(n_samples, scales):
    """
    COI mask True where reliable. For Morlet, energy e-folding time ≈ t_e = sqrt(2)*s (samples).
    """
    t = np.arange(n_samples)
    mask = np.ones((len(scales), n_samples), dtype=bool)
    for i, s in enumerate(scales):
        te = np.sqrt(2.0) * s
        left_ok = t >= te
        right_ok = (n_samples - 1 - t) >= te
        mask[i, :] = left_ok & right_ok
    return mask

def bandpower_db_from_cwt(power_db, freqs_hz, coi_mask, lo, hi):
    """
    Average dB power over time (inside COI) per frequency, then across freqs in [lo, hi].
    Returns one scalar.
    """
    fmask = (freqs_hz >= lo) & (freqs_hz <= hi)
    if not np.any(fmask):
        return 0.0
    vals = []
    rows = np.where(fmask)[0]
    for i in rows:
        valid = coi_mask[i, :]
        if not np.any(valid):
            continue
        vals.append(np.mean(power_db[i, valid]))
    return float(np.mean(vals)) if len(vals) else 0.0

def cwt_fcwt_features(file_path, fs, fmin, fmax, voices_per_oct, omega0, freq_bands):
    """
    Load data.txt -> (channels, samples), mean-center; CWT via fcwt; COI-masked band powers (dB).
    Returns flat vector (n_channels * n_bands).
    """
    data = np.loadtxt(file_path).T  # your convention: (channels, samples)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    n_channels, n_samples = data.shape

    # Mean-center per channel
    data = data - np.mean(data, axis=1, keepdims=True)

    features = []
    for ch in range(n_channels):
        x = data[ch, :]

        # FCWT CWT (complex)
        W, freqs = fcwt_cwt_1d(x, fs, fmin, fmax, voices_per_oct)  # (n_freqs, n_samples), (n_freqs,)

        # Power in dB
        power_db = 10.0 * np.log10(np.abs(W) ** 2 + EPS)

        # COI
        scales = scales_from_freqs(freqs, fs, omega0)
        coi_mask = cone_of_influence_mask(n_samples, scales)

        # Band powers (dB) for this channel
        for _, (lo, hi) in freq_bands.items():
            bp = bandpower_db_from_cwt(power_db, freqs, coi_mask, lo, hi)
            features.append(bp)

    return np.array(features, dtype=float)

# ---------------------- Batch over subfolders (same logic you used) ----------------------
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    data_file = os.path.join(folder_path, 'data.txt')
    if not os.path.exists(data_file):
        print(f"Warning: 'data.txt' not found in {folder_name}")
        continue

    print(f"Processing folder: {folder_name} (using {data_file})...")

    try:
        feats = cwt_fcwt_features(
            data_file,
            fs=fs,
            fmin=fmin, fmax=fmax,
            voices_per_oct=voices_per_oct,
            omega0=omega0,
            freq_bands=freq_bands
        )
    except ImportError as e:
        raise SystemExit(str(e))
    except Exception as e:
        print(f"[WARN] Skipping {folder_name} due to error: {e}")
        continue

    # Decide output subdir (task/rest) by folder name — same heuristic you used
    out_dir = os.path.join(output_dir, 'rest' if 'rest' in folder_name.lower() else 'task')
    os.makedirs(out_dir, exist_ok=True)

    # Save features to CSV with folder name as filename
    csv_filename = f"{folder_name}.csv"
    csv_path = os.path.join(out_dir, csv_filename)
    np.savetxt(csv_path, feats, delimiter=',', fmt='%.6f')
    print(f"Saved features to {csv_path}")
