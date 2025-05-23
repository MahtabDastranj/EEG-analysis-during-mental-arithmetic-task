"""
Generate .locs files for each EDF recording by reordering channel locations
based on your default template 'Standard-10-20-Cap19_thesis.locs'.

This version strips the 'EEG ' prefix from channel names before matching,
and ignores 'A2-A1' & 'ECG ECG', ensuring you get the 19 remaining channels.
"""
import os
import argparse
import pandas as pd
import mne


def load_default_locs(default_path):
    """
    Load the default .locs template into a DataFrame.
    Expects whitespace-delimited columns: orig_idx, theta, radius, name
    Adds lowercase names for matching.
    """
    df = pd.read_csv(
        default_path,
        sep=r"\s+",
        header=None,
        names=["orig_idx", "theta", "radius", "name"]
    )
    df['name_lower'] = df['name'].str.lower()
    return df


def create_locs_for_edf(edf_path, template_df):
    """
    Read the EDF to get its channel names, strip any 'EEG ' prefix,
    exclude unwanted channels, then filter and reorder the template.
    Returns a DataFrame with columns: idx, theta, radius, name
    """
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    raw_chs = raw.ch_names

    # Strip 'EEG ' prefix if present
    cleaned = [ch[4:] if ch.lower().startswith('eeg ') else ch for ch in raw_chs]

    # Channels to ignore after stripping
    ignore_lower = {'a2-a1', 'ecg ecg'}
    present_ignored = [ch for ch in cleaned if ch.lower() in ignore_lower]
    if present_ignored:
        print(f"ℹ️ Ignoring channels in {os.path.basename(edf_path)}: {present_ignored}")

    # Keep only channels not in ignore list
    filtered = [ch for ch in cleaned if ch.lower() not in ignore_lower]
    filtered_lower = [ch.lower() for ch in filtered]

    # Match template rows case-insensitively
    matched = template_df[template_df['name_lower'].isin(filtered_lower)].copy()

    # Warn about any filtered channels not found in template
    missing = [ch for ch in filtered if ch not in matched['name'].tolist()]
    if missing:
        print(f"⚠️ EDF {os.path.basename(edf_path)} channels not in template: {sorted(missing)}")

    # Check count: expect 19 channels
    if len(matched) != 19:
        print(f"⚠️ EDF {os.path.basename(edf_path)} yielded {len(matched)} channels (expected 19)")

    # Renumber sequentially
    matched.reset_index(drop=True, inplace=True)
    matched.insert(0, 'idx', range(1, len(matched) + 1))

    return matched[['idx', 'theta', 'radius', 'name']]


def write_locs(df, out_path):
    """
    Write the DataFrame to a .locs file: tab-delimited, no header or index.
    """
    df.to_csv(out_path, sep='\t', header=False, index=False)


def process_directory(input_dir, default_locs, output_dir):
    """
    Generate .locs for every .edf in input_dir using default_locs,
    writing outputs to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    template_df = load_default_locs(default_locs)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.edf'):
            continue
        edf_path = os.path.join(input_dir, fname)
        try:
            locs_df = create_locs_for_edf(edf_path, template_df)
        except Exception as e:
            print(f"⚠️ Skipping {fname}: {e}")
            continue

        out_name = os.path.splitext(fname)[0] + '.locs'
        out_path = os.path.join(output_dir, out_name)
        write_locs(locs_df, out_path)
        print(f"✔ Generated: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate .locs per EDF based on default template"
    )
    parser.add_argument(
        '--input-dir', '-i',
        default='eeg-during-mental-arithmetic-tasks-1.0.0',
        help='Folder with .edf files (default: eeg-during-mental-arithmetic-tasks-1.0.0)'
    )
    parser.add_argument(
        '--default-locs', '-d',
        default='Standard-10-20-Cap19_thesis.locs',
        help='Path to template .locs file (default: Standard-10-20-Cap19_thesis.locs)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='locs',
        help='Folder to save .locs outputs (default: locs)'
    )
    args = parser.parse_args()

    process_directory(args.input_dir, args.default_locs, args.output_dir)
