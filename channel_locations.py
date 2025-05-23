"""
Generate a .locs file for each EDF recording by reordering
channel locations based on a default template .locs file.

Dependencies:
  - pandas
  - mne (or pyedflib, if you prefer)

Usage example:
  python generate_locs.py \
    --input-dir /path/to/edf_files \
    --default-locs /path/to/Standard-10-20-Cap19_thesis.locs \
    --output-dir /path/to/output_locs
"""
import os
import argparse
import pandas as pd
import mne


def load_default_locs(default_path):
    """
    Load default .locs template into a DataFrame.
    Assumes no header, columns: index, theta, radius, name (separated by whitespace).
    """
    df = pd.read_csv(
        default_path,
        sep=r"\s+",
        header=None,
        names=["orig_idx", "theta", "radius", "name"]
    )
    return df


def create_locs_for_edf(edf_path, default_df):
    """
    Read channel names from EDF and filter/reorder default_df accordingly.
    Returns DataFrame with columns: idx, theta, radius, name.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    edf_channels = raw.ch_names

    # Keep only channels present in EDF, in default order
    df = default_df[default_df['name'].isin(edf_channels)].copy()
    # Renumber from 1..N
    df.insert(0, 'idx', range(1, len(df) + 1))
    return df[['idx', 'theta', 'radius', 'name']]


def write_locs(df, out_path):
    """
    Write DataFrame to .locs file: no header, tab-delimited.
    """
    df.to_csv(out_path, sep='\t', header=False, index=False)


def process_directory(input_dir, default_locs_path, output_dir):
    """
    Walk through input_dir, process all .edf files, and save .locs to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    default_df = load_default_locs(default_locs_path)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.edf'):
            continue
        edf_path = os.path.join(input_dir, fname)
        try:
            locs_df = create_locs_for_edf(edf_path, default_df)
        except Exception as e:
            print(f"⚠️  Could not read {fname}: {e}")
            continue

        base = os.path.splitext(fname)[0]
        out_file = os.path.join(output_dir, f"{base}.locs")
        write_locs(locs_df, out_file)
        print(f"✔ Generated: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate .locs per EDF based on default template"
    )
    parser.add_argument(
        '--input-dir', '-i', required=True,
        help='Directory containing .edf files.'
    )
    parser.add_argument(
        '--default-locs', '-d', required=True,
        help='Path to default template .locs file.'
    )
    parser.add_argument(
        '--output-dir', '-o', required=True,
        help='Directory to save generated .locs files.'
    )
    args = parser.parse_args()

    process_directory(args.input_dir, args.default_locs, args.output_dir)
