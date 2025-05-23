"""
Generate .locs files for each EDF recording by reordering channel locations
based on your default template 'Standard-10-20-Cap19_thesis.locs'.

Defaults assume:
  - EDFs in: eeg-during-mental-arithmetic-tasks-1.0.0/
  - Template locs: Standard-10-20-Cap19_thesis.locs
  - Output folder: locs/

Usage (no flags needed if you keep these names):
  python generate_locs.py

Or override paths:
  python generate_locs.py \
      --input-dir path/to/my_edfs \
      --default-locs path/to/my_template.locs \
      --output-dir path/to/my_output_folder
"""
import os
import argparse
import pandas as pd
import mne


def load_default_locs(default_path):
    """
    Load the default template .locs into a DataFrame.
    Expects whitespace-delimited columns: orig_idx, theta, radius, name
    """
    return pd.read_csv(
        default_path,
        sep=r"\s+",
        header=None,
        names=["orig_idx", "theta", "radius", "name"]
    )


def create_locs_for_edf(edf_path, default_df):
    """
    Read channel names from the EDF and filter/reorder the template accordingly.
    Returns a DataFrame with columns: idx, theta, radius, name
    """
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    edf_channels = raw.ch_names

    # Filter template to channels present in EDF and keep order
    df = default_df[default_df['name'].isin(edf_channels)].copy()
    # Assign new indices 1..N
    df.insert(0, 'idx', range(1, len(df) + 1))
    return df[['idx', 'theta', 'radius', 'name']]


def write_locs(df, out_path):
    """
    Write the DataFrame to a .locs file: tab-delimited, no header or index.
    """
    df.to_csv(out_path, sep='\t', header=False, index=False)


def process_directory(input_dir, default_locs, output_dir):
    """
    Generate .locs for every .edf in input_dir using the default template,
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
