import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

path = Path(r"E:\AUT\thesis\files\feature_reduction\t-test\Results.csv")
df = pd.read_csv(path)

# 2. Pre-process: Split the 'feature' column into 'channel' and 'band'
# Example: 'ch01_theta' becomes channel='ch01', band='theta'
df[['channel', 'band']] = df['feature'].str.split('_', expand=True)

# 3. Apply Selection Criteria: p < 0.05 AND effect size (r) > 0.5
# We create a 'display_val' column. If it doesn't meet criteria, we set it to 0 (white in heatmap).
df['display_val'] = df.apply(
    lambda row: row['effect_size_r'] if (row['p_val'] < 0.05 and row['effect_size_r'] > 0.5) else 0,
    axis=1
)

# 4. Define specific orders for the axes to keep the plot organized
channels = sorted(df['channel'].unique(), key=lambda x: int(x[2:]))
bands = ["delta", "theta", "alpha", "beta", "gamma"]
methods = ["STFT", "CWT", "EMD"]
groups = ["Bad_Counters", "Good_Counters"]

# 5. Initialize the visualization grid (2 rows for Groups x 3 columns for Methods)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), sharex=True, sharey=True)

for i, group in enumerate(groups):
    for j, method in enumerate(methods):
        ax = axes[i, j]

        # Filter data for this specific subplot
        subset = df[(df['group'] == group) & (df['method'] == method)]

        # Create a pivot table for the heatmap: Rows=Channels, Columns=Bands
        pivot = subset.pivot(index='channel', columns='band', values='display_val')

        # Reorder columns and rows to ensure consistency across all subplots
        pivot = pivot.reindex(index=channels, columns=bands).fillna(0)

        # Plot the heatmap
        # vmin=0.5 ensures the color scale starts at our selection threshold
        sns.heatmap(
            pivot,
            annot=False,
            cmap="YlOrRd",
            vmin=0.5,
            vmax=1.0,
            ax=ax,
            cbar_kws={'label': 'Effect Size (r)'} if (j == 2) else None
        )

        ax.set_title(f"{method} | {group}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Frequency Band")
        ax.set_ylabel("EEG Channel")

plt.tight_layout()
plt.savefig('significant_features_heatmap.png', dpi=300)
print("Heatmap generation complete. File saved as 'significant_features_heatmap.png'.")