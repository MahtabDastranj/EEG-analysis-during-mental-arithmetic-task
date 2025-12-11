import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

results_file = Path(r"E:\AUT\thesis\files\feature_reduction\Results_MASTER_ALL.csv")
out_dir = results_file.parent


def run_sensitivity_comparison(df, group_name):
    print(f"\n{'=' * 60}")
    print(f"ANALYZING METHOD SENSITIVITY: {group_name}")
    print(f"Goal: Compare performance despite different scales (STFT/CWT/EMD)")
    print(f"{'=' * 60}")

    # 1. Filter Data for this specific Group (e.g., 'Good_Counters')
    group_df = df[df["group"] == group_name].copy()

    # 2. Pivot the Data
    # Rows = The specific Brain Feature (e.g., ch01_alpha)
    # Columns = The 3 Methods (STFT, CWT, EMD)
    # Values = 'effect_size_r'
    # CRITICAL STEP: We use Effect Size (0-1) instead of raw amounts
    # because raw amounts are in different units (Power vs Amplitude vs Ratio).
    pivot_df = group_df.pivot(index="feature", columns="method", values="effect_size_r")

    # 3. Clean Data
    # We must drop any feature that wasn't successfully calculated for ALL 3 methods
    # to ensures a fair paired comparison (Friedman requires this).
    original_len = len(pivot_df)
    pivot_df = pivot_df.dropna()
    print(f"Features included in comparison: {len(pivot_df)}")
    if original_len != len(pivot_df):
        print(f"Note: Dropped {original_len - len(pivot_df)} features due to missing data.")

    # 4. Descriptive Stats (Mean Sensitivity)
    print("\n[Average Sensitivity (Effect Size r)]")
    # This shows which method is 'strongest' on average
    print(pivot_df.mean().sort_values(ascending=False))

    # 5. The Friedman Test (Non-Parametric ANOVA for Repeated Measures)
    # H0: All methods provide the same distribution of sensitivity.
    stat, p = friedmanchisquare(
        pivot_df['STFT'],
        pivot_df['CWT'],
        pivot_df['EMD']
    )

    print(f"\n[Friedman Test Results]")
    print(f"Chi-Square Statistic: {stat:.3f}")
    print(f"P-Value: {p:.5e}")  # Scientific notation (e.g., 1.2e-05)

    # 6. Post-Hoc Analysis (Only run if Friedman is Significant)
    if p < 0.05:
        print("\n>> RESULT: Significant Difference Found! (p < 0.05)")
        print(">> Performing Pairwise Wilcoxon Tests to find the winner...")

        # We compare every pair: STFT vs CWT, STFT vs EMD, CWT vs EMD
        pairs = [('STFT', 'CWT'), ('STFT', 'EMD'), ('CWT', 'EMD')]

        # Bonferroni Correction: We are making 3 comparisons, so we divide alpha by 3.
        # Standard Alpha = 0.05 -> Corrected Alpha = 0.0167
        alpha_corrected = 0.05 / 3
        print(f"   (Bonferroni Corrected Significance Threshold: p < {alpha_corrected:.4f})")
        print("-" * 50)

        for m1, m2 in pairs:
            # Wilcoxon Signed-Rank Test on the Effect Sizes
            w_stat, w_p = wilcoxon(pivot_df[m1], pivot_df[m2])

            mean_diff = pivot_df[m1].mean() - pivot_df[m2].mean()
            winner = m1 if mean_diff > 0 else m2

            is_sig = w_p < alpha_corrected

            sig_label = "SIGNIFICANT" if is_sig else "Not Sig"
            print(f"   {m1} vs {m2}:")
            print(f"     p-value: {w_p:.5f} [{sig_label}]")
            if is_sig:
                print(f"     >> WINNER: {winner} (Diff: {abs(mean_diff):.3f})")
            print("-" * 50)

    else:
        print("\n>> RESULT: No significant difference. All methods are equally sensitive.")

    # 7. Visualization (Violin Plot)
    # This visualizes the distribution of 'r' values side-by-side
    plt.figure(figsize=(10, 6))

    # Violin plot shows density; Swarm plot shows actual data points
    sns.violinplot(data=pivot_df, inner="quartile", palette="muted")
    sns.swarmplot(data=pivot_df, color="k", size=1.5, alpha=0.5)

    plt.title(f"Method Sensitivity Comparison\nGroup: {group_name}", fontsize=14)
    plt.ylabel("Effect Size (r)\n(Higher is Better)", fontsize=12)
    plt.xlabel("Feature Extraction Method", fontsize=12)
    plt.ylim(0, 1.0)  # r is always between 0 and 1

    # Annotate with Friedman P-value
    plt.annotate(f"Friedman p = {p:.1e}",
                 xy=(0.5, 0.95), xycoords='axes fraction',
                 ha='center', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # Save Graph
    save_path = out_dir / f"Method_Comparison_Graph_{group_name}.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nGraph saved to: {save_path}")
    plt.show()


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    if not results_file.exists():
        print(f"CRITICAL ERROR: File not found at {results_file}")
        print("Please ensure you run the feature analysis code first to generate Results_MASTER_ALL.csv")
        return

    print(f"Loading Master Results from: {results_file}")
    df = pd.read_csv(results_file)

    # Check what groups exist in the file
    available_groups = df["group"].unique()
    print(f"Found Groups: {available_groups}")

    # Run for Good Counters
    if "Good_Counters" in available_groups:
        run_sensitivity_comparison(df, "Good_Counters")
    elif any("Good" in g for g in available_groups):
        # Fallback search for partial match
        g_name = next(g for g in available_groups if "Good" in g)
        run_sensitivity_comparison(df, g_name)

    # Run for Bad Counters
    if "Bad_Counters" in available_groups:
        run_sensitivity_comparison(df, "Bad_Counters")
    elif any("Bad" in g for g in available_groups):
        # Fallback search for partial match
        g_name = next(g for g in available_groups if "Bad" in g)
        run_sensitivity_comparison(df, g_name)


if __name__ == "__main__":
    main()