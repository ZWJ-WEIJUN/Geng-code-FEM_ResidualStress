"""Analyze height profiles for all 9 thin walls.

Calculates mean, standard deviation, and 95% confidence intervals
for height values in the x range [15, 85] mm.
"""

from pathlib import Path

import numpy as np
import pandas as pd


WORKBOOK_PATH = Path(
    "/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/"
    "ComprehensiveCtrl-_DataExtraction_Height/WP_All_Height_03102026.xlsx"
)

SHEET_NAMES = {
    "TW2 - No Ctrl": "WP10_#1_SP_CMM_CMOS_Filtered",
    "TW1 - No Ctrl (0.6 mm)": "WP7_#2_SP_CMM_CMOS_Filtered",
    "TW3 - PFR Ctrl": "WP9_#2_SP_CMM_CMOS_Filtered",
    "TW4 - QPF Height Ctrl": "WP9_#3_SP_CMM_CMOS_Filtered",
    "TW5 - Compr. T-LP Ctrl": "WP10_#3_SP_CMM_CMOS_Filtered",
    "TW6 - Compr. T-FR Ctrl": "WP10_#2_SP_CMM_CMOS_Filtered",
    "TW7 - Compr. W-LP Ctrl": "WP6_#1_SP_CMM_CMOS_Filtered",
    "TW8 - Compr. W-LP & FR Hybrid Ctrl": "WP6_#2_SP_CMM_CMOS_Filtered",
    "TW9 - Compr. W-LP & FR Hybrid Ctrl (Higher Productivity)": "WP8_#1_SP_CMM_CMOS_Filtered",
}

X_MIN = 20 # Set to 20 mm to exclude the very edge region of side thin wall
X_MAX = 80 # Set to 80 mm to exclude the very edge region of side thin wall

def load_height_profile(workbook_path: Path, sheet_name: str) -> pd.DataFrame:
    """Load and clean the height profile data from the Excel sheet."""
    df = pd.read_excel(
        workbook_path,
        sheet_name=sheet_name,
        header=4,
        usecols=["Yshifted", "Z"],
        engine="openpyxl",
    )

    df = df.rename(columns={"Yshifted": "x", "Z": "y"})
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    return df


def calculate_stats(y_values: np.ndarray) -> dict:
    """Calculate mean, std, and 95% CI for the given y values."""
    n = len(y_values)
    mean = np.mean(y_values)
    std = np.std(y_values, ddof=1)  # Sample std dev
    std_err = std / np.sqrt(n)
    ci_margin = 1.96 * std_err  # 95% CI
    ci_lower = mean - ci_margin
    ci_upper = mean + ci_margin
    
    return {
        "mean": mean,
        "std": std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_points": n,
    }


def main() -> None:
    print("=" * 100)
    print(f"Height Profile Analysis (x range: {X_MIN}-{X_MAX} mm)")
    print("=" * 100)
    print()

    results = []
    for tw_label, sheet_name in SHEET_NAMES.items():
        df = load_height_profile(WORKBOOK_PATH, sheet_name)
        
        # Filter to x range [15, 85]
        mask = (df["x"] >= X_MIN) & (df["x"] <= X_MAX)
        filtered_df = df[mask]
        
        if len(filtered_df) == 0:
            print(f"{tw_label:60s} | No data in x range")
            continue
        
        stats = calculate_stats(filtered_df["y"].values)
        results.append((tw_label, stats))
        
        print(f"{tw_label:60s}")
        print(f"  Points (x={X_MIN}-{X_MAX} mm): {stats['n_points']}")
        print(f"  Mean Height (Z):               {stats['mean']:.4f} mm")
        print(f"  Standard Deviation:            {stats['std']:.4f} mm")
        print(f"  95% Confidence Interval:       [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] mm")
        print()

    print("=" * 100)
    print("Summary Table")
    print("=" * 100)
    print(f"{'Thin Wall':<60s} | {'Mean (mm)':>10s} | {'Std Dev (mm)':>12s} | {'95% CI Lower':>12s} | {'95% CI Upper':>12s} | {'N Points':>8s}")
    print("-" * 120)
    for tw_label, stats in results:
        print(f"{tw_label:<60s} | {stats['mean']:>10.4f} | {stats['std']:>12.4f} | {stats['ci_lower']:>12.4f} | {stats['ci_upper']:>12.4f} | {stats['n_points']:>8d}")
    print()


if __name__ == "__main__":
    main()
