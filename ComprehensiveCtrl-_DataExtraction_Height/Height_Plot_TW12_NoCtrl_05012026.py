"""Plot the No Ctrl height profile from the Excel workbook.

The target sheet stores the x-axis values in `Yshifted` and the y-axis values
in `Z`, with the headers located on row 5.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20


WORKBOOK_PATH = Path(
    "/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/"
    "ComprehensiveCtrl-_DataExtraction_Height/WP_All_Height_03102026.xlsx"
)
TW2_NO_CTRL_SHEET_NAME = "WP10_#1_SP_CMM_CMOS_Filtered"
TW1_NO_CTRL_SHEET_NAME = "WP7_#2_SP_CMM_CMOS_Filtered"
# TW3_PFR_CTRL_SHEET_NAME = "WP9_#2_SP_CMM_CMOS_Filtered"
# TW4_QPF_CTRL_SHEET_NAME = "WP9_#3_SP_CMM_CMOS_Filtered"
# TW5_TLP_CTRL_SHEET_NAME = "WP10_#3_SP_CMM_CMOS_Filtered"
OUTPUT_PATH = Path(
    "/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/"
    "CompCtrl_DataExtraction_Width/Created_Figures/Height_Profile_TW12_NoCtrl.pdf"
)


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


def plot_height_profile(tw2_no_ctrl_df: pd.DataFrame, tw1_no_ctrl_df: pd.DataFrame) -> None:
    """Plot the cleaned height profiles with print-friendly formatting."""
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)

    ax.scatter(
        tw2_no_ctrl_df["x"],
        tw2_no_ctrl_df["y"],
        color="black",
        s=18,
        marker="o",
        edgecolors="black",
        linewidths=0.6,
        label="TW2 - No Ctrl (0.8 mm)",
    )

    ax.scatter(
        tw1_no_ctrl_df["x"],
        tw1_no_ctrl_df["y"],
        color="black",
        s=30,
        marker="^",
        edgecolors="black",
        linewidths=0.6,
        label="TW1 - No Ctrl (0.6 mm)",
    )

    # ax.scatter(
    #     tw3_pfr_ctrl_df["x"], tw3_pfr_ctrl_df["y"],
    #     color="grey", s=18, marker="o", edgecolors="grey",
    #     linewidths=0.6, label="TW3 - PFR Ctrl",
    # )
    # ax.scatter(
    #     tw4_qpf_ctrl_df["x"], tw4_qpf_ctrl_df["y"],
    #     color="tab:blue", s=18, marker="o", edgecolors="tab:blue",
    #     linewidths=0.6, label="TW4 - QPF Height Ctrl",
    # )
    # ax.scatter(
    #     tw5_tlp_ctrl_df["x"], tw5_tlp_ctrl_df["y"],
    #     color="tab:red", s=18, marker="o", edgecolors="tab:red",
    #     linewidths=0.6, label="TW5 - Compr. T-LP Ctrl",
    # )

    # Target tolerance lines: 50.40 mm +2.00 mm and -0.00 mm
    target_center = 50.40
    target_upper = target_center + 2.00  # 52.40 mm
    target_lower = target_center - 0.00  # 50.40 mm
    
    ax.axhline(y=target_upper, color="blue", linestyle="--", linewidth=1.5, alpha=0.8, label="Target Upper Limit (52.40 mm)")
    ax.axhline(y=target_lower, color="blue", linestyle="--", linewidth=1.5, alpha=0.8, label="Target Lower Limit (50.40 mm)")

    ax.set_xlabel("Length-Y (mm)", fontweight="bold", fontsize=24)
    ax.set_ylabel("Height-Z (mm)", fontweight="bold", fontsize=24)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", which="major", length=7, width=1.5)
    ax.tick_params(axis="both", which="minor", length=3, width=1.0)
    
    # Make tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    
    ax.legend(loc="best", fontsize=16, frameon=True)
    ax.set_ylim(0, 55)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()


def main() -> None:
    tw2_no_ctrl_df = load_height_profile(WORKBOOK_PATH, TW2_NO_CTRL_SHEET_NAME)
    tw1_no_ctrl_df = load_height_profile(WORKBOOK_PATH, TW1_NO_CTRL_SHEET_NAME)
    # tw3_pfr_ctrl_df = load_height_profile(WORKBOOK_PATH, TW3_PFR_CTRL_SHEET_NAME)
    # tw4_qpf_ctrl_df = load_height_profile(WORKBOOK_PATH, TW4_QPF_CTRL_SHEET_NAME)
    # tw5_tlp_ctrl_df = load_height_profile(WORKBOOK_PATH, TW5_TLP_CTRL_SHEET_NAME)

    print(f"Loaded {len(tw2_no_ctrl_df)} valid rows from {TW2_NO_CTRL_SHEET_NAME}")
    print(f"TW2 x range: {tw2_no_ctrl_df['x'].min():.4f} to {tw2_no_ctrl_df['x'].max():.4f}")
    print(f"TW2 y range: {tw2_no_ctrl_df['y'].min():.4f} to {tw2_no_ctrl_df['y'].max():.4f}")
    print(f"Loaded {len(tw1_no_ctrl_df)} valid rows from {TW1_NO_CTRL_SHEET_NAME}")
    print(f"TW1 x range: {tw1_no_ctrl_df['x'].min():.4f} to {tw1_no_ctrl_df['x'].max():.4f}")
    print(f"TW1 y range: {tw1_no_ctrl_df['y'].min():.4f} to {tw1_no_ctrl_df['y'].max():.4f}")
    # print(f"Loaded {len(tw3_pfr_ctrl_df)} valid rows from {TW3_PFR_CTRL_SHEET_NAME}")
    # print(f"TW3 PFR x range: {tw3_pfr_ctrl_df['x'].min():.4f} to {tw3_pfr_ctrl_df['x'].max():.4f}")
    # print(f"TW3 PFR y range: {tw3_pfr_ctrl_df['y'].min():.4f} to {tw3_pfr_ctrl_df['y'].max():.4f}")
    # print(f"Loaded {len(tw4_qpf_ctrl_df)} valid rows from {TW4_QPF_CTRL_SHEET_NAME}")
    # print(f"TW4 QPF x range: {tw4_qpf_ctrl_df['x'].min():.4f} to {tw4_qpf_ctrl_df['x'].max():.4f}")
    # print(f"TW4 QPF y range: {tw4_qpf_ctrl_df['y'].min():.4f} to {tw4_qpf_ctrl_df['y'].max():.4f}")
    # print(f"Loaded {len(tw5_tlp_ctrl_df)} valid rows from {TW5_TLP_CTRL_SHEET_NAME}")
    # print(f"TW5 x range: {tw5_tlp_ctrl_df['x'].min():.4f} to {tw5_tlp_ctrl_df['x'].max():.4f}")
    # print(f"TW5 y range: {tw5_tlp_ctrl_df['y'].min():.4f} to {tw5_tlp_ctrl_df['y'].max():.4f}")
    print(f"Saving figure to: {OUTPUT_PATH}")
    plot_height_profile(tw2_no_ctrl_df, tw1_no_ctrl_df)


if __name__ == "__main__":
    main()
