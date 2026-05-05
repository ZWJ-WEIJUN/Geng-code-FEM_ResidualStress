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
TW4_QPF_CTRL_SHEET_NAME = "WP9_#3_SP_CMM_CMOS_Filtered"
TW5_TLP_CTRL_SHEET_NAME = "WP10_#3_SP_CMM_CMOS_Filtered"
TW6_TFR_CTRL_SHEET_NAME = "WP10_#2_SP_CMM_CMOS_Filtered"
OUTPUT_PATH = Path(
    "/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/"
    "CompCtrl_DataExtraction_Width/Created_Figures/Height_Profile_TW456.pdf"
)
OUTPUT_ZOOMOUT = OUTPUT_PATH.with_stem(OUTPUT_PATH.stem + "_ZoomOut")
OUTPUT_ZOOMIN  = OUTPUT_PATH.with_stem(OUTPUT_PATH.stem + "_ZoomIn")


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


def _apply_ax_style(ax: plt.Axes) -> None:
    """Apply consistent bold tick and grid style to an axes."""
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", which="major", length=7, width=1.5)
    ax.tick_params(axis="both", which="minor", length=3, width=1.0)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")


def plot_height_profile(
    tw4_qpf_ctrl_df: pd.DataFrame,
    tw5_tlp_ctrl_df: pd.DataFrame,
    tw6_tfr_ctrl_df: pd.DataFrame,
) -> None:
    """Save two separate PDF figures: full height overview and zoomed target region."""
    scatter_kwargs = dict(s=18, marker="o", linewidths=0.6)
    zoom_kwargs    = dict(s=28, marker="o", linewidths=0.7)

    datasets = [
        (tw4_qpf_ctrl_df, "tab:blue",   "TW4 - QPF Height Ctrl"),
        (tw5_tlp_ctrl_df, "tab:red",    "TW5 - Compr. T-LP Ctrl"),
        (tw6_tfr_ctrl_df, "tab:purple", "TW6 - Compr. T-FR Ctrl"),
    ]

    # ── Figure 1: ZoomOut — full height profile (0–55 mm) ────────────────────
    fig1, ax1 = plt.subplots(figsize=(11, 8.5), dpi=300)
    for df, color, label in datasets:
        ax1.scatter(df["x"], df["y"], color=color, edgecolors=color,
                    label=label, **scatter_kwargs)
    ax1.axhline(y=52.40, color="blue", linestyle="--", linewidth=1.5, alpha=0.8,
                label="Target Upper Limit (52.40 mm)")
    ax1.axhline(y=50.40, color="blue", linestyle="--", linewidth=1.5, alpha=0.8,
                label="Target Lower Limit (50.40 mm)")
    ax1.set_xlabel("Length-Y (mm)", fontweight="bold", fontsize=24)
    ax1.set_ylabel("Height-Z (mm)", fontweight="bold", fontsize=24)
    ax1.set_ylim(0, 55)
    ax1.legend(loc="best", fontsize=16, frameon=True)
    _apply_ax_style(ax1)
    fig1.tight_layout()
    fig1.savefig(OUTPUT_ZOOMOUT, bbox_inches="tight")
    print(f"Saved: {OUTPUT_ZOOMOUT}")

    # ── Figure 2: ZoomIn — target region 48.5–54.5 mm ────────────────────────
    fig2, ax2 = plt.subplots(figsize=(11 * 2, 8.5 / 2), dpi=300)
    for df, color, label in datasets:
        ax2.scatter(df["x"], df["y"], color=color, edgecolors=color,
                    label=label, **zoom_kwargs)
    ax2.axhline(y=52.40, color="blue", linestyle="--", linewidth=2.0, alpha=0.8,
                label="Target Upper Limit (52.40 mm)")
    ax2.axhline(y=50.40, color="blue", linestyle="--", linewidth=2.0, alpha=0.8,
                label="Target Lower Limit (50.40 mm)")
    ax2.set_xlabel("Length-Y (mm)", fontweight="bold", fontsize=24)
    ax2.set_ylabel("Height-Z (mm)", fontweight="bold", fontsize=24)
    ax2.set_ylim(48.5, 54.5)
    ax2.legend(loc="best", fontsize=16, frameon=True)
    _apply_ax_style(ax2)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_ZOOMIN, bbox_inches="tight")
    print(f"Saved: {OUTPUT_ZOOMIN}")

    if plt.get_backend().lower() != "agg":
        plt.show()


def main() -> None:
    tw4_qpf_ctrl_df = load_height_profile(WORKBOOK_PATH, TW4_QPF_CTRL_SHEET_NAME)
    tw5_tlp_ctrl_df = load_height_profile(WORKBOOK_PATH, TW5_TLP_CTRL_SHEET_NAME)
    tw6_tfr_ctrl_df = load_height_profile(WORKBOOK_PATH, TW6_TFR_CTRL_SHEET_NAME)

    print(f"Loaded {len(tw4_qpf_ctrl_df)} valid rows from {TW4_QPF_CTRL_SHEET_NAME}")
    print(f"TW4 x range: {tw4_qpf_ctrl_df['x'].min():.4f} to {tw4_qpf_ctrl_df['x'].max():.4f}")
    print(f"TW4 y range: {tw4_qpf_ctrl_df['y'].min():.4f} to {tw4_qpf_ctrl_df['y'].max():.4f}")
    print(f"Loaded {len(tw5_tlp_ctrl_df)} valid rows from {TW5_TLP_CTRL_SHEET_NAME}")
    print(f"TW5 x range: {tw5_tlp_ctrl_df['x'].min():.4f} to {tw5_tlp_ctrl_df['x'].max():.4f}")
    print(f"TW5 y range: {tw5_tlp_ctrl_df['y'].min():.4f} to {tw5_tlp_ctrl_df['y'].max():.4f}")
    print(f"Loaded {len(tw6_tfr_ctrl_df)} valid rows from {TW6_TFR_CTRL_SHEET_NAME}")
    print(f"TW6 x range: {tw6_tfr_ctrl_df['x'].min():.4f} to {tw6_tfr_ctrl_df['x'].max():.4f}")
    print(f"TW6 y range: {tw6_tfr_ctrl_df['y'].min():.4f} to {tw6_tfr_ctrl_df['y'].max():.4f}")
    print(f"Saving figure to: {OUTPUT_PATH}")
    plot_height_profile(tw4_qpf_ctrl_df, tw5_tlp_ctrl_df, tw6_tfr_ctrl_df)


if __name__ == "__main__":
    main()
