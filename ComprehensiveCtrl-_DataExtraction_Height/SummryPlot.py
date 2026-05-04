"""Create a summary performance plot for thin wall control strategies.

The figure combines height, width, and cycle time information into a single
quadrant-style chart for all 9 thin wall cases.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["ytick.labelsize"] = 22


OUTPUT_PATH = Path(__file__).with_name("ThinWall_Summary_Plot.pdf")

TARGET_WIDTH = 3.00
TARGET_HEIGHT = 50.40
IDEAL_WIDTH_MAX = 3.50
IDEAL_HEIGHT_MAX = 52.40

X_MIN, X_MAX = 2.3, 4.5
Y_MIN, Y_MAX = 31.0, 64.0

# (label, height_mean, height_ci, width_mean, width_ci, time, time_reduction,
#  color, marker, alpha, text_xy, ha)
SUMMARY_DATA = [
    {
        "label": "TW2\nNo Ctrl",
        "height_mean": 32.9185,
        "height_ci": (32.6779, 33.1592),
        "width_mean": 3.0488,
        "width_ci": (2.8334, 3.2641),
        "time": 9.62,
        "time_reduction": "-25%",
        "color": "black",
        "marker": "o",
        "alpha": 1.0,
        "text_xy": (2.35, 38.5),
        "ha": "left",
    },
    {
        "label": "TW1\nNo Ctrl (0.6mm)",
        "height_mean": 52.8139,
        "height_ci": (52.7716, 52.8563),
        "width_mean": 3.2445,
        "width_ci": (3.1541, 3.3348),
        "time": 12.80,
        "time_reduction": "---",
        "color": "black",
        "marker": "^",
        "alpha": 1.0,
        "text_xy": (3.52, 60.0),
        "ha": "left",
    },
    {
        "label": "TW3\nPFR Ctrl",
        "height_mean": 43.2877,
        "height_ci": (43.0150, 43.5604),
        "width_mean": 3.2063,
        "width_ci": (3.0727, 3.3399),
        "time": 9.62,
        "time_reduction": "-25%",
        "color": "grey",
        "marker": "o",
        "alpha": 1.0,
        "text_xy": (3.95, 40.0),
        "ha": "left",
    },
    {
        "label": "TW4\nQPF Height Ctrl",
        "height_mean": 50.4224,
        "height_ci": (50.3452, 50.4997),
        "width_mean": 3.2719,
        "width_ci": (3.1527, 3.3911),
        "time": 9.62,
        "time_reduction": "-25%",
        "color": "tab:blue",
        "marker": "o",
        "alpha": 1.0,
        "text_xy": (3.1, 46.0),
        "ha": "left",
    },
    {
        "label": "TW5\nCompr. T-LP Ctrl",
        "height_mean": 50.7576,
        "height_ci": (50.6817, 50.8336),
        "width_mean": 3.1025,
        "width_ci": (3.0167, 3.1884),
        "time": 9.62,
        "time_reduction": "-25%",
        "color": "tab:red",
        "marker": "o",
        "alpha": 1.0,
        "text_xy": (3.1, 60.0),
        "ha": "left",
    },
    {
        "label": "TW6\nCompr. T-FR Ctrl",
        "height_mean": 51.2078,
        "height_ci": (51.1408, 51.2748),
        "width_mean": 2.9865,
        "width_ci": (2.8975, 3.0755),
        "time": 8.62,
        "time_reduction": "-33%",
        "color": "tab:purple",
        "marker": "o",
        "alpha": 1.0,
        "text_xy": (2.35, 53.5),
        "ha": "left",
    },
    {
        "label": "TW7\nCompr. W-LP Ctrl",
        "height_mean": 50.5863,
        "height_ci": (50.5190, 50.6536),
        "width_mean": 3.3943,
        "width_ci": (3.2997, 3.4890),
        "time": 9.62,
        "time_reduction": "-25%",
        "color": "tab:red",
        "marker": "o",
        "alpha": 0.5,
        "text_xy": (3.95, 55.5),
        "ha": "left",
    },
    {
        "label": "TW8\nCompr. W-LP & FR",
        "height_mean": 50.9946,
        "height_ci": (50.9674, 51.0218),
        "width_mean": 3.3967,
        "width_ci": (3.3142, 3.4792),
        "time": 8.39,
        "time_reduction": "-34%",
        "color": "tab:purple",
        "marker": "o",
        "alpha": 0.45,
        "text_xy": (3.95, 59.5),
        "ha": "left",
    },
    {
        "label": "TW9\nCompr. W-LP & FR\n(1.0mm)",
        "height_mean": 51.2833,
        "height_ci": (51.2350, 51.3316),
        "width_mean": 3.7099,
        "width_ci": (3.6026, 3.8172),
        "time": 7.31,
        "time_reduction": "-43%",
        "color": "tab:green",
        "marker": "s",
        "alpha": 1.0,
        "text_xy": (3.95, 51.0),
        "ha": "left",
    },
]


def plot_summary() -> None:
    fig, ax = plt.subplots(figsize=(13, 10), dpi=300)

    ax.set_facecolor("white")

    # Fraction of target height within y limits — used for axvspan ymin/ymax
    yf = (TARGET_HEIGHT - Y_MIN) / (Y_MAX - Y_MIN)

    # Quadrant backgrounds
    ax.axvspan(X_MIN,        TARGET_WIDTH, ymin=yf,  ymax=1.0, color="#f8d7da", alpha=0.5, zorder=0)  # upper-left:  unacceptable (underbuilt width)
    ax.axvspan(TARGET_WIDTH, X_MAX,        ymin=yf,  ymax=1.0, color="#fff3cd", alpha=0.8, zorder=0)  # upper-right: acceptable but not ideal (light yellow)
    ax.axvspan(X_MIN,        TARGET_WIDTH, ymin=0.0, ymax=yf,  color="#f8d7da", alpha=0.8, zorder=0)  # lower-left:  unacceptable (height & width)
    ax.axvspan(TARGET_WIDTH, X_MAX,        ymin=0.0, ymax=yf,  color="#f8d7da", alpha=0.5, zorder=0)  # lower-right: unacceptable (height)

    # Ideal region rectangle: width 3.0–3.5 mm, height 50.4–52.4 mm
    ideal_rect = Rectangle(
        (TARGET_WIDTH, TARGET_HEIGHT),
        IDEAL_WIDTH_MAX - TARGET_WIDTH,
        IDEAL_HEIGHT_MAX - TARGET_HEIGHT,
        facecolor="#c3e6cb", alpha=0.95, edgecolor="#28a745",
        linewidth=1.8, linestyle="--", zorder=1,
    )
    ax.add_patch(ideal_rect)

    # Target crosshair lines
    ax.axvline(TARGET_WIDTH,    color="#aaaaaa", linewidth=1.3, zorder=2)
    ax.axhline(TARGET_HEIGHT,   color="#aaaaaa", linewidth=1.3, zorder=2)
    ax.axvline(IDEAL_WIDTH_MAX, color="#aaaaaa", linewidth=1.0, linestyle=":", zorder=2)
    ax.axhline(IDEAL_HEIGHT_MAX, color="#aaaaaa", linewidth=1.0, linestyle=":", zorder=2)

    # Data points with error bars and annotations
    for item in SUMMARY_DATA:
        x = item["width_mean"]
        y = item["height_mean"]
        xerr = [[x - item["width_ci"][0]], [item["width_ci"][1] - x]]
        yerr = [[y - item["height_ci"][0]], [item["height_ci"][1] - y]]

        ax.errorbar(
            x, y,
            xerr=xerr,
            yerr=yerr,
            fmt=item["marker"],
            markersize=9,
            color=item["color"],
            ecolor=item["color"],
            elinewidth=1.8,
            capsize=5,
            capthick=1.5,
            alpha=item["alpha"],
            zorder=3,
        )

        if item["time_reduction"] == "---":
            ann_text = f"{item['label']}\nTime: {item['time']:.2f} min"
        else:
            ann_text = f"{item['label']}\n{item['time']:.2f} min ({item['time_reduction']})"

        ax.annotate(
            ann_text,
            xy=(x, y),
            xytext=item["text_xy"],
            fontsize=16,
            fontweight="bold",
            color=item["color"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=item["color"], lw=1.5, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=item["color"], lw=1.6, shrinkA=5, shrinkB=5),
            ha=item["ha"],
            va="center",
        )

    # Target star — label placed directly below with a short arrow
    ax.scatter(
        TARGET_WIDTH, TARGET_HEIGHT,
        marker="*", s=350,
        color="#ff9f1a", edgecolors="#ff9f1a",
        zorder=4,
    )
    ax.annotate(
        "Target",
        xy=(TARGET_WIDTH, TARGET_HEIGHT),
        xytext=(TARGET_WIDTH, TARGET_HEIGHT - 1.8),
        fontsize=18,
        fontweight="bold",
        color="#ff9f1a",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ff9f1a", lw=1.5, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#ff9f1a", lw=1.4, shrinkA=3, shrinkB=4),
        ha="center",
        va="top",
    )

    # Quadrant corner labels (large, bold — matching reference style)
    # Ideal region label (inside the rectangle)
    ax.text(
        (TARGET_WIDTH + IDEAL_WIDTH_MAX) / 2, (TARGET_HEIGHT + IDEAL_HEIGHT_MAX) / 2,
        "Ideal",
        fontsize=18, fontweight="bold", ha="center", va="center",
        color="#155724", alpha=0.7,
    )

    # Quadrant corner labels
    ax.text(2.33, 63.8, "Unacceptable\n(Underbuilt in Width)",
            fontsize=18, fontweight="bold", ha="left", va="top", color="black")
    ax.text(4.47, 63.8, "Acceptable\n(Not Ideal)",
            fontsize=18, fontweight="bold", ha="right", va="top", color="black")
    ax.text(2.33, 31.2, "Unacceptable\n(Underbuilt in Height\nand Width)",
            fontsize=18, fontweight="bold", ha="left", va="bottom", color="black")
    ax.text(4.47, 31.2, "Unacceptable\n(Underbuilt in Height)",
            fontsize=18, fontweight="bold", ha="right", va="bottom", color="black")

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Width (mm)", fontweight="bold", fontsize=24)
    ax.set_ylabel("Height (mm)", fontweight="bold", fontsize=24)

    ax.tick_params(axis="both", length=7, width=1.5)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    plot_summary()
