"""Generate the speedup comparison bar chart for README.

Adapted from the paper figure (scripts/gen_speedup_fig.py) for web display.

Usage:
    uv run --with matplotlib python .github/assets/gen_speedup_chart.py

Output:
    .github/assets/speedup_comparison.png
    .github/assets/speedup_comparison.svg
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.12,
        "grid.linewidth": 0.4,
    }
)

# --- Data (from paper Table / Section III-C) ---
benchmarks = ["LIBERO", "CALVIN", "SimplerEnv"]
sequential = [14, 8.6, 1.73]  # hours
parallel = [0.3, 0.55, 0.14]  # hours
speedup_labels = ["47\u00d7", "16\u00d7", "12\u00d7"]
par_labels = ["18 min", "33 min", "8.5 min"]
seq_labels = ["14 h", "8.6 h", "1.7 h"]

x = np.arange(len(benchmarks))
width = 0.34

# --- Colors ---
C_SEQ = "#1b5e5e"
C_PAR = "#e24a8d"
C_BADGE = "#da9679"

fig, ax = plt.subplots(figsize=(7, 3))

bars_seq = ax.bar(
    x - width / 2, sequential, width, color=C_SEQ, edgecolor="white", linewidth=0.5, label="Sequential", zorder=3
)
bars_par = ax.bar(
    x + width / 2, parallel, width, color=C_PAR, edgecolor="white", linewidth=0.5, label="Batch Parallel", zorder=3
)

# Sequential labels
for bar, lbl in zip(bars_seq, seq_labels):
    y_pos = bar.get_height()
    if y_pos > 4:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos - 0.4,
            lbl,
            ha="center",
            va="top",
            fontsize=11,
            color="white",
            fontweight="bold",
        )
    else:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos + 0.15,
            lbl,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#134a4a",
            fontweight="bold",
        )

# Parallel labels
for bar, lbl in zip(bars_par, par_labels):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.25,
        lbl,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#b83a70",
        fontweight="bold",
    )

# Speedup badges
badge_y = [sequential[0] + 0.6, sequential[1] + 0.6, 4.5]
for i, s in enumerate(speedup_labels):
    ax.text(
        x[i],
        badge_y[i],
        s,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=C_BADGE,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f5ddd0", edgecolor="#da9679", linewidth=0.8),
    )
# Arrow from SimplerEnv badge to bar
ax.annotate(
    "", xy=(x[2], sequential[2] + 0.3), xytext=(x[2], 4.4), arrowprops=dict(arrowstyle="->", color=C_BADGE, lw=0.9)
)

ax.set_ylabel("Wall-clock time (hours)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=12)
ax.set_ylim(0, 18)
ax.yaxis.set_major_locator(mticker.MultipleLocator(4))
ax.yaxis.grid(True, linestyle="-", alpha=0.12, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=10)
ax.legend(loc="upper right", fontsize=10, frameon=True, fancybox=False, edgecolor="#bbbbbb", framealpha=0.92)

plt.tight_layout()
fig.savefig(".github/assets/speedup_comparison.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(".github/assets/speedup_comparison.svg", bbox_inches="tight", facecolor="white")
print("Saved .github/assets/speedup_comparison.{png,svg}")
