"""Generate the demand/supply throughput figure for README.

Adapted from the paper figure (scripts/plot_multiline.py) for web display.

Usage:
    uv run --with matplotlib python .github/assets/gen_demand_supply_chart.py

Output:
    .github/assets/demand_supply.png
    .github/assets/demand_supply.svg
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.4,
    }
)

# --- Data (LIBERO + CogACT, H100 — from docs/tuning-guide.md) ---
N = [1, 8, 16, 24, 32, 50, 64, 80, 100]
lam = [11.2, 82.9, 152.1, 214.2, 267.1, 364.6, 410.2, 446.8, 389.4]

# Supply ceilings: batch size B -> max throughput (obs/s)
supply_lines = [(1, 165.2), (2, 255.4), (4, 347.3), (8, 423.8), (16, 468.2)]

# Operating point
N_star, lam_star = 50, 364.6

# --- Colors (AI2 palette, matching paper) ---
C_DEMAND = "#0a3235"
C_SUPPLY = "#8b4049"  # muted burgundy
C_SUPPLY_TXT = "#6b2d35"
C_OP = "#f0529c"
C_OP_TXT = "#c0407d"

fig, ax = plt.subplots(figsize=(7, 3.5))

# Supply ceiling lines
for b, m in supply_lines:
    ax.axhline(y=m, color=C_SUPPLY, linestyle="--", linewidth=1.0, alpha=0.85, zorder=1)
    ax.text(2, m + 6, f"$B$={b}", fontsize=8, color=C_SUPPLY_TXT, fontstyle="italic")

# Demand curve: normal (up to operating point)
ax.plot(
    N[:7],
    lam[:7],
    "o-",
    color=C_DEMAND,
    markersize=5,
    linewidth=1.6,
    markerfacecolor=C_DEMAND,
    markeredgecolor="white",
    markeredgewidth=0.5,
    zorder=3,
)
# Demand curve: oversaturated region
ax.plot(
    N[6:], lam[6:], "o--", color=C_DEMAND, markersize=4, linewidth=0.9, markerfacecolor=C_DEMAND, alpha=0.35, zorder=3
)

# Operating point
ax.plot(N_star, lam_star, "D", color=C_OP, markersize=9, zorder=5, markeredgecolor="white", markeredgewidth=0.7)
ax.annotate(
    r"$N^*\!=\!50$",
    xy=(N_star, lam_star),
    xytext=(70, 260),
    fontsize=11,
    color=C_OP_TXT,
    ha="center",
    weight="bold",
    arrowprops=dict(arrowstyle="->", color=C_OP_TXT, lw=1.0),
)

ax.set_xlabel("Episode workers $N$", fontsize=11)
ax.set_ylabel("Throughput (obs/s)", fontsize=11)
ax.set_xlim(-2, 105)
ax.set_ylim(0, 540)
ax.tick_params(labelsize=10)

legend_elements = [
    Line2D(
        [0],
        [0],
        color=C_DEMAND,
        marker="o",
        markersize=5,
        linewidth=1.4,
        markerfacecolor=C_DEMAND,
        label=r"Demand $\lambda(N)$",
    ),
    Line2D([0], [0], color=C_SUPPLY, linestyle="--", linewidth=1.0, alpha=0.85, label=r"Supply $\mu(B)$ ceilings"),
    Line2D(
        [0], [0], color=C_OP, marker="D", markersize=6, linewidth=0, markeredgecolor="white", label="Operating point"
    ),
]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right", framealpha=0.95, edgecolor="#ddd")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(".github/assets/demand_supply.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(".github/assets/demand_supply.svg", bbox_inches="tight", facecolor="white")
print("Saved .github/assets/demand_supply.{png,svg}")
