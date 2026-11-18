"""Shared matplotlib style for the EDA notebooks.

Color carries one meaning everywhere: blue = thread WON a delta, orange = did not,
grays = neutral counts / reference marks. (Palette validated for color-vision deficiency.)
"""

import matplotlib.pyplot as plt

WIN, LOSE = "#2a78d6", "#eb6834"
NEUTRAL = "#898781"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.labelsize": 11, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "legend.frameon": False, "legend.labelcolor": INK2,
        "font.family": "sans-serif", "font.size": 11, "figure.dpi": 110,
    })
