"""Shared figure style for the example scripts.

Ported from the E-PSS paper's `subroutines.utils.figure_style` so the demo
figures match the paper: seaborn "ticks" theme, Fira Sans, the project color
cycle, gridded axes. seaborn / Fira Sans are applied when available and fall
back to plain matplotlib otherwise, so the examples never hard-depend on them.
"""

from __future__ import annotations

# Project color cycle (Fresnel/empirical/seapol/hybrid series, in order).
COLOR_LIST = ['#4C2882', '#367588', '#A52A2A', '#C39953', '#2A52BE', '#006611']


def figure_style(title_fontsize=10, label_fontsize=10, tick_fontsize=10):
    """Apply the paper figure style. Returns (color_list, fullwidth, fullheight,
    fsize): the color cycle and the full letter-page figure dimensions [in]."""
    import matplotlib.pyplot as plt

    fsize = 10
    lw = 1.0

    try:
        import seaborn as sns
        sns.set_theme(style="ticks", palette="deep", font="Fira Sans")
    except Exception:
        pass

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLOR_LIST)
    plt.rcParams.update({
        'axes.grid': True,
        'font.size': fsize,
        'axes.titlesize': title_fontsize,
        'axes.labelsize': label_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'legend.fontsize': label_fontsize,
        'grid.linewidth': lw,
        'xtick.major.width': lw,
        'ytick.major.width': lw,
    })

    # Full page figure size (letter paper with 0.5 inch margins).
    fullwidth = 7.5
    fullheight = 10

    return COLOR_LIST, fullwidth, fullheight, fsize
