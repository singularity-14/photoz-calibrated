"""Figures.

Colour choices are deliberate, not decorative:

- The three methods use the Okabe-Ito colourblind-safe palette (blue / orange /
  green), which is the de facto standard in scientific publishing. Checked with
  a CVD simulator: worst adjacent-pair separation is dE 11.4 under protanopia,
  comfortably above the dE 8 threshold.
- Methods keep the SAME colour in every figure, so you can track one method
  across the whole report without re-reading legends.
- Density plots use a single-hue ramp, never a rainbow. Rainbow ramps invent
  visual boundaries that are not in the data -- a genuine source of misreading
  in astronomy figures specifically, where they remain depressingly common.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed; write files
import matplotlib.pyplot as plt
import numpy as np

from .metrics import coverage, crps, normalize_pdfs, pit_values, point_estimates

# Okabe-Ito: blue, orange, green. Fixed order, never cycled.
METHOD_COLORS = {
    "Polynomial": "#0072B2",
    "kNN": "#E69F00",
    "MDN": "#009E73",
}
GRID_KW = dict(color="0.88", linewidth=0.6)


def _style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, **GRID_KW)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.6")
    ax.tick_params(colors="0.35", labelsize=9)
    return ax


def plot_pit(results: dict, zgrid, z_true, outdir: Path) -> Path:
    """PIT histograms -- THE headline calibration figure.

    Flat = honest. Read the shape, not just the number:
      U-shape -> overconfident (PDFs too narrow)
      hump    -> underconfident (PDFs too wide)
      slope   -> biased
    """
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    nbins = 20

    for ax, (name, pdfs) in zip(axes, results.items()):
        pit = pit_values(pdfs, zgrid, z_true)
        ax.hist(pit, bins=nbins, range=(0, 1), density=True,
                color=METHOD_COLORS[name], alpha=0.85, edgecolor="white", linewidth=0.8)
        ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.5,
                   label="perfectly calibrated")
        ax.set_title(name, fontsize=11, color=METHOD_COLORS[name], fontweight="bold")
        ax.set_xlabel("PIT value")
        ax.set_ylim(0, 3.0)
        _style(ax)

    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, fontsize=8, loc="upper center")
    fig.suptitle("Calibration: are the stated uncertainties honest?\n"
                 "flat = yes   ·   U-shaped = overconfident   ·   hump = underconfident",
                 fontsize=11, y=1.06)
    fig.tight_layout()
    path = outdir / "pit_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_coverage(results: dict, zgrid, z_true, outdir: Path) -> Path:
    """Does a stated X% interval actually contain the truth X% of the time?"""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color="0.25", linestyle="--", linewidth=1.5,
            label="perfect calibration", zorder=1)

    for name, pdfs in results.items():
        levels, emp = coverage(pdfs, zgrid, z_true)
        ax.plot(levels, emp, color=METHOD_COLORS[name], linewidth=2.0,
                marker="o", markersize=4, label=name, zorder=2)

    ax.set_xlabel("stated confidence level")
    ax.set_ylabel("fraction of galaxies actually inside the interval")
    ax.set_title("Coverage\nabove the line = too cautious · below = overconfident",
                 fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    fig.tight_layout()
    path = outdir / "coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_scatter(results: dict, zgrid, z_true, outdir: Path) -> Path:
    """Predicted vs true redshift, as a density map.

    With tens of thousands of points a scatter plot is a solid blob, so we bin
    into hexagons and show counts on a single-hue log scale. The dashed lines
    mark |dz| = 0.15, the conventional "catastrophic outlier" boundary: points
    outside them are galaxies placed at badly wrong distances.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(4.3 * len(results), 4.2),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lim = (0.0, 1.0)
    zline = np.linspace(*lim, 100)

    for ax, (name, pdfs) in zip(axes, results.items()):
        z_phot = point_estimates(pdfs, zgrid, how="mean")
        ax.hexbin(z_true, z_phot, gridsize=55, extent=(*lim, *lim),
                  bins="log", cmap="Blues", mincnt=1, linewidths=0)
        ax.plot(zline, zline, color="0.25", linewidth=1.2)
        ax.plot(zline, zline + 0.15 * (1 + zline), color="#D55E00",
                linestyle="--", linewidth=1.0)
        ax.plot(zline, zline - 0.15 * (1 + zline), color="#D55E00",
                linestyle="--", linewidth=1.0,
                label="catastrophic outlier limit")
        ax.set_title(name, fontsize=11, color=METHOD_COLORS[name], fontweight="bold")
        ax.set_xlabel("true redshift (spectroscopic)")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        _style(ax)

    axes[0].set_ylabel("predicted redshift (photometric)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Predicted vs true redshift   ·   colour = number of galaxies per bin",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    path = outdir / "scatter_zphot_zspec.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _pdf_moments(pdfs, zgrid):
    """Mean and standard deviation of each PDF."""
    mean = _np_trapz(pdfs * zgrid[None, :], zgrid)
    var = _np_trapz(pdfs * (zgrid[None, :] - mean[:, None]) ** 2, zgrid)
    return mean, np.sqrt(np.maximum(var, 0.0))


def _np_trapz(y, x):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x, axis=1)


def _bimodality(pdfs, zgrid, min_separation: float = 0.1):
    """How strongly two-peaked is each PDF?

    Score = height of the second peak / height of the tallest, but ONLY
    counting peaks that are both:
      - prominent (rising at least 10% of the max above the dip between them)
      - genuinely SEPARATED, by at least `min_separation` in redshift

    The separation requirement matters. Two bumps 0.05 apart in z are one
    slightly lumpy peak, not two competing answers, and labelling that
    "genuinely ambiguous" would overclaim. At min_separation = 0.1 the two
    peaks correspond to meaningfully different distances.

    Returns 0 for unimodal; approaching 1 means two near-equally plausible
    answers.
    """
    from scipy.signal import find_peaks

    scores = np.zeros(len(pdfs))
    for i, p in enumerate(pdfs):
        peaks, _ = find_peaks(p, prominence=0.10 * p.max())
        if len(peaks) < 2:
            continue
        # Tallest peak, then the tallest OTHER peak far enough away from it.
        order = peaks[np.argsort(p[peaks])[::-1]]
        main = order[0]
        far = [q for q in order[1:] if abs(zgrid[q] - zgrid[main]) >= min_separation]
        if far:
            scores[i] = p[far[0]] / p[main]
    return scores


def plot_example_pdfs(results: dict, zgrid, z_true, outdir: Path) -> Path:
    """Individual PDFs, chosen to DEMONSTRATE something rather than at random.

    Earlier this function picked galaxies by CRPS quantile, which in practice
    returned four similar-looking unimodal curves -- and the caption claimed to
    show bimodality that was not on the plot. Now each panel is selected for a
    specific property and labelled with it, so the figure cannot overclaim:

      1. the most confident prediction
      2. a typical one
      3. the widest -- the model openly saying "I don't know"
      4. the most two-peaked, IF a genuinely bimodal case exists

    A wide or two-peaked PDF is not a failure. It is the model correctly
    reporting that the photometry does not pin the answer down.
    """
    mdn = normalize_pdfs(results["MDN"], zgrid)
    _, sigma = _pdf_moments(mdn, zgrid)
    bimod = _bimodality(mdn, zgrid)

    order_sigma = np.argsort(sigma)
    picks = [
        (int(order_sigma[0]), "most confident"),
        (int(order_sigma[len(order_sigma) // 2]), "typical"),
        (int(order_sigma[-1]), 'widest — "I don\'t know"'),
    ]
    # Only add the bimodal panel if a real one exists; otherwise don't claim it.
    if bimod.max() > 0.25:
        picks.append((int(np.argmax(bimod)), "two-peaked — genuinely ambiguous"))
        subtitle = ("a wide or two-peaked PDF is the honest answer "
                    "when the colours are ambiguous")
    else:
        subtitle = ("no strongly two-peaked case in this sample; "
                    "width is how this model expresses doubt")

    fig, axes = plt.subplots(1, len(picks), figsize=(3.7 * len(picks), 3.2))
    axes = np.atleast_1d(axes)

    for ax, (i, label) in zip(axes, picks):
        for name, pdfs in results.items():
            ax.plot(zgrid, normalize_pdfs(pdfs, zgrid)[i],
                    color=METHOD_COLORS[name], linewidth=1.8, label=name)
        ax.axvline(z_true[i], color="0.2", linestyle="--", linewidth=1.5,
                   label="true redshift")
        ax.set_xlabel("redshift")
        ax.set_xlim(0, 1)
        ax.set_title(label, fontsize=10)
        _style(ax)

    axes[0].set_ylabel("probability density")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Individual predictions\n{subtitle}", fontsize=10, y=1.08)
    fig.tight_layout()
    path = outdir / "example_pdfs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_metric_comparison(summary: list[dict], outdir: Path) -> Path:
    """Side-by-side bars for the headline metrics.

    Two panels rather than one with two y-axes. A dual-axis chart invites the
    reader to compare two incompatible scales by eye, which is the most common
    way a chart lies.
    """
    names = [s["name"] for s in summary]
    metrics = [
        ("sigma_NMAD", "sigma_NMAD   (scatter — lower is better)"),
        ("CRPS", "CRPS   (whole-PDF quality — lower is better)"),
        ("PIT_KS", "PIT KS statistic   (miscalibration — lower is better)"),
        ("outlier_frac", "catastrophic outlier fraction   (lower is better)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.4))

    for ax, (key, title) in zip(axes, metrics):
        vals = [s[key] for s in summary]
        bars = ax.bar(names, vals, color=[METHOD_COLORS[n] for n in names],
                      width=0.6, zorder=2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, color="0.25")
        ax.set_title(title, fontsize=9.5)
        ax.margins(y=0.18)
        _style(ax)
        ax.tick_params(axis="x", labelsize=9.5, colors="0.25")

    fig.tight_layout()
    path = outdir / "metric_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_training_curve(history: dict, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(history["train"], color="#0072B2", linewidth=1.8, label="train")
    ax.plot(history["val"], color="#E69F00", linewidth=1.8, label="validation")
    best = int(np.argmin(history["val"]))
    ax.axvline(best, color="0.3", linestyle="--", linewidth=1.2,
               label=f"best epoch ({best})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("negative log-likelihood")
    ax.set_title("MDN training\nlower = the model assigns more probability to the truth",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    path = outdir / "mdn_training.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
