"""How we judge a probabilistic prediction.

This module is the heart of the project. Everything else is machinery to
produce PDFs; this is where we decide whether those PDFs are any good.

Every model here outputs, for each galaxy, a probability density p(z) evaluated
on a shared grid of redshifts. That shared grid is what makes the comparison
between methods fair.

THE KEY IDEA
------------
An ordinary ML project asks "how close is the prediction to the truth?".
That question is not sufficient here. If a model says "z = 0.5 +/- 0.01" and
the truth is 0.6, it is not merely a bit wrong -- it is *confidently* wrong,
and in astronomy that is far more damaging than saying "z = 0.5 +/- 0.2".
Downstream cosmology propagates your error bars. A wrong error bar corrupts
the science in a way a wrong central value does not.

So we measure two separate things:
  - SHARPNESS / ACCURACY: is the distribution concentrated near the truth?
  - CALIBRATION: are the stated uncertainties honest?

A model can be excellent at one and terrible at the other.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kstest

# numpy renamed trapz -> trapezoid in 2.0; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def normalize_pdfs(pdfs: np.ndarray, zgrid: np.ndarray) -> np.ndarray:
    """Force every PDF to integrate to 1 over the grid.

    This also handles a subtlety: a Gaussian mixture can put probability mass
    at z < 0, which is physically impossible. By defining the grid to start at
    z = 0 and renormalizing here, we are implicitly truncating to z >= 0 and
    redistributing that mass. That is a real modelling choice, applied
    identically to every method so the comparison stays fair.
    """
    pdfs = np.clip(np.asarray(pdfs, dtype=np.float64), 0.0, None)
    area = _trapz(pdfs, zgrid, axis=1)
    area[area <= 0] = 1.0
    return pdfs / area[:, None]


def pdfs_to_cdfs(pdfs: np.ndarray, zgrid: np.ndarray) -> np.ndarray:
    """Cumulative integral of each PDF, forced to end at exactly 1."""
    cdfs = cumulative_trapezoid(pdfs, zgrid, axis=1, initial=0.0)
    cdfs /= cdfs[:, -1][:, None]
    return np.clip(cdfs, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------

def pit_values(pdfs: np.ndarray, zgrid: np.ndarray, z_true: np.ndarray) -> np.ndarray:
    """Probability Integral Transform: evaluate each CDF at its own true value.

    THE INTUITION, because this is the single most important concept here:

    Suppose a model is perfectly honest. For a galaxy whose true redshift sits
    at the 30th percentile of that galaxy's predicted distribution, PIT = 0.30.
    Across many galaxies, if the model is honest, the truth should land at
    every percentile equally often -- so the PIT values should be UNIFORM
    between 0 and 1, and a histogram of them should be FLAT.

    Deviations are diagnostic, and you can read the failure mode off the shape:

      U-shaped (piled up at 0 and 1)  -> OVERCONFIDENT. The truth keeps landing
                                         in the tails, so the PDFs are too
                                         narrow. This is the common failure.
      Hump in the middle              -> UNDERCONFIDENT. PDFs too wide; the
                                         model is hedging more than it needs to.
      Sloped / lopsided               -> BIASED. Predictions systematically
                                         too high or too low.

    This one plot tells you more about a photo-z model than any accuracy number.
    """
    cdfs = pdfs_to_cdfs(normalize_pdfs(pdfs, zgrid), zgrid)
    return np.array([
        np.interp(zt, zgrid, cdf) for zt, cdf in zip(z_true, cdfs)
    ])


def pit_uniformity_ks(pit: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov test of PIT values against Uniform(0,1).

    Returns (statistic, p-value). Smaller statistic = flatter = better
    calibrated. Treat the p-value with care: with tens of thousands of
    galaxies, any real model will be rejected at high significance. The
    statistic is the useful number for RANKING models, not the p-value.
    """
    res = kstest(pit, "uniform")
    return float(res.statistic), float(res.pvalue)


def coverage(
    pdfs: np.ndarray, zgrid: np.ndarray, z_true: np.ndarray,
    levels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical coverage of central credible intervals.

    For a stated 80% interval, does the truth actually fall inside it 80% of
    the time? Perfect calibration plots as the diagonal line.

    This is PIT information in a form that is easier to explain to someone who
    has never seen a PIT histogram: "when my model says it's 90% sure, it's
    right 90% of the time."
    """
    if levels is None:
        levels = np.linspace(0.05, 0.95, 19)
    pit = pit_values(pdfs, zgrid, z_true)
    # The truth lies inside the central interval of width `level` exactly when
    # its PIT value lies within that central band of [0, 1].
    empirical = np.array([
        np.mean(np.abs(pit - 0.5) <= lvl / 2.0) for lvl in levels
    ])
    return levels, empirical


# ----------------------------------------------------------------------------
# Accuracy of the whole distribution
# ----------------------------------------------------------------------------

def crps(pdfs: np.ndarray, zgrid: np.ndarray, z_true: np.ndarray) -> np.ndarray:
    """Continuous Ranked Probability Score, per galaxy. Lower is better.

        CRPS = integral over z of ( F(z) - H(z - z_true) )^2 dz

    where F is the predicted CDF and H is a step function that jumps from 0 to
    1 at the true redshift.

    Read it as: "how far is my predicted CDF from the CDF of a model that knew
    the exact answer?" It rewards being both SHARP and CORRECT, so it cannot be
    gamed by hedging -- a very wide PDF scores badly even though it is honest,
    and a narrow PDF in the wrong place scores badly too.

    It also has a nice property: for a PDF that is a spike at a single value,
    CRPS reduces to plain absolute error. So it is a strict generalisation of
    the point-estimate metric, in the same units as z.

    NUMERICAL NOTE (this one bit us, and the test suite caught it):
    The integrand has a jump discontinuity at z_true, because H does. Running
    a trapezoid rule straight across that jump smears it over one grid cell and
    leaves an O(dz) error -- about 0.6% on a 241-point grid, which is large
    enough to change how methods rank.

    So we split the integral at z_true instead:

        CRPS = integral_{zmin}^{y} F^2 dz  +  integral_{y}^{zmax} (F-1)^2 dz

    Both integrands are now smooth, we take their cumulative integrals once,
    and evaluate at y by interpolation. The discontinuity is handled exactly
    rather than approximated, which buys ~3 orders of magnitude of accuracy
    for no meaningful extra cost.
    """
    pdfs = normalize_pdfs(pdfs, zgrid)
    cdfs = pdfs_to_cdfs(pdfs, zgrid)
    y = np.asarray(z_true, dtype=np.float64)

    # Cumulative integrals of the two smooth pieces.
    lower = cumulative_trapezoid(cdfs ** 2, zgrid, axis=1, initial=0.0)
    upper = cumulative_trapezoid((cdfs - 1.0) ** 2, zgrid, axis=1, initial=0.0)

    y_clipped = np.clip(y, zgrid[0], zgrid[-1])
    lower_at_y = np.array([np.interp(yi, zgrid, row) for yi, row in zip(y_clipped, lower)])
    upper_at_y = np.array([np.interp(yi, zgrid, row) for yi, row in zip(y_clipped, upper)])

    out = lower_at_y + (upper[:, -1] - upper_at_y)

    # If the truth falls outside the grid, the region between contributes 1 per
    # unit length (F is 0 or 1 there while H is the opposite).
    out += np.where(y < zgrid[0], zgrid[0] - y, 0.0)
    out += np.where(y > zgrid[-1], y - zgrid[-1], 0.0)
    return out


def crps_gaussian_closed_form(mu, sigma, y):
    """Exact CRPS for a single Gaussian. Used only to TEST `crps` above.

        CRPS(N(mu, sigma), y) = sigma * [ w(2*Phi(w) - 1) + 2*phi(w) - 1/sqrt(pi) ]

    with w = (y - mu) / sigma. Having an analytic answer to check the numerical
    grid integration against is why the test suite can be trusted.
    """
    from scipy.stats import norm
    mu, sigma, y = np.asarray(mu), np.asarray(sigma), np.asarray(y)
    w = (y - mu) / sigma
    return sigma * (w * (2 * norm.cdf(w) - 1) + 2 * norm.pdf(w) - 1 / np.sqrt(np.pi))


# ----------------------------------------------------------------------------
# Point estimates: the metrics the photo-z literature reports
# ----------------------------------------------------------------------------

def point_estimates(pdfs: np.ndarray, zgrid: np.ndarray, how: str = "mean") -> np.ndarray:
    """Collapse each PDF to one number.

    'mean' is the expected value (best for squared-error-type questions),
    'mode' is the single most probable redshift (what template-fitting codes
    traditionally report, and more robust when the PDF is bimodal).
    """
    pdfs = normalize_pdfs(pdfs, zgrid)
    if how == "mean":
        return _trapz(pdfs * zgrid[None, :], zgrid, axis=1)
    if how == "mode":
        return zgrid[np.argmax(pdfs, axis=1)]
    if how == "median":
        cdfs = pdfs_to_cdfs(pdfs, zgrid)
        return np.array([np.interp(0.5, c, zgrid) for c in cdfs])
    raise ValueError(f"unknown point estimate: {how}")


def point_metrics(z_phot: np.ndarray, z_spec: np.ndarray) -> dict[str, float]:
    """The standard photo-z summary statistics.

    Everything is measured in dz = (z_phot - z_spec) / (1 + z_spec). The
    (1 + z) denominator is not cosmetic: redshift uncertainty naturally scales
    that way, so dividing by it makes the metric comparable across redshift.
    Reporting raw (z_phot - z_spec) instead is a common giveaway that someone
    has not read the photo-z literature.

      bias          mean of dz. Systematic over/under-prediction.
      sigma_NMAD    1.4826 * median(|dz - median(dz)|). A robust standard
                    deviation: the constant makes it equal the usual sigma for
                    Gaussian data, but outliers cannot inflate it. This is THE
                    headline number in photo-z papers.
      outlier_frac  fraction with |dz| > 0.15. The conventional threshold for
                    a "catastrophic" failure -- a galaxy placed at a completely
                    wrong distance.
    """
    dz = (z_phot - z_spec) / (1.0 + z_spec)
    return {
        "bias": float(np.mean(dz)),
        "sigma_NMAD": float(1.4826 * np.median(np.abs(dz - np.median(dz)))),
        "outlier_frac": float(np.mean(np.abs(dz) > 0.15)),
        "rmse": float(np.sqrt(np.mean(dz ** 2))),
    }


def evaluate(
    pdfs: np.ndarray, zgrid: np.ndarray, z_true: np.ndarray, name: str = "",
) -> dict[str, float]:
    """Everything above, in one dict."""
    pdfs = normalize_pdfs(pdfs, zgrid)
    pit = pit_values(pdfs, zgrid, z_true)
    ks_stat, _ = pit_uniformity_ks(pit)
    z_point = point_estimates(pdfs, zgrid, how="mean")

    out = {"name": name, "n": len(z_true)}
    out.update(point_metrics(z_point, z_true))
    out["CRPS"] = float(np.mean(crps(pdfs, zgrid, z_true)))
    out["PIT_KS"] = ks_stat
    # How far empirical coverage sits from the ideal diagonal, averaged.
    levels, emp = coverage(pdfs, zgrid, z_true)
    out["coverage_err"] = float(np.mean(np.abs(emp - levels)))
    return out
