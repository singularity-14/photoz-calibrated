"""Tests for the scoring code.

These matter more than they look. Every conclusion in this project rests on
CRPS and PIT being computed correctly, and both are easy to get subtly wrong
(off-by-one in the integration, an unnormalised PDF, a CDF that does not reach
1). So we check them against cases where the right answer is known analytically.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from photoz import metrics  # noqa: E402


@pytest.fixture
def zgrid():
    """A grid wide enough that truncating the Gaussian tails is negligible.

    This width is not arbitrary. With sigma up to 1.5 in these tests, a +/-6
    grid cuts off enough probability mass that renormalisation shifts CRPS by
    ~1e-3 relative -- which showed up as a real test failure. Widening to +/-10
    (at coarser spacing!) cut that error 3x, which is how we confirmed the
    residual was truncation rather than a bug in the integration.
    See test_crps_converges_as_the_grid_refines below.
    """
    return np.linspace(-10.0, 10.0, 4001)


def gaussian_pdfs(mu, sigma, zgrid):
    mu = np.atleast_1d(mu)[:, None]
    sigma = np.atleast_1d(sigma)[:, None]
    return norm.pdf(zgrid[None, :], loc=mu, scale=sigma)


# ---------------------------------------------------------------------------
# CRPS against its closed form -- the single most important test here
# ---------------------------------------------------------------------------

def test_crps_matches_closed_form_for_gaussian(zgrid):
    """Our grid-integrated CRPS must agree with the analytic Gaussian result."""
    rng = np.random.default_rng(0)
    mu = rng.uniform(-1.5, 1.5, 200)
    sigma = rng.uniform(0.3, 1.5, 200)
    y = rng.uniform(-2.5, 2.5, 200)

    numerical = metrics.crps(gaussian_pdfs(mu, sigma, zgrid), zgrid, y)
    analytic = metrics.crps_gaussian_closed_form(mu, sigma, y)

    np.testing.assert_allclose(numerical, analytic, rtol=1e-3, atol=1e-4)


def test_crps_converges_as_the_grid_refines():
    """Refining the grid must drive the error toward zero.

    This is the test that distinguishes "numerically approximate" from "wrong".
    A bug in the integration would leave a floor that refinement cannot cross;
    a correct scheme converges. It also documents the accuracy you actually get
    at a given resolution.
    """
    rng = np.random.default_rng(0)
    mu = rng.uniform(-1.5, 1.5, 100)
    sigma = rng.uniform(0.3, 1.5, 100)
    y = rng.uniform(-2.5, 2.5, 100)
    analytic = metrics.crps_gaussian_closed_form(mu, sigma, y)

    errors = []
    for n_points in (1001, 2001, 4001):
        grid = np.linspace(-10.0, 10.0, n_points)
        numerical = metrics.crps(gaussian_pdfs(mu, sigma, grid), grid, y)
        errors.append(np.max(np.abs(numerical - analytic) / analytic))

    assert errors[0] > errors[1] > errors[2], f"not converging: {errors}"
    assert errors[-1] < 1e-3


def test_crps_reduces_to_absolute_error_for_a_spike(zgrid):
    """A near-delta-function PDF should score CRPS ~= |prediction - truth|."""
    mu, y = np.array([0.5]), np.array([1.1])
    pdfs = gaussian_pdfs(mu, np.array([0.002]), zgrid)
    assert metrics.crps(pdfs, zgrid, y)[0] == pytest.approx(abs(1.1 - 0.5), abs=5e-3)


def test_crps_is_minimised_by_the_true_distribution(zgrid):
    """Sharper is only better when you are right. Lying about width costs you."""
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1.0, 4000)

    honest = metrics.crps(gaussian_pdfs(np.zeros(4000), np.ones(4000), zgrid), zgrid, y).mean()
    overconfident = metrics.crps(
        gaussian_pdfs(np.zeros(4000), np.full(4000, 0.3), zgrid), zgrid, y).mean()
    underconfident = metrics.crps(
        gaussian_pdfs(np.zeros(4000), np.full(4000, 3.0), zgrid), zgrid, y).mean()

    assert honest < overconfident
    assert honest < underconfident


# ---------------------------------------------------------------------------
# PIT: the calibration diagnostic
# ---------------------------------------------------------------------------

def test_pit_is_uniform_when_the_model_is_correct(zgrid):
    """A correctly specified model gives flat PIT. This is the whole premise."""
    rng = np.random.default_rng(2)
    n = 5000
    mu = rng.uniform(-1, 1, n)
    sigma = rng.uniform(0.5, 1.5, n)
    y = rng.normal(mu, sigma)          # truth drawn FROM the predicted dist

    pit = metrics.pit_values(gaussian_pdfs(mu, sigma, zgrid), zgrid, y)
    ks, _ = metrics.pit_uniformity_ks(pit)

    assert ks < 0.03, f"correctly specified model should be calibrated, KS={ks}"
    assert 0.47 < pit.mean() < 0.53


def test_pit_detects_overconfidence(zgrid):
    """Too-narrow PDFs must pile PIT up at 0 and 1 (the U shape)."""
    rng = np.random.default_rng(3)
    n = 5000
    y = rng.normal(0.0, 1.0, n)
    # claim sigma=0.3 when the truth scatters with sigma=1.0
    pit = metrics.pit_values(
        gaussian_pdfs(np.zeros(n), np.full(n, 0.3), zgrid), zgrid, y)

    in_tails = np.mean((pit < 0.1) | (pit > 0.9))
    assert in_tails > 0.5, "overconfident model should push PIT into the tails"
    assert metrics.pit_uniformity_ks(pit)[0] > 0.2


def test_pit_detects_underconfidence(zgrid):
    """Too-wide PDFs must bunch PIT in the middle."""
    rng = np.random.default_rng(4)
    n = 5000
    y = rng.normal(0.0, 1.0, n)
    pit = metrics.pit_values(
        gaussian_pdfs(np.zeros(n), np.full(n, 4.0), zgrid), zgrid, y)

    in_middle = np.mean(np.abs(pit - 0.5) < 0.15)
    assert in_middle > 0.5, "underconfident model should bunch PIT near 0.5"


def test_coverage_is_the_diagonal_when_calibrated(zgrid):
    rng = np.random.default_rng(5)
    n = 4000
    mu = rng.uniform(-1, 1, n)
    sigma = rng.uniform(0.5, 1.5, n)
    y = rng.normal(mu, sigma)

    levels, emp = metrics.coverage(gaussian_pdfs(mu, sigma, zgrid), zgrid, y)
    np.testing.assert_allclose(emp, levels, atol=0.03)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def test_normalize_makes_pdfs_integrate_to_one(zgrid):
    pdfs = gaussian_pdfs([0.0, 1.0], [1.0, 0.5], zgrid) * 17.3  # wrong scale
    normed = metrics.normalize_pdfs(pdfs, zgrid)
    area = np.trapezoid(normed, zgrid, axis=1)
    np.testing.assert_allclose(area, 1.0, rtol=1e-6)


def test_cdfs_are_monotonic_and_end_at_one(zgrid):
    cdfs = metrics.pdfs_to_cdfs(
        metrics.normalize_pdfs(gaussian_pdfs([0.0, 0.5], [1.0, 0.7], zgrid), zgrid), zgrid)
    assert np.all(np.diff(cdfs, axis=1) >= -1e-12)
    np.testing.assert_allclose(cdfs[:, -1], 1.0, rtol=1e-9)


def test_point_estimates_recover_the_gaussian_centre(zgrid):
    pdfs = gaussian_pdfs([0.3, -0.8], [0.4, 0.6], zgrid)
    for how in ("mean", "mode", "median"):
        np.testing.assert_allclose(
            metrics.point_estimates(pdfs, zgrid, how=how), [0.3, -0.8], atol=0.02)


def test_point_metrics_definitions():
    """sigma_NMAD must ignore outliers; rmse must not."""
    z_spec = np.zeros(1000)
    z_phot = np.full(1000, 0.0)
    z_phot[:950] = 0.01          # small errors
    z_phot[950:] = 5.0           # 5% catastrophic

    m = metrics.point_metrics(z_phot, z_spec)
    assert m["sigma_NMAD"] < 0.02, "robust scatter must ignore the 5% outliers"
    assert m["rmse"] > 1.0, "rmse must feel the outliers"
    assert m["outlier_frac"] == pytest.approx(0.05, abs=1e-9)
