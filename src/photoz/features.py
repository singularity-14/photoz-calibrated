"""Turning a raw catalogue into model inputs.

Two ideas do all the work here, and both are physics, not machine learning.

1. DEREDDENING. Dust in our own Milky Way sits between us and every galaxy we
   look at. It dims light, and it dims blue light more than red light. So it
   makes galaxies look redder than they are -- which is exactly the signal we
   are trying to measure. SDSS provides a per-band estimate of this effect
   ("extinction"); we subtract it. Skipping this step puts a spatial pattern
   (the shape of the Milky Way's dust) straight into your redshift predictions.

2. USE COLOURS, NOT MAGNITUDES. A galaxy's observed brightness depends on how
   far away it is AND how intrinsically luminous it is -- two unknowns tangled
   together. But the RATIO of brightness between two filters (a "colour", which
   in magnitudes is a difference) is mostly independent of distance. Colour
   tracks the shape of the spectrum, and as a galaxy recedes, its spectrum
   shifts to the red, so its colours change in a predictable way. Colour is
   where the redshift information actually lives.

   We keep one magnitude (r) as well, because it carries weak extra information:
   at fixed colour, a fainter galaxy is on average further away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BANDS

# u-g, g-r, r-i, i-z : adjacent pairs, which is the standard choice.
COLOUR_PAIRS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]
FEATURE_NAMES = [f"{a}-{b}" for a, b in COLOUR_PAIRS] + ["r"]

# SDSS writes -9999 (and friends) when a measurement failed.
_BAD_SENTINEL = -100.0
_MAG_MAX = 40.0


def deredden(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract Milky Way dust extinction from each band."""
    out = df.copy()
    for b in BANDS:
        out[f"mag_{b}"] = df[f"modelMag_{b}"] - df[f"extinction_{b}"]
    return out


def quality_cut(df: pd.DataFrame, max_magerr: float = 0.5, verbose: bool = True) -> pd.DataFrame:
    """Drop rows we should not train on.

    `max_magerr` of 0.5 mag is a loose cut. A 0.5 mag uncertainty is already a
    ~60% uncertainty in brightness; beyond that the colour is mostly noise and
    the galaxy contributes confusion rather than information.

    IMPORTANT, AND NOT A DETAIL:
    On real SDSS data this single cut removes about half the sample, and almost
    all of that comes from the **u band alone** (~48%, versus <1% for every
    other band). SDSS's u filter is by far its least sensitive -- both the
    atmosphere and the detector work against near-ultraviolet light.

    That matters scientifically, because the cut is not random. Galaxies that
    are faint in u are preferentially the faint, red and distant ones -- which
    is exactly the population the model is worst at. So this data-cleaning
    choice *compounds* the bright-galaxy selection bias measured in
    `run_all.py`, rather than being neutral with respect to it.

    Loosening `max_magerr` keeps more faint galaxies at the price of noisier
    colours. That tradeoff is a real experiment worth running, not a settled
    question -- which is why the threshold is a parameter and not a constant.
    """
    keep = np.ones(len(df), dtype=bool)
    per_band_dropped = {}

    for b in BANDS:
        mag = df[f"mag_{b}"].to_numpy()
        err = df[f"modelMagErr_{b}"].to_numpy()
        band_ok = (
            np.isfinite(mag) & (mag > _BAD_SENTINEL) & (mag < _MAG_MAX)
            & np.isfinite(err) & (err >= 0) & (err < max_magerr)
        )
        per_band_dropped[b] = int((~band_ok).sum())
        keep &= band_ok

    z = df["z_spec"].to_numpy()
    keep &= np.isfinite(z) & (z > 0.01) & (z < 1.0)

    n_dropped = int((~keep).sum())
    if verbose:
        pct = 100 * n_dropped / max(len(df), 1)
        print(f"Quality cut: dropped {n_dropped} / {len(df)} rows ({pct:.1f}%)")
        worst = max(per_band_dropped, key=per_band_dropped.get)
        print("  per band: " + ", ".join(
            f"{b}={per_band_dropped[b]}" for b in BANDS))
        print(f"  dominated by the {worst} band -- see the note in quality_cut():")
        print("  this cut preferentially removes faint red galaxies, so it")
        print("  compounds the selection bias rather than being neutral.")
    return df.loc[keep].reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, z) ready for a model.

    X has 5 columns: four colours and the r-band magnitude.
    """
    X = np.column_stack(
        [df[f"mag_{a}"].to_numpy() - df[f"mag_{b}"].to_numpy() for a, b in COLOUR_PAIRS]
        + [df["mag_r"].to_numpy()]
    )
    z = df["z_spec"].to_numpy()
    return X.astype(np.float64), z.astype(np.float64)


def prepare(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Full path: raw catalogue -> (X, z)."""
    return build_features(quality_cut(deredden(df)))


class Standardizer:
    """Subtract the mean, divide by the standard deviation, per feature.

    Fit on the TRAINING set only, then apply unchanged to validation and test.
    Fitting on everything would leak information about the test set into the
    model -- a small leak here, but the habit matters.
    """

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def train_val_test_split(
    X: np.ndarray, z: np.ndarray, seed: int = 0,
    frac_train: float = 0.7, frac_val: float = 0.15,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Random 70/15/15 split.

    A note on honesty: a random split is the easy case. Real photo-z deployment
    is harder, because the galaxies with spectroscopic redshifts are brighter
    and nearer than the ones you want to apply the model to. We measure that
    separately in `evaluate_faint_extrapolation` (see run_all.py) rather than
    pretending the random split tells the whole story.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_tr = int(frac_train * len(X))
    n_va = int(frac_val * len(X))
    parts = {
        "train": idx[:n_tr],
        "val": idx[n_tr:n_tr + n_va],
        "test": idx[n_tr + n_va:],
    }
    return {k: (X[v], z[v]) for k, v in parts.items()}
