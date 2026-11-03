"""The non-ML baselines. Build these FIRST, before any neural network.

This is a rule in this field, and it is worth understanding why rather than
just obeying it. If you skip the baseline, you cannot answer the only question
a reviewer actually cares about: "is your neural network buying me anything
over what we already had?" A deep model that loses to a 1995 polynomial is a
useful result; a deep model with nothing to compare against is not a result at
all.

Two baselines here, from the two historical families of photo-z methods:

  PolynomialPhotoZ  - the classic EMPIRICAL fit (Connolly et al. 1995). Fit a
                      polynomial from colours to redshift. Fast, simple, and
                      deliberately naive about uncertainty -- which makes it an
                      excellent illustration of what bad calibration looks like.

  KNNPhotoZ         - nearest neighbours in colour space. Still empirical, but
                      it gets a genuine PDF almost for free: find the 100 most
                      similar galaxies that have known redshifts and look at
                      how their redshifts are distributed. If they all agree,
                      you get a narrow PDF; if they disagree, a wide one. The
                      uncertainty emerges from the data rather than being
                      assumed.

A NOTE ON THE THIRD FAMILY (template fitting)
---------------------------------------------
The other major approach -- used by codes like BPZ, EAZY and LePhare -- does no
training at all. It takes model spectra of real galaxy types, mathematically
redshifts them, computes what colours they WOULD have at each trial redshift,
and asks which redshift best reproduces the observed colours. Its chi-squared
curve over redshift converts directly into a P(z), which is elegant.

It is not implemented here because it requires an external library of galaxy
spectral templates and the SDSS filter response curves. It is the single best
extension to this project -- see "Where this goes next" in the README.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures


class PolynomialPhotoZ:
    """Polynomial regression from colours to redshift (Connolly et al. 1995).

    The PDF is a Gaussian centred on the fit, with a single width taken from
    the scatter of the training residuals. Note what that assumption means:
    EVERY galaxy is assigned the SAME uncertainty, regardless of whether it is
    bright and easy or faint and ambiguous. That is obviously wrong, and the
    calibration plots will show exactly how wrong. That is the point of
    including it.
    """

    def __init__(self, degree: int = 3):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)

    def fit(self, X: np.ndarray, z: np.ndarray) -> "PolynomialPhotoZ":
        A = self.poly.fit_transform(X)
        # Least squares; lstsq is stable even when columns are near-collinear,
        # which high-degree polynomial features certainly are.
        self.coef_, *_ = np.linalg.lstsq(A, z, rcond=None)
        resid = z - A @ self.coef_
        self.sigma_ = float(np.std(resid))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.poly.transform(X) @ self.coef_

    def predict_pdf(self, X: np.ndarray, zgrid: np.ndarray) -> np.ndarray:
        mu = self.predict(X)[:, None]
        return np.exp(-0.5 * ((zgrid[None, :] - mu) / self.sigma_) ** 2)


class KNNPhotoZ:
    """k-nearest-neighbour photo-z with a kernel-density PDF.

    For each test galaxy: find its k most similar training galaxies in colour
    space, then smooth their known redshifts into a continuous density.

    Why this gets uncertainty right in a way the polynomial cannot: in parts of
    colour space where two different redshifts produce nearly identical
    colours, the neighbours disagree, and the resulting PDF is genuinely wide
    -- or even two-peaked. A bimodal PDF is the honest answer to a question
    that has two plausible answers, and no single number can express it.
    """

    def __init__(self, k: int = 100, bandwidth: float | None = None):
        self.k = k
        self.bandwidth = bandwidth  # None => adaptive, per galaxy

    def fit(self, X: np.ndarray, z: np.ndarray) -> "KNNPhotoZ":
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(X)
        self.z_train_ = np.asarray(z)
        return self

    def predict_pdf(
        self, X: np.ndarray, zgrid: np.ndarray, batch_size: int = 512,
    ) -> np.ndarray:
        """Gaussian KDE over the neighbours' redshifts.

        Batched, because the intermediate array is (n_galaxies, k, n_grid) and
        would otherwise be gigabytes.
        """
        out = np.empty((len(X), len(zgrid)), dtype=np.float64)

        for start in range(0, len(X), batch_size):
            Xb = X[start:start + batch_size]
            _, idx = self.nn_.kneighbors(Xb)
            z_nn = self.z_train_[idx]                      # (b, k)

            if self.bandwidth is None:
                # Scott's rule, per galaxy: narrow when the neighbours agree,
                # wide when they do not. The floor stops the PDF collapsing to
                # a spike when all k neighbours happen to be identical.
                spread = z_nn.std(axis=1, keepdims=True)
                h = np.maximum(spread * self.k ** (-1 / 5), 0.005)
            else:
                h = np.full((len(Xb), 1), self.bandwidth)

            diff = (zgrid[None, None, :] - z_nn[:, :, None]) / h[:, :, None]
            out[start:start + batch_size] = np.exp(-0.5 * diff ** 2).mean(axis=1)

        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, idx = self.nn_.kneighbors(X)
        return np.median(self.z_train_[idx], axis=1)
