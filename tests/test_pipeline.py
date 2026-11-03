"""End-to-end tests on the synthetic catalogue (fast, offline, deterministic)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from photoz import baselines, data, features, mdn, metrics  # noqa: E402

ZGRID = np.linspace(0.0, 1.2, 121)


@pytest.fixture(scope="module")
def dataset():
    df = data.make_synthetic(n=4000, seed=0)
    X, z = features.prepare(df)
    return features.train_val_test_split(X, z, seed=0)


def test_synthetic_catalogue_has_expected_columns():
    df = data.make_synthetic(n=100, seed=0)
    for band in data.BANDS:
        assert f"modelMag_{band}" in df.columns
        assert f"modelMagErr_{band}" in df.columns
        assert f"extinction_{band}" in df.columns
    assert df["z_spec"].between(0.0, 1.0).all()


def test_dereddening_subtracts_extinction():
    df = data.make_synthetic(n=50, seed=1)
    df["extinction_r"] = 0.3
    out = features.deredden(df)
    np.testing.assert_allclose(out["mag_r"], df["modelMag_r"] - 0.3)


def test_colours_are_magnitude_differences():
    df = data.make_synthetic(n=50, seed=2)
    dered = features.deredden(df)
    X, _ = features.build_features(dered)
    np.testing.assert_allclose(X[:, 1], dered["mag_g"] - dered["mag_r"])
    np.testing.assert_allclose(X[:, 4], dered["mag_r"])


def test_standardizer_fits_on_train_only():
    rng = np.random.default_rng(0)
    X_tr, X_te = rng.normal(5, 2, (100, 3)), rng.normal(50, 20, (40, 3))
    s = features.Standardizer().fit(X_tr)
    np.testing.assert_allclose(s.transform(X_tr).mean(axis=0), 0, atol=1e-9)
    # test set must NOT come out standardized - it uses the train statistics
    assert np.abs(s.transform(X_te).mean()) > 1.0


def test_splits_are_disjoint_and_complete():
    X = np.arange(1000).reshape(-1, 1).astype(float)
    z = np.random.default_rng(0).uniform(0, 1, 1000)
    split = features.train_val_test_split(X, z, seed=0)
    ids = [set(Xs[:, 0].astype(int)) for Xs, _ in split.values()]
    assert set.union(*ids) == set(range(1000))
    assert sum(len(i) for i in ids) == 1000  # no overlap


@pytest.mark.parametrize("name", ["poly", "knn"])
def test_baselines_produce_valid_pdfs(dataset, name):
    (X_tr, z_tr), (X_te, _) = dataset["train"], dataset["test"]
    scaler = features.Standardizer().fit(X_tr)

    model = (baselines.PolynomialPhotoZ(degree=2) if name == "poly"
             else baselines.KNNPhotoZ(k=25))
    model.fit(scaler.transform(X_tr), z_tr)
    pdfs = model.predict_pdf(scaler.transform(X_te), ZGRID)

    assert pdfs.shape == (len(X_te), len(ZGRID))
    assert np.all(pdfs >= 0), "a probability density cannot be negative"
    assert np.all(np.isfinite(pdfs))
    area = np.trapezoid(metrics.normalize_pdfs(pdfs, ZGRID), ZGRID, axis=1)
    np.testing.assert_allclose(area, 1.0, rtol=1e-6)


def test_knn_pdf_is_wider_when_neighbours_disagree():
    """The core claim of the kNN method: uncertainty comes from the data."""
    rng = np.random.default_rng(0)
    # two regions of feature space: one where z is tight, one where it is not
    X_tr = np.vstack([np.zeros((200, 1)), np.ones((200, 1))])
    z_tr = np.concatenate([
        rng.normal(0.3, 0.005, 200),   # tight
        rng.uniform(0.1, 0.9, 200),    # ambiguous
    ])
    knn = baselines.KNNPhotoZ(k=50).fit(X_tr, z_tr)
    pdfs = metrics.normalize_pdfs(knn.predict_pdf(np.array([[0.0], [1.0]]), ZGRID), ZGRID)

    # standard deviation of each PDF
    mean = np.trapezoid(pdfs * ZGRID, ZGRID, axis=1)
    var = np.trapezoid(pdfs * (ZGRID - mean[:, None]) ** 2, ZGRID, axis=1)
    assert np.sqrt(var[1]) > 3 * np.sqrt(var[0])


def test_mdn_trains_and_improves_on_the_prior(dataset):
    (X_tr, z_tr), (X_va, z_va), (X_te, z_te) = (
        dataset["train"], dataset["val"], dataset["test"])
    scaler = features.Standardizer().fit(X_tr)

    model, history = mdn.train_mdn(
        scaler.transform(X_tr), z_tr, scaler.transform(X_va), z_va,
        n_components=3, epochs=30, patience=10, seed=0, verbose=False,
    )
    assert history["val"][-1] < history["val"][0], "validation NLL should improve"

    pdfs = model.predict_pdf(scaler.transform(X_te), ZGRID)
    assert np.all(np.isfinite(pdfs)) and np.all(pdfs >= 0)

    res = metrics.evaluate(pdfs, ZGRID, z_te, name="MDN")
    # On this synthetic data the signal is strong; a model that learned nothing
    # would have sigma_NMAD comparable to the spread of z itself (~0.2).
    assert res["sigma_NMAD"] < 0.1, f"MDN failed to learn: {res}"


def test_evaluate_returns_all_expected_keys(dataset):
    (X_te, z_te) = dataset["test"]
    pdfs = np.ones((len(X_te), len(ZGRID)))  # a deliberately useless flat PDF
    res = metrics.evaluate(pdfs, ZGRID, z_te, name="flat")
    for key in ("sigma_NMAD", "bias", "outlier_frac", "CRPS", "PIT_KS", "coverage_err"):
        assert key in res and np.isfinite(res[key])
