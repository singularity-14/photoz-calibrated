"""Run the whole project end to end.

    python run_all.py                  # uses cached real data, else synthetic
    python run_all.py --fetch          # download fresh data from SDSS first
    python run_all.py --synthetic      # force the fake catalogue (offline, fast)

Produces every figure and the results table in results/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from photoz import baselines, data, features, mdn, metrics, plots  # noqa: E402

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sdss_photoz.csv"
RESULTS = ROOT / "results"

# Every method is evaluated on this same grid of trial redshifts. Sharing the
# grid is what makes CRPS and PIT comparable between methods.
ZGRID = np.linspace(0.0, 1.2, 241)


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def load_dataset(args) -> tuple[pd.DataFrame, bool]:
    """Returns (dataframe, is_real_data)."""
    if args.synthetic:
        print("Using SYNTHETIC data (no scientific claims may be made from it).")
        return data.make_synthetic(n=args.n_synthetic, seed=0), False

    if args.fetch or not DATA_PATH.exists():
        banner("Downloading from SDSS DR17")
        try:
            df = data.fetch_sdss(n_total=args.n_fetch, out_path=DATA_PATH)
            return df, True
        except Exception as exc:  # noqa: BLE001
            print(f"\nDownload failed: {type(exc).__name__}: {exc}")
            print("Falling back to SYNTHETIC data so the pipeline still runs.")
            return data.make_synthetic(n=args.n_synthetic, seed=0), False

    print(f"Loading cached real data from {DATA_PATH}")
    return data.load(DATA_PATH), True


def fit_all(split, args):
    """Train all three methods. Returns {name: pdfs_on_test_grid}."""
    (X_tr, z_tr) = split["train"]
    (X_va, z_va) = split["val"]
    (X_te, z_te) = split["test"]

    scaler = features.Standardizer().fit(X_tr)
    X_tr_s, X_va_s, X_te_s = (scaler.transform(x) for x in (X_tr, X_va, X_te))

    results = {}

    banner("Baseline 1/2 — polynomial colour fit (Connolly et al. 1995)")
    poly = baselines.PolynomialPhotoZ(degree=3).fit(X_tr_s, z_tr)
    print(f"  fitted; residual scatter sigma = {poly.sigma_:.4f}")
    results["Polynomial"] = poly.predict_pdf(X_te_s, ZGRID)

    banner("Baseline 2/2 — k nearest neighbours in colour space")
    knn = baselines.KNNPhotoZ(k=args.k_neighbors).fit(X_tr_s, z_tr)
    print(f"  fitted on {len(X_tr_s)} galaxies, k = {args.k_neighbors}")
    results["kNN"] = knn.predict_pdf(X_te_s, ZGRID)

    banner("Mixture Density Network")
    model, history = mdn.train_mdn(
        X_tr_s, z_tr, X_va_s, z_va,
        n_components=args.n_components, epochs=args.epochs, seed=0,
    )
    results["MDN"] = model.predict_pdf(X_te_s, ZGRID)

    return results, history, (X_te_s, z_te)


def summarise(results, z_te) -> list[dict]:
    return [metrics.evaluate(pdfs, ZGRID, z_te, name=name)
            for name, pdfs in results.items()]


def print_table(summary: list[dict]) -> None:
    cols = ["name", "sigma_NMAD", "bias", "outlier_frac", "CRPS", "PIT_KS", "coverage_err"]
    name_width = max(12, max(len(s["name"]) for s in summary) + 2)
    widths = [name_width, 11, 9, 13, 9, 9, 13]
    print("  " + "".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  " + "-" * sum(widths))
    for s in summary:
        row = [s["name"]] + [f"{s[c]:.4f}" for c in cols[1:]]
        print("  " + "".join(v.ljust(w) for v, w in zip(row, widths)))


def selection_bias_experiment(X, z, df, args) -> dict | None:
    """The scientific core: what happens when training and target differ?

    In reality you train on galaxies that have spectra -- which are the BRIGHT,
    nearby ones, because taking a spectrum of a faint galaxy is expensive. You
    then apply the model to the faint galaxies you have no spectra for. The two
    populations are not the same, and a random train/test split hides that
    completely.

    So: train only on the brightest 50%, test only on the faintest 25%, and see
    what it costs. This is the number that matters for real deployment, and it
    is always worse than the random-split number.
    """
    r_mag = X[:, -1]  # last feature is the r-band magnitude
    bright_cut = np.percentile(r_mag, 50)
    faint_cut = np.percentile(r_mag, 75)

    train_mask = r_mag <= bright_cut
    test_mask = r_mag > faint_cut

    if train_mask.sum() < 500 or test_mask.sum() < 200:
        print("  not enough galaxies for the selection-bias split; skipping")
        return None

    print(f"  train on r <= {bright_cut:.2f}  ({train_mask.sum()} galaxies)")
    print(f"  test  on r >  {faint_cut:.2f}  ({test_mask.sum()} galaxies)")

    X_tr, z_tr = X[train_mask], z[train_mask]
    X_te, z_te = X[test_mask], z[test_mask]

    # carve a validation slice out of the bright training set
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X_tr))
    n_va = max(int(0.15 * len(idx)), 50)
    va_idx, tr_idx = idx[:n_va], idx[n_va:]

    scaler = features.Standardizer().fit(X_tr[tr_idx])
    model, _ = mdn.train_mdn(
        scaler.transform(X_tr[tr_idx]), z_tr[tr_idx],
        scaler.transform(X_tr[va_idx]), z_tr[va_idx],
        n_components=args.n_components, epochs=args.epochs, seed=0, verbose=False,
    )
    pdfs = model.predict_pdf(scaler.transform(X_te), ZGRID)
    return metrics.evaluate(pdfs, ZGRID, z_te, name="MDN (bright->faint)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true", help="download fresh SDSS data")
    p.add_argument("--synthetic", action="store_true", help="force synthetic data")
    p.add_argument("--n-fetch", type=int, default=60_000)
    p.add_argument("--n-synthetic", type=int, default=20_000)
    p.add_argument("--k-neighbors", type=int, default=100)
    p.add_argument("--n-components", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()

    RESULTS.mkdir(exist_ok=True)
    df, is_real = load_dataset(args)

    banner("Preparing features")
    X, z = features.prepare(df)
    print(f"{len(X)} galaxies, {X.shape[1]} features: {features.FEATURE_NAMES}")
    print(f"redshift range {z.min():.3f} - {z.max():.3f}, median {np.median(z):.3f}")

    split = features.train_val_test_split(X, z, seed=0)
    for k, (Xs, _) in split.items():
        print(f"  {k:6s} {len(Xs):6d}")

    results, history, (_, z_te) = fit_all(split, args)

    banner("Results on the random test split")
    summary = summarise(results, z_te)
    print_table(summary)

    banner("Results under realistic selection bias (train bright -> apply faint)")
    bias_result = selection_bias_experiment(X, z, df, args)
    if bias_result:
        print_table([s for s in summary if s["name"] == "MDN"] + [bias_result])
        degradation = bias_result["sigma_NMAD"] / next(
            s["sigma_NMAD"] for s in summary if s["name"] == "MDN")
        print(f"\n  sigma_NMAD is {degradation:.2f}x worse on the faint population.")
        print("  This is the honest number for real-world use.")

    banner("Writing figures")
    for fn, label in [
        (lambda: plots.plot_pit(results, ZGRID, z_te, RESULTS), "PIT histograms"),
        (lambda: plots.plot_coverage(results, ZGRID, z_te, RESULTS), "coverage"),
        (lambda: plots.plot_scatter(results, ZGRID, z_te, RESULTS), "scatter"),
        (lambda: plots.plot_example_pdfs(results, ZGRID, z_te, RESULTS), "example PDFs"),
        (lambda: plots.plot_metric_comparison(summary, RESULTS), "metric comparison"),
        (lambda: plots.plot_training_curve(history, RESULTS), "training curve"),
    ]:
        print(f"  {label:20s} -> {fn().name}")

    payload = {
        "data_source": "SDSS DR17" if is_real else "SYNTHETIC (not real data)",
        "n_galaxies": int(len(X)),
        "random_split": summary,
        "selection_bias": bias_result,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"\n  metrics.json        -> results/metrics.json")

    if not is_real:
        print("\n" + "!" * 70)
        print("! Ran on SYNTHETIC data. Numbers above are NOT scientific results.")
        print("! Run with --fetch to use real SDSS DR17 galaxies.")
        print("!" * 70)


if __name__ == "__main__":
    main()
