"""Getting the data.

We need galaxies where we know BOTH:
  - their brightness through 5 colour filters (u, g, r, i, z)  -> the *input*
  - their true redshift, measured from a spectrum               -> the *label*

SDSS (the Sloan Digital Sky Survey) has millions of these. We query its public
SkyServer database with SQL.

NOTE ON WHY WE DON'T USE `astroquery`:
`astroquery.sdss` is the usual convenience wrapper, but in testing it returned
unparseable responses intermittently. Talking to the SkyServer HTTP endpoint
directly is fewer moving parts, and it lets us control retries, timeouts and
our User-Agent string. Identifying your client honestly is good manners when
you are hitting someone else's public science archive.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# DR17 is the final data release of SDSS-IV. Pinning the release makes this
# reproducible: a query run next year returns the same rows.
SKYSERVER_URL = "https://skyserver.sdss.org/dr17/en/tools/search/x_results.aspx"

USER_AGENT = "photoz-calibrated/0.1 (educational portfolio project)"

# The five SDSS filters, ordered from ultraviolet to infrared.
BANDS = ["u", "g", "r", "i", "z"]


def _build_query(ra_min: float, ra_max: float, limit: int) -> str:
    """One SQL query covering a slice of sky between ra_min and ra_max.

    The WHERE clauses are quality cuts. Each one is a real decision:

    - class = 'GALAXY'   we only want galaxies (not stars or quasars)
    - zWarning = 0       SDSS's own flag saying "this redshift is trustworthy".
                         Without it you inherit bad spectroscopic fits as labels.
    - clean = 1          SDSS's flag for photometry free of known defects
                         (saturation, bad columns, deblending problems...).
    - z BETWEEN 0.01 AND 1.0
                         Below 0.01 the "redshift" is dominated by the galaxy's
                         own motion, not cosmic expansion. Above ~1.0 SDSS has
                         very few galaxies, so there is nothing to learn from.
    - modelMag_r 14..21  Brighter than 14 saturates the detector; fainter than
                         21 the photometry is too noisy to be useful.
    """
    cols = ", ".join(
        ["s.z AS z_spec", "s.zErr AS z_spec_err"]
        + [f"p.modelMag_{b}" for b in BANDS]
        + [f"p.modelMagErr_{b}" for b in BANDS]
        + [f"p.extinction_{b}" for b in BANDS]
        + ["p.ra", "p.dec"]
    )
    return (
        f"SELECT TOP {limit} {cols} "
        "FROM SpecObj AS s JOIN PhotoObj AS p ON s.bestObjID = p.objID "
        "WHERE s.class = 'GALAXY' AND s.zWarning = 0 AND p.clean = 1 "
        "AND s.z BETWEEN 0.01 AND 1.0 "
        "AND p.modelMag_r BETWEEN 14 AND 21 "
        f"AND p.ra >= {ra_min} AND p.ra < {ra_max}"
    )


def _run_query(sql: str, timeout: int = 120, retries: int = 4) -> pd.DataFrame:
    """Send one SQL query, with retries and polite backoff."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(
                SKYSERVER_URL,
                params={"cmd": sql, "format": "csv", "searchtool": "SQL"},
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.text
            # SkyServer prefixes results with a '#Table1' line. An error instead
            # comes back as plain text starting with 'ERROR' or as HTML.
            if text.lstrip().lower().startswith(("error", "<")):
                raise RuntimeError(f"SkyServer returned an error:\n{text[:500]}")
            # Drop any leading comment lines before the CSV header.
            lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
            return pd.read_csv(io.StringIO("\n".join(lines)))
        except Exception as exc:  # noqa: BLE001 - we genuinely want to retry anything
            last_err = exc
            wait = 5 * (attempt + 1)
            print(f"  query attempt {attempt + 1}/{retries} failed "
                  f"({type(exc).__name__}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"All {retries} query attempts failed") from last_err


def fetch_sdss(
    n_total: int = 60_000,
    n_chunks: int = 12,
    out_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Download a photo-z training set from SDSS DR17.

    We split the sky into `n_chunks` slices in Right Ascension (celestial
    longitude) and query each separately. Three reasons:

      1. One enormous query will time out; many small ones will not.
      2. It is reproducible -- the slice boundaries are fixed, so re-running
         this next year returns the same galaxies.
      3. It is RESUMABLE. Each chunk is written to `cache_dir` as soon as it
         arrives, and an existing chunk file is skipped on a later run.

    That third point is not a nicety. SkyServer is a free public service that
    regularly times out mid-download; without per-chunk caching a failure at
    chunk 11 of 12 throws away everything. Re-run this function after a failure
    and it picks up where it stopped.

    Note the slices will NOT return equal numbers of galaxies, because SDSS did
    not survey the whole sky uniformly -- it covers roughly a third of it,
    mostly the northern galactic cap. Some RA slices are nearly empty. That is
    a real feature of the survey, not a bug in the query.
    """
    per_chunk = int(np.ceil(n_total / n_chunks))
    edges = np.linspace(0, 360, n_chunks + 1)

    if cache_dir is None and out_path is not None:
        cache_dir = Path(out_path).parent / "chunks"
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i in range(n_chunks):
        label = f"[{i + 1}/{n_chunks}] RA {edges[i]:.0f}-{edges[i + 1]:.0f} deg"
        chunk_path = cache_dir / f"chunk_{i:02d}.csv" if cache_dir else None

        if chunk_path and chunk_path.exists():
            df = pd.read_csv(chunk_path)
            print(f"{label} ... cached ({len(df)} galaxies)", flush=True)
            frames.append(df)
            continue

        print(f"{label} ...", flush=True)
        try:
            df = _run_query(_build_query(edges[i], edges[i + 1], per_chunk))
        except RuntimeError as exc:
            # Don't lose the chunks we already have because one slice failed.
            print(f"      !! chunk failed permanently: {exc}")
            print("      !! continuing; re-run to retry just this chunk")
            continue

        print(f"      -> {len(df)} galaxies")
        if chunk_path:
            df.to_csv(chunk_path, index=False)
        frames.append(df)
        time.sleep(1)  # be gentle with a free public archive

    if not frames:
        raise RuntimeError("Every chunk failed; no data downloaded")

    out = pd.concat(frames, ignore_index=True)
    print(f"\nTotal downloaded: {len(out)} galaxies from {len(frames)}/{n_chunks} slices")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
    return out


def make_synthetic(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """A fake catalogue, for tests and for running with no internet.

    THIS IS NOT REAL DATA and no scientific claim may be made from it. It exists
    so the pipeline and the test suite can run offline.

    It is built to have the three features that make real photo-z hard:
      1. colours that vary smoothly with redshift (the learnable signal)
      2. noise that grows for fainter galaxies (heteroscedastic)
      3. a population of "catastrophic outliers" whose colours mimic a galaxy at
         a completely different redshift -- the thing that breaks naive models
    """
    rng = np.random.default_rng(seed)

    z = rng.beta(2.0, 5.0, size=n) * 0.9 + 0.02           # skewed toward low z
    r_mag = 17.0 + 4.0 * rng.beta(2.0, 2.0, size=n)       # 17..21
    noise_scale = 0.02 + 0.25 * (r_mag - 17.0) / 4.0      # fainter => noisier

    # Smooth, monotonic-ish colour-redshift relations, loosely imitating how a
    # galaxy's spectral break shifts through the filters as it recedes.
    colours = np.column_stack([
        1.00 + 2.2 * z + 0.9 * z**2,   # u-g
        0.45 + 2.6 * z - 0.6 * z**2,   # g-r
        0.30 + 1.1 * z,                # r-i
        0.18 + 0.7 * z,                # i-z
    ])
    colours += rng.normal(0.0, noise_scale[:, None], size=colours.shape)

    # ~7% catastrophic outliers: colours consistent with a different redshift.
    n_out = int(0.07 * n)
    idx = rng.choice(n, size=n_out, replace=False)
    z_fake = rng.uniform(0.02, 0.9, size=n_out)
    colours[idx] = np.column_stack([
        1.00 + 2.2 * z_fake + 0.9 * z_fake**2,
        0.45 + 2.6 * z_fake - 0.6 * z_fake**2,
        0.30 + 1.1 * z_fake,
        0.18 + 0.7 * z_fake,
    ]) + rng.normal(0.0, 0.05, size=(n_out, 4))

    # Convert colours back into magnitudes, since that is what a catalogue holds.
    mags = {
        "modelMag_r": r_mag,
        "modelMag_i": r_mag - colours[:, 2],
    }
    mags["modelMag_z"] = mags["modelMag_i"] - colours[:, 3]
    mags["modelMag_g"] = r_mag + colours[:, 1]
    mags["modelMag_u"] = mags["modelMag_g"] + colours[:, 0]

    df = pd.DataFrame({"z_spec": z, "z_spec_err": np.full(n, 1e-4), **mags})
    for b in BANDS:
        df[f"modelMagErr_{b}"] = noise_scale
        df[f"extinction_{b}"] = 0.0   # no galactic dust in the fake universe
    df["ra"] = rng.uniform(0, 360, n)
    df["dec"] = rng.uniform(-10, 60, n)
    return df


def load(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)
