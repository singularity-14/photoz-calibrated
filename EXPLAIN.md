# Understanding this project from zero

No astronomy assumed. If you can read a histogram, you can read this.

The goal is that by the end you can explain, to anyone who asks, *what* this
project does, *why* it is set up this way, and *what the result means*.

---

## 1. The physical problem

### How do you measure the distance to a galaxy?

You cannot put a tape measure on the universe. But there is a trick.

The universe is expanding. Every distant galaxy is moving away from us, and the
further away it is, the faster it recedes. When something that emits light
moves away from you, its light gets stretched to longer wavelengths — toward
the red end of the spectrum. This is called **redshift**, written `z`.

So redshift is a proxy for distance. Measure the redshift, and you know how far
away the galaxy is, and therefore how far back in time you are looking.

`z = 0.1` is relatively nearby. `z = 1.0` means the light has been travelling
for roughly 8 billion years. Redshift is the single most important number in
extragalactic astronomy — almost every other quantity depends on knowing it.

### The expensive way: spectroscopy

Spread a galaxy's light into a full rainbow and you see sharp lines at specific
wavelengths, produced by particular elements. We know the exact wavelengths
those lines have in a laboratory. Measure how far they have shifted, and you
get redshift to about four decimal places.

The catch: this takes a large telescope a long time for a *single galaxy*,
because you are splitting already-faint light into hundreds of wavelength bins.
It does not scale.

### The cheap way: photometry (what this project does)

Instead of a full rainbow, take a picture through five coloured filters:

```
   u          g          r          i          z
 ultraviolet  blue      green      red      near-infrared
```

Now you have just five numbers per galaxy instead of a full spectrum. You lose
nearly all the detail — but you can photograph *millions of galaxies at once*,
because every galaxy in the image gets measured simultaneously.

The question this project answers:

> **Can you recover the redshift from only those five numbers?**

This is called **photometric redshift**, or **photo-z**.

### Why it works at all

A galaxy's spectrum is not flat. It has broad features — in particular a sharp
step around 400 nm called the *4000 Å break*, caused by metals in stellar
atmospheres absorbing light.

As the galaxy recedes, that whole pattern slides toward redder wavelengths. So
the break moves through the filters:

```
nearby galaxy      break falls between u and g   ->  big u-g difference
distant galaxy     break has slid to between g and r  ->  big g-r difference
```

The *pattern of brightness across the five filters* therefore encodes the
redshift. Crudely: a galaxy's colour tells you roughly how far away it is.

### Why it is hard

Different combinations of redshift, galaxy age and dust can produce nearly
identical colours. A galaxy at `z = 0.2` can genuinely look like one at
`z = 0.7` through five broad filters. When that happens there is **no** single
right answer — the data is genuinely ambiguous.

This is exactly why the project is built around probability distributions
rather than single numbers. Hold that thought.

---

## 2. Why a single number is the wrong output

Suppose a model predicts `z = 0.5` for a galaxy whose true redshift is `0.6`.
Is that good?

**You cannot tell**, and that is the whole point.

- If the model meant "0.5, give or take 0.2" → it was right. The truth is
  comfortably inside its uncertainty.
- If the model meant "0.5, give or take 0.01" → it was *badly* wrong. It was
  certain, and it was certain of the wrong thing.

Same prediction, same error, opposite verdicts. The error alone does not tell
you whether the model worked.

### Why this matters more in science than in typical ML

In a product recommender, a confidently wrong answer costs you one bad
recommendation. In astronomy, photo-z estimates feed into measurements of how
matter is distributed across the universe, and **those downstream calculations
propagate your error bars**.

If you say ±0.01 when you should have said ±0.2, you do not just introduce an
error — you tell the next scientist that your number is trustworthy enough to
build on. The error silently corrupts their result too.

> An honest uncertainty is more valuable than a small one.

This is why the project outputs a full probability distribution, `p(z)`, for
every galaxy: a curve saying how likely each possible redshift is.

```
   confident galaxy              ambiguous galaxy
        ▐▌                         ▗▄▖      ▗▄▖
       ▐██▌                       ▐███▌    ▐███▌
      ▗█████▖                    ▗█████▖  ▗█████▖
   ───────────────            ──────────────────────
   0.3  0.4  0.5                0.2        0.7

   "it's 0.4, I'm sure"       "it's 0.2 or 0.7,
                               and I genuinely can't tell which"
```

That second curve is not a failure. It is the correct answer to a question that
has two plausible answers, and **no single number can express it.**

---

## 3. The three methods, and why each exists

### Method 1 — Polynomial fit (the 1995 baseline)

Fit a polynomial mapping colours to redshift. Compute the typical error on the
training set. Report that same error for every galaxy.

**Why include something this crude?** Because it is the honest comparison
point. If an elaborate neural network cannot beat a polynomial from 1995, that
is important to know — and the only way to know it is to build the polynomial.

It also illustrates a specific failure beautifully: it gives **every galaxy the
same uncertainty**, whether bright and obvious or faint and ambiguous. The
calibration plots show exactly what that costs.

### Method 2 — Nearest neighbours

For a galaxy whose redshift you want: find the 100 most similar galaxies (in
colour) that *do* have measured spectroscopic redshifts. Look at how their
redshifts are distributed. That distribution is your answer.

This is simple but genuinely clever, because **the uncertainty emerges from the
data instead of being assumed**:

- neighbours all agree → narrow `p(z)` → confident
- neighbours disagree → wide `p(z)` → uncertain
- neighbours split into two groups → **two-peaked** `p(z)`

That last case is the ambiguity from §1 being reported correctly. The
polynomial method is structurally incapable of expressing it.

### Method 3 — Mixture Density Network (the ML method)

A neural network, but one that outputs a *distribution* instead of a number.
Specifically it outputs the ingredients of a mixture of 5 Gaussians: 5 heights,
5 centres, 5 widths. Add those bell curves together and you get `p(z)`.

Because it can place several bumps, it can represent "0.2 or 0.7" directly.

**It is not trained with mean squared error.** It is trained to maximise the
probability it assigns to the true redshift. This matters: that loss punishes
overconfidence automatically. Claim a narrow peak in the wrong place and the
loss blows up. The network learns honest widths because dishonest widths are
expensive.

---

## 4. How we judge the answers

This is the part worth understanding properly, because it is what separates
this project from an ordinary ML exercise.

### PIT — the calibration test

**The single most important plot in this project.**

For each galaxy, ask: *where does the true redshift fall within my predicted
distribution?* If the truth sits at the 30th percentile of the predicted
curve, record 0.30. That number is the PIT value.

Now the key insight. If the model is honest, the truth should land at every
percentile equally often — sometimes low, sometimes high, no preference. So
across thousands of galaxies, **a histogram of PIT values should be FLAT.**

And when it is not flat, the *shape* tells you the failure mode:

```
   FLAT               U-SHAPED              HUMPED              SLOPED
 ▐▌▐▌▐▌▐▌▐▌         ▐█▌      ▐█▌            ▗▄█▄▖            ▗▄▟█
 ▐▌▐▌▐▌▐▌▐▌         ▐█▌      ▐█▌          ▗▟█████▙▖        ▗▄▟████
 ──────────         ───────────────      ──────────────    ──────────
  honest            OVERCONFIDENT        UNDERCONFIDENT      BIASED
                   (PDFs too narrow —    (PDFs too wide,    (systematically
                    truth keeps landing   hedging too much)  too high or low)
                    in the tails)
```

Overconfidence is the common failure, and the dangerous one.

### Coverage — the same idea, in plainer words

> "When my model says it is 90% confident, is it actually right 90% of the time?"

Plot stated confidence against how often the truth really falls inside. Perfect
calibration is the diagonal line. Below the line = overconfident. This is the
version to use when explaining to someone who has never seen a PIT histogram.

### CRPS — scoring the whole distribution

A single number for "how good is this PDF?", which rewards being **both** sharp
**and** correct. It cannot be gamed:

- very wide PDF → honest but useless → scores badly
- very narrow PDF in the wrong place → scores badly
- narrow PDF in the right place → scores well

Lower is better. Nice property: if your PDF is a spike, CRPS becomes ordinary
absolute error — so it is a strict generalisation of the usual metric.

### sigma_NMAD — the headline number in photo-z papers

A robust measure of scatter. Two details that signal domain familiarity:

1. Errors are measured as `(z_pred − z_true) / (1 + z_true)`, not raw
   difference. Redshift uncertainty naturally scales with `(1+z)`, so dividing
   makes the number comparable across near and distant galaxies.
2. It uses the *median* absolute deviation, so a handful of catastrophic
   failures cannot inflate it. Those get reported separately as the **outlier
   fraction** (`|Δz| > 0.15`), because they are a different kind of problem:
   not noise, but a galaxy placed at the wrong distance entirely.

---

## 5. The experiment that matters most

Everything above uses a random train/test split. **That flatters the model, and
real deployment does not work that way.**

Think about where the training labels come from. Spectroscopic redshifts exist
only for galaxies someone pointed a big telescope at — which means the
**bright, nearby** ones, because faint galaxies are expensive.

But you want to apply the model to the galaxies you *don't* have spectra for:
the **faint, distant** ones.

So your training set and your target are systematically different populations.
A random split hides this completely, because it puts bright and faint galaxies
on both sides.

So this project runs the honest version too:

> **train on the brightest 50%, then test on the faintest 25%.**

Here is what that costs, on real SDSS galaxies:

| | random split | bright → faint |
|---|---|---|
| scatter (sigma_NMAD) | 0.0180 | **0.1066** — 5.9× worse |
| catastrophic outliers | 0.80% | **28.3%** — 1 galaxy in 4 |
| bias | +0.0002 | **−0.0911** |
| calibration (PIT KS) | 0.029 | **0.492** |

This is a dramatic collapse, and every part of it is interpretable:

- **The bias turns strongly negative.** The model places faint galaxies
  *closer* than they really are. Of course it does — it never saw distant
  galaxies during training, so it has no reason to predict large redshifts.
  It is extrapolating, and extrapolation pulls toward what it knows.
- **Outliers explode to 28%.** More than a quarter of faint galaxies are
  assigned the wrong distance entirely.
- **Calibration collapses.** PIT KS of 0.49 means the error bars have stopped
  carrying information. This is the worst part: the model is not just wrong,
  it is wrong *without warning you*.

A model that looks excellent on a random split is nearly unusable on the
population it was actually built for. In the photo-z literature this is a known
hard problem, not a detail — it is one of the main reasons photo-z remains an
open research area rather than a solved one.

---

## 6. What to say when someone asks

**"What did you build?"**
> A model that estimates how far away a galaxy is from a five-colour photograph,
> instead of from an expensive spectrum. The point isn't the estimate — it's
> that it reports honest uncertainty, including saying "this could be two very
> different distances" when the data genuinely can't distinguish them.

**"Why not just predict a number?"**
> Because a confidently wrong answer and an appropriately uncertain one look
> identical if all you report is the number. In astronomy those error bars get
> propagated into other people's results, so a wrong error bar corrupts work
> downstream of yours.

**"How do you know the uncertainties are honest?"**
> A PIT histogram. For each galaxy I check where the true answer fell inside my
> predicted distribution. If I'm honest, the truth should land at every
> percentile equally often, so the histogram should be flat. If it's U-shaped,
> I'm overconfident.

**"Did the neural network beat the simple methods?"**
> Yes, but the interesting part is *where*. It's 5.7× better than the 1995
> polynomial at calibration, and only 1.3× better at raw scatter. The old
> baselines are decent at predicting redshift — they're bad at knowing how
> wrong they are. That's exactly why I built the baselines instead of assuming
> the neural network would win.

**"What's the weakness?"**
> The training data is biased toward bright nearby galaxies, which is the
> population you least need predictions for. I measured what that costs:
> training on bright galaxies and testing on faint ones makes the scatter 5.9×
> worse and sends catastrophic outliers from under 1% to 28%. The calibration
> collapses completely. That's the honest number, and it's the one I'd quote.

**"So is the good number meaningless?"**
> Not meaningless, but it answers a question nobody asked. A random train/test
> split assumes you'll apply the model to the same kind of galaxy you trained
> on, and in this field you never do. Any photo-z result quoted without saying
> which split produced it should be treated carefully — including mine.

---

## 7. Where to go deeper

- **Ivezić et al., _Statistics, Data Mining and Machine Learning in Astronomy_**
  — the standard text; written for people arriving from statistics/ML.
- **Hogg, "Data analysis recipes: Fitting a model to data"** (arXiv) — short,
  opinionated, and will change how you think about fitting anything.
- **Salvato, Ilbert & Hoyle (2019), "The many flavours of photometric
  redshifts"** (Nature Astronomy) — a review of this exact problem.
- The next step for this codebase specifically is **template fitting** — the
  physics-based method that needs no training data at all. See "Where this goes
  next" in the README.
