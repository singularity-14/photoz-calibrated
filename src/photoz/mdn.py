"""Mixture Density Network: a neural network that outputs a DISTRIBUTION.

THE PROBLEM WITH AN ORDINARY NETWORK
------------------------------------
A standard regression network maps colours -> one number, trained with mean
squared error. Two things go wrong for photo-z:

  1. It cannot express uncertainty at all. It says "0.43" with no hint whether
     that is solid or a guess.
  2. When a galaxy's colours are genuinely consistent with EITHER z = 0.2 or
     z = 0.7 (this really happens -- different spectral features can shift into
     the same filters), MSE training drives the network toward the average,
     ~0.45, which is the one answer that is definitely wrong.

THE FIX
-------
Have the network output the parameters of a mixture of K Gaussians instead:
K weights, K means, K widths. Then:

  - it can be narrow where it is confident and wide where it is not
  - it can be genuinely two-peaked for the ambiguous case above, putting mass
    at 0.2 AND 0.7 and none in between

TRAINING
--------
Not MSE. We maximise the likelihood the model assigns to the true redshift,
i.e. minimise

    -log( sum_k w_k * Normal(z_true | mu_k, sigma_k) )

This loss punishes overconfidence automatically: claiming a narrow PDF that
misses the truth produces a very large negative log-likelihood. The model
learns honest widths because dishonest ones are expensive.

Bishop (1994) introduced these; they remain the simplest good answer to
"I want a neural network, but I need a real PDF out of it."
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

LOG_2PI = float(np.log(2.0 * np.pi))


class MixtureDensityNetwork(nn.Module):
    def __init__(
        self, n_features: int, n_components: int = 5, hidden: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_components = n_components

        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)

        # Three heads: mixture weights, means, and (log) widths.
        self.head_logits = nn.Linear(prev, n_components)
        self.head_mu = nn.Linear(prev, n_components)
        self.head_log_sigma = nn.Linear(prev, n_components)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        log_w = torch.log_softmax(self.head_logits(h), dim=-1)  # weights sum to 1
        mu = self.head_mu(h)
        # Clamp the width: too small and the loss explodes (a delta function on
        # a training point gives infinite likelihood -- a real failure mode of
        # MDNs); too large and the model just hedges.
        log_sigma = torch.clamp(self.head_log_sigma(h), min=-7.0, max=2.0)
        return log_w, mu, log_sigma

    def nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of the true redshifts under the mixture."""
        log_w, mu, log_sigma = self(x)
        y = y[:, None]
        # log of a Gaussian density, per component
        log_prob = -0.5 * (((y - mu) / log_sigma.exp()) ** 2 + LOG_2PI) - log_sigma
        # logsumexp over components, done stably
        return -torch.logsumexp(log_w + log_prob, dim=-1).mean()

    @torch.no_grad()
    def predict_pdf(self, X: np.ndarray, zgrid: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Evaluate the mixture density on a grid of redshifts."""
        self.eval()
        grid = torch.as_tensor(zgrid, dtype=torch.float32)
        out = np.empty((len(X), len(zgrid)), dtype=np.float64)

        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32)
            log_w, mu, log_sigma = self(xb)
            # shapes -> (batch, components, grid)
            z = grid[None, None, :]
            mu_ = mu[:, :, None]
            ls_ = log_sigma[:, :, None]
            log_prob = -0.5 * (((z - mu_) / ls_.exp()) ** 2 + LOG_2PI) - ls_
            dens = torch.logsumexp(log_w[:, :, None] + log_prob, dim=1).exp()
            out[start:start + batch_size] = dens.numpy()

        return out


def train_mdn(
    X_train: np.ndarray, z_train: np.ndarray,
    X_val: np.ndarray, z_val: np.ndarray,
    n_components: int = 5,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 20,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[MixtureDensityNetwork, dict]:
    """Train with early stopping on validation NLL.

    Early stopping on the VALIDATION likelihood -- not on training loss, and not
    on a point-estimate error -- is what keeps the uncertainties honest. If you
    early-stop on something like RMSE you are optimising sharpness and will
    happily ship an overconfident model.
    """
    torch.manual_seed(seed)

    model = MixtureDensityNetwork(X_train.shape[1], n_components=n_components)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)

    Xtr = torch.as_tensor(X_train, dtype=torch.float32)
    ytr = torch.as_tensor(z_train, dtype=torch.float32)
    Xva = torch.as_tensor(X_val, dtype=torch.float32)
    yva = torch.as_tensor(z_val, dtype=torch.float32)

    best_val, best_state, bad_epochs = np.inf, None, 0
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        running = 0.0
        for start in range(0, len(Xtr), batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = model.nll(Xtr[idx], ytr[idx])
            loss.backward()
            # MDN gradients can spike when a component collapses; clip them.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            running += loss.item() * len(idx)
        train_nll = running / len(Xtr)

        model.eval()
        with torch.no_grad():
            val_nll = model.nll(Xva, yva).item()
        sched.step(val_nll)

        history["train"].append(train_nll)
        history["val"].append(val_nll)

        if val_nll < best_val - 1e-4:
            best_val, bad_epochs = val_nll, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if verbose and (epoch % 10 == 0 or bad_epochs >= patience):
            print(f"  epoch {epoch:3d}  train NLL {train_nll:+.4f}  val NLL {val_nll:+.4f}")

        if bad_epochs >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best val NLL {best_val:+.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
