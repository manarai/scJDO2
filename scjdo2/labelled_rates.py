"""Labelled rate model (§7c, V2_PLAN §9d).

Joint NB fit on new + total UMI counts with per-cell labelling pulse
length ``t_c`` entering the likelihood directly:

    E[total_cg]  = (α_g(z_c) / γ_g) · s_c   (steady-state total)
    E[new_cg]    = E[total_cg] · (1 − exp(−γ_g · t_c))

``α_g(z_c)`` is the per-gene transcription rate learned as a neural
head on top of the latent ``z``; ``γ_g`` is the per-gene degradation
rate shared across batches (one global γ per gene — the whole point
of §7c is to force γ to the same per-hour footing across 3h and 5h
batches, bypassing dynamo's per-batch γ fit).

The model exposes the same surface as :class:`scjdo2.rates.GeneRateModel`:
``eta_total(z)``, ``eta_new(z, t_c)``, ``gamma``, ``theta_new``,
``theta_total``, ``grad_eta(z)`` on the total view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class LabelledRatesConfig:
    hidden: Tuple[int, ...] = (128, 128)
    n_epochs: int = 400
    lr: float = 1e-3
    batch_size: int = 256
    held_out_frac: float = 0.10
    patience: int = 30
    seed: int = 0
    device: Optional[str] = None
    # Numerical bounds for γ (per hour): basically allow any
    # realistic cellular degradation rate.
    gamma_min: float = 1e-3
    gamma_max: float = 10.0


class _LabelledNBDecoder(nn.Module if nn is not None else object):
    """Shared trunk + ``α(z)`` head, plus per-gene scalar
    ``log_gamma`` and ``log_theta_{new,total}``."""

    def __init__(self, d_in: int, n_genes: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.alpha_head = nn.Linear(prev, n_genes)                 # log α(z)
        self.log_gamma = nn.Parameter(torch.zeros(n_genes))        # γ_g per gene
        self.log_theta_total = nn.Parameter(torch.zeros(n_genes))
        self.log_theta_new = nn.Parameter(torch.zeros(n_genes))

    def log_total(self, z: "torch.Tensor") -> "torch.Tensor":
        """``log E[total_cg] = log α(z) − log γ_g``."""
        return self.alpha_head(self.trunk(z)) - self.log_gamma.unsqueeze(0)

    def log_frac_new(self, t_c: "torch.Tensor",
                     clamp: Tuple[float, float] = (1e-3, 10.0)) -> "torch.Tensor":
        """``log(1 − exp(−γ · t_c))`` per cell per gene, numerically stable.

        ``t_c``: (B,) per-cell pulse length in hours.
        """
        gamma = torch.exp(self.log_gamma).clamp(min=clamp[0], max=clamp[1])
        a = gamma.unsqueeze(0) * t_c.unsqueeze(1)                  # (B, G)
        # log1p(-exp(-a)) via log1mexp for stability.
        # For small a: log1p(-exp(-a)) ≈ log(a) − a/2 − a²/24.
        small = a < 1e-3
        safe_a = torch.where(small, torch.full_like(a, 1e-3), a)
        log_series = torch.log(safe_a) - safe_a / 2.0
        log_main = torch.log1p(-torch.exp(-safe_a.clamp(min=1e-6)))
        return torch.where(small, log_series, log_main)

    def log_new(self, z: "torch.Tensor",
                t_c: "torch.Tensor") -> "torch.Tensor":
        return self.log_total(z) + self.log_frac_new(t_c)


def _nb_log_prob(counts, log_mu, log_theta):
    theta = torch.exp(log_theta).clamp(min=1e-6, max=1e8)
    mu = torch.exp(log_mu)
    log_theta_full = log_theta.expand_as(log_mu)
    return (
        torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        + theta * (log_theta_full - torch.log(theta + mu))
        + counts * (log_mu - torch.log(theta + mu))
    )


class LabelledRateModel:
    """NB rate model on (new, total) counts with per-cell pulse ``t_c``.

    After :meth:`fit`:

    * :meth:`eta_total(z)` — ``log(α/γ)`` per cell per gene.
    * :meth:`eta_new(z, t_c)`` — ``log E[new_cg]`` per cell per gene.
    * :meth:`gamma` — per-gene γ_g (hour⁻¹).
    * :meth:`held_out_nll` — joint NLL on the recorded test split.
    """

    def __init__(self, config: Optional[LabelledRatesConfig] = None,
                 **overrides) -> None:
        self.config = config or LabelledRatesConfig()
        for k, v in overrides.items():
            if not hasattr(self.config, k):
                raise AttributeError(f"LabelledRatesConfig has no field {k!r}")
            setattr(self.config, k, v)
        self._model: Optional[_LabelledNBDecoder] = None
        self._device: Optional[str] = None
        self._n_latent: Optional[int] = None
        self._n_genes: Optional[int] = None
        self._best_val_nll: Optional[float] = None
        self._fitted = False

    def fit(self, counts_new: np.ndarray, counts_total: np.ndarray,
            z: np.ndarray, t_c: np.ndarray,
            size_factor_total: Optional[np.ndarray] = None,
            verbose: bool = False) -> "LabelledRateModel":
        if torch is None:
            raise RuntimeError("torch is required to fit LabelledRateModel")
        counts_new = np.asarray(counts_new, dtype=np.float32)
        counts_total = np.asarray(counts_total, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        t_c = np.asarray(t_c, dtype=np.float32)
        n, G = counts_total.shape
        if counts_new.shape != (n, G) or z.shape[0] != n or t_c.shape != (n,):
            raise ValueError("shape mismatch across inputs")
        d_in = z.shape[1]
        rng = np.random.default_rng(self.config.seed)
        device = self.config.device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = _LabelledNBDecoder(d_in, G, self.config.hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(),
                                 lr=self.config.lr, weight_decay=1e-4)

        if size_factor_total is None:
            lib = counts_total.sum(axis=1)
            size_factor_total = lib / float(np.median(lib[lib > 0]))
        s = torch.as_tensor(size_factor_total, dtype=torch.float32, device=device)
        log_s = torch.log(s.clamp(min=1e-8)).unsqueeze(1)

        C_new = torch.as_tensor(counts_new, device=device)
        C_tot = torch.as_tensor(counts_total, device=device)
        Z = torch.as_tensor(z, device=device)
        T = torch.as_tensor(t_c, device=device)

        rng2 = np.random.default_rng(self.config.seed + 1)
        test_idx = rng2.choice(n, size=max(50, int(self.config.held_out_frac * n)),
                                replace=False)
        test_mask = np.zeros(n, dtype=bool); test_mask[test_idx] = True
        train_idx = np.where(~test_mask)[0]
        self._train_idx, self._test_idx = train_idx, test_idx

        bs = min(self.config.batch_size, len(train_idx))
        best_val = float("inf"); patience = 0; best_state = None
        for epoch in range(self.config.n_epochs):
            model.train()
            idx = torch.as_tensor(
                rng.choice(train_idx, size=bs, replace=False),
                dtype=torch.long, device=device,
            )
            log_total = model.log_total(Z[idx]) + log_s[idx]
            log_new = log_total + model.log_frac_new(T[idx])
            lp_tot = _nb_log_prob(C_tot[idx], log_total, model.log_theta_total)
            lp_new = _nb_log_prob(C_new[idx], log_new, model.log_theta_new)
            loss = -(lp_tot.mean() + lp_new.mean())
            opt.zero_grad(); loss.backward(); opt.step()
            if (epoch + 1) % max(1, self.config.n_epochs // 20) == 0:
                model.eval()
                with torch.no_grad():
                    log_total_v = model.log_total(Z[test_idx]) + log_s[test_idx]
                    log_new_v = log_total_v + model.log_frac_new(T[test_idx])
                    lp_tot_v = _nb_log_prob(C_tot[test_idx], log_total_v, model.log_theta_total).mean()
                    lp_new_v = _nb_log_prob(C_new[test_idx], log_new_v, model.log_theta_new).mean()
                    val = -(lp_tot_v + lp_new_v).item()
                if verbose and (epoch + 1) % max(1, self.config.n_epochs // 10) == 0:
                    print(f"[labelled-rate] epoch {epoch+1}/{self.config.n_epochs}  "
                          f"train_loss={float(loss.detach()):.3f}  val={val:.3f}")
                if val < best_val:
                    best_val = val; patience = 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= self.config.patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self._device = device
        self._n_latent = d_in
        self._n_genes = G
        self._best_val_nll = float(best_val)
        self._fitted = True
        return self

    # ---------------------------------------------------------- inference
    def _require_fit(self):
        if not self._fitted:
            raise RuntimeError("LabelledRateModel.fit() must be called first")

    @property
    def best_val_nll(self) -> float:
        self._require_fit()
        return float(self._best_val_nll or float("nan"))

    @property
    def gamma(self) -> np.ndarray:
        """Per-gene γ_g (hour⁻¹)."""
        self._require_fit()
        with torch.no_grad():
            g = torch.exp(self._model.log_gamma).clamp(
                min=self.config.gamma_min, max=self.config.gamma_max,
            ).detach().cpu().numpy()
        return g.astype(np.float64)

    def eta_total(self, z: np.ndarray) -> np.ndarray:
        self._require_fit()
        with torch.no_grad():
            Z = torch.as_tensor(z, dtype=torch.float32, device=self._device)
            return self._model.log_total(Z).detach().cpu().numpy()

    def eta_new(self, z: np.ndarray, t_c: np.ndarray) -> np.ndarray:
        self._require_fit()
        with torch.no_grad():
            Z = torch.as_tensor(z, dtype=torch.float32, device=self._device)
            T = torch.as_tensor(np.asarray(t_c, dtype=np.float32),
                                 device=self._device)
            return self._model.log_new(Z, T).detach().cpu().numpy()
