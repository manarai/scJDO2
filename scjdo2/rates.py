"""Steps 3 and 4 — NB gene rate model on fixed latent positions.

Step 3 (``GeneRateModel``). Shared smooth trunk (SiLU, weight decay) with
gene-specific linear heads produces the NB linear predictor::

    η_g(z_c) = head_g( trunk(z_c) )
    μ_cg     = s_c · exp( η_g(z_c) )
    log μ_cg = log s_c + η_g(z_c)

Size factors ``s_c`` enter as a **log-offset**, never a division. Dispersion
``θ_g`` is a per-gene learned free parameter.

Latent positions ``z`` are **fixed** (Step 1); the rate model never updates
them. Fitting minimises the NB negative log-likelihood on raw counts, with
a 10% held-out cell split for early stopping and model selection — the
same split is reused by Step 5's ridge-λ grid search.

Step 4 (``rate_along``, ``grad_eta``). The chain rule in η-coordinates::

    dη_g/dt = ∇_z η_g(z_c) · f(z_c)        [autodiff]
    dμ_g/dt = μ_g · dη_g/dt

J is estimated in η-coordinates in Step 5 — η is NB's natural linear
predictor so Fisher weights (``μθ/(μ+θ)``) condition the ridge normal
equations well. Eigenvalues of the η-space and rate-space Jacobians agree
at fixed points; loadings differ by a diagonal similarity.

The decoder is differentiable w.r.t. ``z``; the plotted drift and the
classified Jacobian are the same function (V2_PLAN R12 invariant).

See ``docs/opspace_figure_spec.md`` §8.1–§8.2 for the full design lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # pragma: no cover — torch is required at .fit()/.eta()/.mu() time
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class RatesConfig:
    hidden: tuple[int, ...] = (128, 128)
    n_epochs: int = 400
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    held_out_frac: float = 0.10
    patience: int = 20                   # held-out NLL early-stop patience
    min_dispersion: float = 1e-3         # θ_g floor for numerical stability
    max_dispersion: float = 1e4          # θ_g ceiling
    seed: int = 0
    device: Optional[str] = None


# ----------------------------------------------------------------- torch nn

class _NBDecoder(nn.Module if nn is not None else object):
    """Shared SiLU trunk + gene-specific linear heads producing η_g(z).

    Dispersion ``θ_g`` is a free parameter per gene, parameterised through
    ``log_theta`` so it stays positive without a clamp inside the backward.
    """

    def __init__(self, d_in: int, n_genes: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.Linear(prev, n_genes)      # one linear head per gene
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def eta(self, z: "torch.Tensor") -> "torch.Tensor":
        return self.heads(self.trunk(z))

    def forward(self, z: "torch.Tensor") -> "torch.Tensor":
        return self.eta(z)


def _nb_log_prob(
    counts: "torch.Tensor",      # (B, G) non-negative, int-ish
    log_mu: "torch.Tensor",      # (B, G)
    log_theta: "torch.Tensor",   # (G,)
) -> "torch.Tensor":
    """Negative-binomial log-likelihood in log-mean parameterisation.

    θ is the inverse-dispersion so ``Var(X) = μ + μ²/θ``. The log-pmf is
    the standard form; we drop the ``lgamma(counts+1)`` term since it is
    constant in the parameters and the trainer only needs gradients.
    """
    theta = torch.exp(log_theta).clamp(min=1e-6, max=1e8)   # (G,)
    mu = torch.exp(log_mu)                                   # (B, G)
    log_theta_full = log_theta.expand_as(log_mu)
    # Standard NB log-pmf (minus the counts!-term which drops in grads)
    log_prob = (
        torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        + theta * (log_theta_full - torch.log(theta + mu))
        + counts * (log_mu - torch.log(theta + mu))
    )
    return log_prob


# ------------------------------------------------------------------- model

class GeneRateModel:
    """NB rate model with autodiff ``∂η/∂z``.

    After :meth:`fit`:

    * :meth:`eta(z)` returns the log-rate ``η_g(z_c)``, shape ``(n, G)``.
    * :meth:`mu(z, size_factor)` returns the mean ``μ_cg``.
    * :meth:`grad_eta(z)` returns ``∂η_g/∂z`` of shape ``(n, G, d)`` by autograd.
    * :meth:`rate_along(z, f)` returns ``dμ/dt`` of shape ``(n, G)``.
    * :meth:`theta` returns the per-gene inverse-dispersion.
    * :meth:`held_out_nll(counts, z, size_factor)` returns the held-out NLL
      used to select the ridge λ in Step 5.
    """

    def __init__(self, config: Optional[RatesConfig] = None, **overrides) -> None:
        self.config = config or RatesConfig()
        for k, v in overrides.items():
            if not hasattr(self.config, k):
                raise AttributeError(f"RatesConfig has no field {k!r}")
            setattr(self.config, k, v)
        self._model: Optional[_NBDecoder] = None
        self._device: Optional[str] = None
        self._n_latent: Optional[int] = None
        self._n_genes: Optional[int] = None
        self._train_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None
        self._best_val_nll: Optional[float] = None
        self._fitted = False

    # -------------------------------------------------------------- fit
    def fit(
        self,
        counts: np.ndarray,          # (n_cells, n_genes), raw integer counts
        z: np.ndarray,               # (n_cells, n_latent), frozen positions
        size_factors: Optional[np.ndarray] = None,  # (n_cells,); default: lib/median
        verbose: bool = False,
    ) -> "GeneRateModel":
        if torch is None:
            raise RuntimeError("torch is required to fit GeneRateModel")
        counts = np.asarray(counts, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        if counts.ndim != 2 or z.ndim != 2 or counts.shape[0] != z.shape[0]:
            raise ValueError(f"counts {counts.shape} vs z {z.shape} mismatch")
        n, G = counts.shape
        d = z.shape[1]

        if size_factors is None:
            lib = counts.sum(axis=1).astype(np.float32)
            med = float(np.median(lib[lib > 0])) if np.any(lib > 0) else 1.0
            size_factors = (lib / med).astype(np.float32)
        s = np.asarray(size_factors, dtype=np.float32)
        log_s = np.log(np.clip(s, 1e-8, None)).astype(np.float32)

        rng = np.random.default_rng(int(self.config.seed))
        n_test = max(1, int(round(self.config.held_out_frac * n)))
        perm = rng.permutation(n)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        self._train_idx = train_idx
        self._test_idx = test_idx

        device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._device = str(device)

        torch.manual_seed(int(self.config.seed))
        model = _NBDecoder(d, G, tuple(self.config.hidden)).to(device)
        opt = torch.optim.AdamW(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay,
        )

        C = torch.as_tensor(counts, dtype=torch.float32, device=device)
        Z = torch.as_tensor(z, dtype=torch.float32, device=device)
        LS = torch.as_tensor(log_s, dtype=torch.float32, device=device)
        train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        test_t = torch.as_tensor(test_idx, dtype=torch.long, device=device)

        bs = int(self.config.batch_size)
        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        bad = 0

        for epoch in range(int(self.config.n_epochs)):
            model.train()
            idx = train_t[torch.randperm(train_t.size(0), device=device)]
            for s_i in range(0, idx.size(0), bs):
                b = idx[s_i: s_i + bs]
                opt.zero_grad(set_to_none=True)
                eta = model.eta(Z[b])                      # (B, G)
                log_mu = eta + LS[b].unsqueeze(1)         # offset
                lp = _nb_log_prob(C[b], log_mu, model.log_theta)
                loss = -lp.mean()
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                eta_v = model.eta(Z[test_t])
                log_mu_v = eta_v + LS[test_t].unsqueeze(1)
                val = -_nb_log_prob(C[test_t], log_mu_v, model.log_theta).mean().item()
            if verbose and (epoch % max(1, self.config.n_epochs // 20) == 0):
                print(f"  [rates] epoch {epoch:4d}  val NLL {val:.4f}")
            if val < best_val - 1e-5:
                best_val = val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.config.patience:
                    if verbose:
                        print(f"  [rates] early-stop at epoch {epoch}, val NLL {best_val:.4f}")
                    break

        model.load_state_dict(best_state)
        self._model = model
        self._best_val_nll = float(best_val)
        self._n_latent = d
        self._n_genes = G
        self._fitted = True
        return self

    # --------------------------------------------------------- introspect
    def _require_fit(self) -> None:
        if not self._fitted or self._model is None:
            raise RuntimeError("GeneRateModel.fit() must be called first")

    @property
    def best_val_nll(self) -> float:
        self._require_fit()
        return float(self._best_val_nll or float("nan"))

    @property
    def test_idx(self) -> np.ndarray:
        self._require_fit()
        return np.asarray(self._test_idx, dtype=np.int64)

    @property
    def train_idx(self) -> np.ndarray:
        self._require_fit()
        return np.asarray(self._train_idx, dtype=np.int64)

    def theta(self) -> np.ndarray:
        self._require_fit()
        return torch.exp(self._model.log_theta).detach().cpu().numpy()

    # ----------------------------------------------------------- forward
    def eta(self, z: np.ndarray) -> np.ndarray:
        """Return ``η_g(z_c)`` of shape ``(n, G)``."""
        self._require_fit()
        Z = torch.as_tensor(z, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            return self._model.eta(Z).cpu().numpy()

    def mu(self, z: np.ndarray, size_factor: Optional[np.ndarray] = None) -> np.ndarray:
        """Return ``μ_cg = s_c · exp(η_g(z_c))`` of shape ``(n, G)``.

        ``size_factor`` defaults to 1 for every cell (useful when
        comparing geometries); callers that want the fitted dataset's
        library should pass the lib/median vector used in :meth:`fit`.
        """
        self._require_fit()
        Z = torch.as_tensor(z, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            eta = self._model.eta(Z)
            if size_factor is None:
                log_s = torch.zeros(Z.size(0), 1, device=self._device)
            else:
                s = np.asarray(size_factor, dtype=np.float32)
                log_s = torch.as_tensor(
                    np.log(np.clip(s, 1e-8, None)), dtype=torch.float32,
                    device=self._device,
                ).unsqueeze(1)
            return torch.exp(eta + log_s).cpu().numpy()

    # ---------------------------------------------------- autograd grad_η
    def grad_eta(
        self, z: np.ndarray, chunk: int = 256, dtype: str = "float32",
    ) -> np.ndarray:
        """Return ``∂η_g/∂z_c`` of shape ``(n, G, d)`` by autograd.

        Chain-rule factor for Step 4 / Step 5. Uses forward-mode AD
        (``torch.func.jacfwd``) when the latent dim ``d`` is smaller
        than the gene dim ``G`` — which is the usual case
        (``d ≈ 50 ≪ G ≈ 2000``) and ~``G/d`` times faster than reverse
        mode. Falls back to ``jacrev`` when ``d ≥ G``.
        Chunked into cell batches of ``chunk`` so peak memory stays
        below ``chunk · G · d · sizeof(dtype)``; ``dtype="float32"``
        halves the final tensor's RAM footprint for free.
        """
        self._require_fit()
        import torch
        try:
            from torch.func import jacfwd, jacrev, vmap
        except ImportError:  # fallback for older torch
            from functorch import jacfwd, jacrev, vmap  # type: ignore[no-redef]
        Z = torch.as_tensor(z, dtype=torch.float32, device=self._device)
        model = self._model
        np_dtype = np.float32 if dtype == "float32" else np.float64

        def _eta_single(z_row: "torch.Tensor") -> "torch.Tensor":
            return model.eta(z_row.unsqueeze(0)).squeeze(0)

        n, d = Z.shape
        G = int(self._n_genes or model.eta(Z[:1]).shape[1])
        jac = jacfwd(_eta_single) if d < G else jacrev(_eta_single)
        out = np.empty((n, G, d), dtype=np_dtype)
        chunk = max(1, int(chunk))
        with torch.no_grad():
            for s in range(0, n, chunk):
                e = min(n, s + chunk)
                Jc = vmap(jac)(Z[s:e])                            # (chunk, G, d)
                out[s:e] = Jc.cpu().numpy().astype(np_dtype, copy=False)
                del Jc
        return out

    def rate_along_eta(
        self, z: np.ndarray, f: np.ndarray, chunk: int = 256,
    ) -> np.ndarray:
        """η-coord drift: ``dη_g/dt = ∇_z η_g(z_c) · f(z_c)``.

        Returns an ``(n, G)`` array. Uses a direct Jacobian-vector
        product (``torch.func.jvp``) so the full ``(n, G, d)`` tensor
        is never materialised — one forward pass per cell gives
        ``∇η(z_c) · f(z_c)`` directly, which on real datasets
        (G ≈ 2000, n ≈ 3000, d ≈ 50) is dramatically faster than
        reverse-mode on grad_eta followed by einsum.
        """
        self._require_fit()
        f = np.asarray(f, dtype=np.float32)
        if f.shape != z.shape:
            raise ValueError(f"f {f.shape} must match z {z.shape}")
        import torch
        try:
            from torch.func import jvp, vmap
        except ImportError:
            from functorch import jvp, vmap  # type: ignore[no-redef]
        Z = torch.as_tensor(z, dtype=torch.float32, device=self._device)
        F = torch.as_tensor(f, dtype=torch.float32, device=self._device)
        model = self._model

        def _eta_single(z_row: "torch.Tensor") -> "torch.Tensor":
            return model.eta(z_row.unsqueeze(0)).squeeze(0)

        n = Z.shape[0]
        G = int(self._n_genes or model.eta(Z[:1]).shape[1])
        out = np.empty((n, G), dtype=np.float64)
        chunk = max(1, int(chunk))
        with torch.no_grad():
            for s in range(0, n, chunk):
                e = min(n, s + chunk)
                z_b, f_b = Z[s:e], F[s:e]
                # vmap jvp across cells: jvp(_eta_single, (z,), (v,)) → (η, ∇η·v).
                _, jvp_out = vmap(lambda z, v: jvp(_eta_single, (z,), (v,)))(z_b, f_b)
                out[s:e] = jvp_out.cpu().numpy().astype(np.float64, copy=False)
                del jvp_out
        return out

    def rate_along(self, z: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Chain rule: ``dμ_g/dt = μ_g · (∂η_g/∂z · f)``.

        Returns an ``(n, G)`` array in the same units as ``μ_g``. Call
        :meth:`mu` with the same ``size_factor`` the drift was fit on
        to keep units consistent across the pipeline.
        """
        d_eta_dt = self.rate_along_eta(z, f)              # (n, G)
        mu = self.mu(z)                                   # (n, G) at size_factor=1
        return mu * d_eta_dt

    def nb_weights(self, z: np.ndarray, size_factor: Optional[np.ndarray] = None) -> np.ndarray:
        """Per-cell per-gene NB precision of η: ``w_cg = μ_cg · θ_g / (μ_cg + θ_g)``.

        This is the weight used by Step 5's per-gene-row weighted ridge
        regression (docs/opspace_figure_spec.md §8.3).
        """
        self._require_fit()
        mu = self.mu(z, size_factor=size_factor)          # (n, G)
        theta = self.theta()                              # (G,)
        return (mu * theta / (mu + theta + 1e-12)).astype(np.float64)

    def held_out_nll(
        self,
        counts: np.ndarray,
        z: np.ndarray,
        size_factor: Optional[np.ndarray] = None,
    ) -> float:
        """NB NLL on the held-out cells recorded during :meth:`fit`.

        Reused by Step 5 to score candidate ridge λs on the same split.
        """
        self._require_fit()
        import torch
        if self._test_idx is None:
            raise RuntimeError("fit() did not record a held-out split")
        idx = self._test_idx
        C = torch.as_tensor(counts[idx], dtype=torch.float32, device=self._device)
        Z = torch.as_tensor(z[idx], dtype=torch.float32, device=self._device)
        if size_factor is None:
            log_s = torch.zeros(Z.size(0), device=self._device).unsqueeze(1)
        else:
            s = np.asarray(size_factor, dtype=np.float32)[idx]
            log_s = torch.as_tensor(
                np.log(np.clip(s, 1e-8, None)), dtype=torch.float32,
                device=self._device,
            ).unsqueeze(1)
        with torch.no_grad():
            log_mu = self._model.eta(Z) + log_s
            lp = _nb_log_prob(C, log_mu, self._model.log_theta)
        return float((-lp.mean()).item())
