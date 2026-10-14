"""Energy-parameterised score drift (Phase D §5 rework).

Reparameterises the DSM drift as the gradient of a learned scalar
energy ``E_θ(z)``:

    s_θ(z) = −∇_z E_θ(z)      (score, by autodiff of a scalar)
    u(z)   = β · s_θ(z)         (drift, same scale as v1's score head)
    J(z)   = β · ∂s_θ/∂z = −β · ∂²E_θ/∂z∂z

Because ``s`` is the gradient of a scalar, ``J`` is **symmetric by
construction** (up to float precision — mixed partials commute). This
restores the theoretical property the user's PBMC-arm spec assumed:
only holder / drift / fork can be reached; imaginary-part eigenvalues
cannot arise from the drift itself.

Trained with the same multi-σ denoising score matching loss v1 uses,
weighted so each σ contributes a floor-constant loss of
``σ² · (s − target)² ≈ (σ·s + ε)²``.

No pseudotime conditioning: on a snapshot the DSM target is time-free.
This also matches the §5 PBMC setup (``t = 0`` everywhere).

The symmetry gate is cheap — on 100 random cells
``‖J − Jᵀ‖_F / ‖J‖_F`` should be at float32 noise (``< 1e-5``). The
run script calls :meth:`EnergyDriftAdapter.symmetry_check` and prints
the measured value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class EnergyDriftConfig:
    hidden: int = 96
    depth: int = 3                        # layers before the scalar readout
    beta: float = 0.1                     # drift scale (matches v1 β)
    n_epochs: int = 2000
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: Optional[str] = None
    # Multi-σ DSM schedule. (σ_min, σ_max) log-uniform per step; σ²
    # weighting makes the per-element loss floor equal 1.
    dsm_sigma_range: Tuple[float, float] = (0.05, 0.5)
    clock_calibrated: bool = False


class _EnergyMLP(nn.Module):
    """Scalar MLP ``z ↦ E(z)``. SiLU activations, no bias on the readout
    (constant shifts have zero gradient and only slow training)."""

    def __init__(self, dim: int, hidden: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = dim
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_d, hidden))
            layers.append(nn.SiLU())
            in_d = hidden
        layers.append(nn.Linear(in_d, 1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, z: "torch.Tensor") -> "torch.Tensor":          # (N, d) → (N,)
        return self.net(z).squeeze(-1)


class EnergyDriftAdapter:
    """Learned energy ``E_θ(z)`` with score / drift / Jacobian views.

    ``u = β · s = -β · ∇E``; ``J = -β · Hess(E)`` is symmetric by
    construction.
    """

    def __init__(self, dim: int, config: Optional[EnergyDriftConfig] = None,
                 **cfg_overrides) -> None:
        self.dim = int(dim)
        self.config = config or EnergyDriftConfig()
        for k, v in cfg_overrides.items():
            if not hasattr(self.config, k):
                raise AttributeError(f"EnergyDriftConfig has no field {k!r}")
            setattr(self.config, k, v)
        self._model: Optional[_EnergyMLP] = None
        self._device: Optional[str] = None
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, z: np.ndarray, verbose: bool = False) -> "EnergyDriftAdapter":
        if torch is None:
            raise ImportError("torch is required for EnergyDriftAdapter.fit")
        rng = np.random.default_rng(self.config.seed)
        torch.manual_seed(self.config.seed)

        device = self.config.device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        X_ref = torch.as_tensor(z, dtype=torch.float32, device=device)
        n = X_ref.shape[0]

        model = _EnergyMLP(self.dim, self.config.hidden, self.config.depth).to(device)
        opt = torch.optim.AdamW(model.parameters(),
                                lr=self.config.lr,
                                weight_decay=self.config.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, self.config.n_epochs),
        )

        log_lo = float(np.log(self.config.dsm_sigma_range[0]))
        log_hi = float(np.log(self.config.dsm_sigma_range[1]))
        bs = min(self.config.batch_size, n)

        model.train()
        for epoch in range(self.config.n_epochs):
            idx = torch.as_tensor(rng.integers(0, n, size=bs), dtype=torch.long,
                                  device=device)
            x_b = X_ref[idx]
            log_sig = torch.rand(bs, device=device) * (log_hi - log_lo) + log_lo
            sigmas = torch.exp(log_sig).unsqueeze(-1)             # (B, 1)
            noise = torch.randn_like(x_b)
            x_noisy = x_b + sigmas * noise                        # perturbed
            # s_θ(x_noisy) = -∇_x E(x_noisy). Enable grads through x.
            x_noisy = x_noisy.detach().requires_grad_(True)
            E = model(x_noisy)                                     # (B,)
            grad_x = torch.autograd.grad(
                outputs=E.sum(), inputs=x_noisy,
                create_graph=True, retain_graph=True,
            )[0]
            s = -grad_x                                            # (B, d)
            # DSM σ²-weighted loss: ((σ·s + ε)²).mean()
            diff = sigmas * s + noise
            loss = (diff * diff).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if verbose and (epoch + 1) % max(1, self.config.n_epochs // 10) == 0:
                print(f"[energy-drift] epoch {epoch + 1}/{self.config.n_epochs}  "
                      f"loss={float(loss.detach()):.4f}")

        model.eval()
        self._model = model
        self._device = device
        self._fitted = True
        return self

    # ---------------------------------------------------------- inference
    def _require_fit(self) -> None:
        if not self._fitted or self._model is None:
            raise RuntimeError("EnergyDriftAdapter.fit() must be called first")

    def energy(self, z: np.ndarray) -> np.ndarray:
        self._require_fit()
        with torch.no_grad():
            z_t = torch.as_tensor(z, dtype=torch.float32, device=self._device)
            return self._model(z_t).detach().cpu().numpy()

    def score(self, z: np.ndarray) -> np.ndarray:
        """``s = -∇_z E_θ(z)``."""
        self._require_fit()
        z_t = torch.as_tensor(z, dtype=torch.float32, device=self._device,
                               ).requires_grad_(True)
        E = self._model(z_t)
        grad_x = torch.autograd.grad(E.sum(), z_t, create_graph=False)[0]
        return (-grad_x).detach().cpu().numpy()

    def velocity(self, z: np.ndarray) -> np.ndarray:
        return self.config.beta * self.score(z)

    def jacobian_autodiff(self, z: np.ndarray, chunk: int = 128) -> np.ndarray:
        """``J = β · ∂s/∂z = -β · ∂²E/∂z∂z`` — symmetric by construction.

        Computed in chunks to keep peak memory bounded on 2k-cell / d=15
        latents (≈ 2k × 15 × 15 ≈ 450k mixed partials)."""
        self._require_fit()
        n, d = z.shape
        out = np.empty((n, d, d), dtype=np.float32)
        for start in range(0, n, int(chunk)):
            end = min(n, start + int(chunk))
            z_c = torch.as_tensor(z[start:end], dtype=torch.float32,
                                   device=self._device).requires_grad_(True)
            # vectorised Hessian: differentiate each coord of ∇E w.r.t. z
            # using torch.autograd.functional.jacobian in batched mode.
            def score_fn(x):
                xr = x.requires_grad_(True) if not x.requires_grad else x
                E = self._model(xr).sum()
                g = torch.autograd.grad(E, xr, create_graph=True)[0]
                return -g                                           # (B, d)
            s = score_fn(z_c)                                       # (B, d)
            J_rows = []
            for j in range(d):
                g_j = torch.autograd.grad(
                    s[:, j].sum(), z_c, create_graph=False, retain_graph=True,
                )[0]                                                 # (B, d)
                J_rows.append(g_j)
            J = torch.stack(J_rows, dim=1)                          # (B, d, d) rows=∂s_j/∂z
            out[start:end] = (self.config.beta * J).detach().cpu().numpy()
        return out

    def symmetry_check(self, z: np.ndarray, n_cells: int = 100,
                       seed: int = 0) -> dict:
        """Measure ``‖J − Jᵀ‖_F / ‖J‖_F`` on a random subsample. For a
        correct energy parameterisation the median should be at float32
        autodiff noise (``~ 1e-6``)."""
        rng = np.random.default_rng(seed)
        n = z.shape[0]
        idx = rng.choice(n, size=min(int(n_cells), n), replace=False)
        J = self.jacobian_autodiff(z[idx])
        asym = []
        for Jc in J:
            num = np.linalg.norm(Jc - Jc.T, ord="fro")
            den = np.linalg.norm(Jc, ord="fro") + 1e-12
            asym.append(float(num / den))
        return {
            "n_sample": int(len(idx)),
            "median_asym": float(np.median(asym)),
            "max_asym": float(np.max(asym)),
            "mean_asym": float(np.mean(asym)),
        }
