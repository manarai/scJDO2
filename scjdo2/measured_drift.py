"""Measured-flow drift wrapper (V2_PLAN R16 Phase D §7).

An alternative to Step 2's energy drift: a small MLP ``f(z, θ)``
trained by direct MSE regression on a *measured* per-cell velocity
field (e.g. labelled scNT-seq rates from Dynamo, or scVelo dynamical).
Exposes the same ``.velocity(z, t)`` + ``.jacobian_autodiff(z, t)``
surface as :class:`scjdo2.energy_drift.EnergyDriftAdapter`, so Phase C's rate-model
+ anchor-fit path consumes it unchanged.

Direction-free in the behaviour sense: the MLP is a smooth interpolant
of the measured flow. Direction here comes from the chemistry
(labelling pulse) or from splicing kinetics, not from pseudotime or
annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # pragma: no cover
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class MeasuredDriftConfig:
    hidden: tuple[int, ...] = (96, 96, 96)
    n_epochs: int = 1500
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: Optional[str] = None
    # ``clock_calibrated`` records whether the measured flow is in
    # physical time units (e.g. per hour from labelling). If True, the
    # downstream opspace figures carry no ``(relative)`` label.
    clock_calibrated: bool = False


class _MLP(nn.Module if nn is not None else object):
    def __init__(self, d: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = d
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        layers.append(nn.Linear(prev, d))
        self.net = nn.Sequential(*layers)

    def forward(self, z: "torch.Tensor") -> "torch.Tensor":
        return self.net(z)


class MeasuredDrift:
    """Minimal measured-flow drift. ``.fit(z, v_measured)`` runs
    MSE regression; ``.velocity(z, t)`` returns the MLP prediction;
    ``.jacobian_autodiff(z, t)`` returns ``∂f/∂z`` via autograd.
    """

    def __init__(self, dim: int, config: Optional[MeasuredDriftConfig] = None,
                 **cfg_overrides):
        self.dim = int(dim)
        self.config = config or MeasuredDriftConfig()
        for k, v in cfg_overrides.items():
            if not hasattr(self.config, k):
                raise AttributeError(f"MeasuredDriftConfig has no field {k!r}")
            setattr(self.config, k, v)
        self._model = None
        self._fitted = False

    def fit(
        self,
        z: np.ndarray,
        v_measured: np.ndarray,
        verbose: bool = False,
    ) -> "MeasuredDrift":
        if torch is None:
            raise ImportError("torch is required for MeasuredDrift.fit")
        rng = np.random.default_rng(self.config.seed)
        torch.manual_seed(self.config.seed)
        n = z.shape[0]
        d = z.shape[1]
        if v_measured.shape != (n, d):
            raise ValueError(
                f"v_measured {v_measured.shape} must match z {(n, d)}"
            )
        device = self.config.device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        X = torch.as_tensor(z, dtype=torch.float32, device=device)
        V = torch.as_tensor(v_measured, dtype=torch.float32, device=device)
        model = _MLP(d, tuple(self.config.hidden)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.config.lr,
                                weight_decay=self.config.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.config.n_epochs,
        )
        bs = min(self.config.batch_size, n)
        model.train()
        for epoch in range(self.config.n_epochs):
            idx = torch.as_tensor(rng.integers(0, n, size=bs),
                                   dtype=torch.long, device=device)
            pred = model(X[idx])
            loss = ((pred - V[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if verbose and (epoch + 1) % max(1, self.config.n_epochs // 10) == 0:
                print(f"[measured-drift] epoch {epoch+1}/{self.config.n_epochs}  "
                      f"MSE={float(loss.detach()):.6g}")
        model.eval()
        self._model = model
        self._fitted = True
        return self

    def _tensor(self, z: np.ndarray) -> "torch.Tensor":
        if self._model is None:
            raise RuntimeError("MeasuredDrift.fit must be called first")
        device = next(self._model.parameters()).device
        return torch.as_tensor(z, dtype=torch.float32, device=device)

    def velocity(self, z: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
        """Return per-cell velocity ``f(z)``. ``t`` is ignored (the
        measured flow is time-homogeneous in this wrapper)."""
        with torch.no_grad():
            v = self._model(self._tensor(z))
        return v.cpu().numpy()

    def jacobian_autodiff(
        self, z: np.ndarray, t: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-cell ``∂f/∂z`` via autograd — the Phase B Jacobian path.
        Returns ``(n, d, d)`` float64.
        """
        if self._model is None:
            raise RuntimeError("MeasuredDrift.fit must be called first")
        zt = self._tensor(z)
        n, d = zt.shape
        J = np.empty((n, d, d), dtype=np.float64)
        for i in range(n):
            zi = zt[i:i + 1].detach().requires_grad_(True)
            vi = self._model(zi)[0]
            for g in range(d):
                grad = torch.autograd.grad(
                    vi[g], zi, retain_graph=(g < d - 1),
                    create_graph=False,
                )[0]
                J[i, g, :] = grad[0].detach().cpu().numpy()
        return J
