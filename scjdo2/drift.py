"""Step 2 — drift field.

Thin adapter around v1's :class:`scjdo.models.drift.DriftField`. v2 writes its
own ~50-line training loop reusing v1's :func:`scjdo.losses.denoising_score_matching`
and :func:`scjdo.losses.control_energy`; this avoids importing
``scjdo.tl._drift.fit_drift`` which is entangled with pseudotime and the
pseudotime-windowed Jacobian pipeline we are replacing.

Clock calibration: pass ``clock_scale`` to :meth:`DriftAdapter.fit` to rescale
``t`` per cell; the adapter records ``clock_calibrated = True``. Otherwise
``t`` is treated as a relative timescale and ``clock_calibrated = False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:  # pragma: no cover — v1 import; only required at .fit() time.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class DriftConfig:
    hidden: int = 128
    depth: int = 4
    beta: float = 0.1
    use_spectral_norm: bool = False     # v1 default is True — v2 defaults off per V2_PLAN R4
    use_velocity_prior: bool = False
    vel_scale: float = 2.0
    alpha_control: float = 1e-3
    n_epochs: int = 2000
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-4
    sigma: float = 0.1
    seed: int = 42
    device: Optional[str] = None
    # When True (default), compute a pseudotime-gradient velocity prior via
    # scjdo.tl._drift._pseudotime_velocity and add it as a soft anchor during
    # fit — needed to learn non-gradient flows (toggle-switch pitchfork).
    # Set to False to run DSM-only: the drift then recovers σ² ∇_z log p(z|t),
    # whose Jacobian is the Hessian of log p (symmetric, negative near modes).
    inject_velocity_prior: bool = True
    clock_calibrated: bool = field(default=False, init=False)


class DriftAdapter:
    """Learned drift ``f(z)`` on the latent space; wraps v1 DriftField."""

    def __init__(self, dim: int, config: Optional[DriftConfig] = None, **cfg_overrides):
        self.dim = int(dim)
        self.config = config or DriftConfig()
        for k, v in cfg_overrides.items():
            if not hasattr(self.config, k):
                raise AttributeError(f"DriftConfig has no field {k!r}")
            setattr(self.config, k, v)
        self._model = None  # lazy — created in fit()
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        z: np.ndarray,
        t: Optional[np.ndarray] = None,
        V_ref: Optional[np.ndarray] = None,
        clock_scale: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "DriftAdapter":
        if torch is None:
            raise ImportError("torch is required for DriftAdapter.fit")
        from scjdo.models.drift import DriftField, DriftConfig as V1DriftConfig
        from scjdo.losses import denoising_score_matching, control_energy
        from scjdo.tl._drift import _pseudotime_velocity

        rng = np.random.default_rng(self.config.seed)
        torch.manual_seed(self.config.seed)

        n = z.shape[0]
        if t is None:
            t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        else:
            t = np.asarray(t, dtype=np.float32)
        if clock_scale is not None:
            if clock_scale.shape != (n,):
                raise ValueError("clock_scale must have shape (n_cells,)")
            t = t * np.asarray(clock_scale, dtype=np.float32)
            self.config.clock_calibrated = True

        device = self.config.device or ("mps" if torch.backends.mps.is_available()
                                         else "cuda" if torch.cuda.is_available() else "cpu")
        X_ref = torch.as_tensor(z, dtype=torch.float32, device=device)
        t_t = torch.as_tensor(t, dtype=torch.float32, device=device)

        # Compute pseudotime-gradient velocity prior unless inject_velocity_prior
        # is False. v1 always does this; without it DSM alone recovers the
        # score of p(z|t), i.e. f ≈ σ² ∇_z log p(z|t). That is enough on
        # gradient flows (density follows a potential) but fails to converge
        # on non-gradient flows like the toggle-switch pitchfork — hence the
        # prior is on by default.
        use_prior = bool(self.config.inject_velocity_prior)
        if use_prior:
            if V_ref is None:
                V_np = _pseudotime_velocity(z.astype(np.float32), t.astype(np.float32), k=15)
            else:
                V_np = np.asarray(V_ref, dtype=np.float32)
            V_ref_t = torch.as_tensor(V_np, dtype=torch.float32, device=device)
        else:
            V_ref_t = None
        self.config.use_velocity_prior = use_prior

        v1_cfg = V1DriftConfig(
            dim=self.dim,
            hidden=self.config.hidden,
            depth=self.config.depth,
            beta=self.config.beta,
            use_spectral_norm=self.config.use_spectral_norm,
            use_velocity_prior=use_prior,
            vel_scale=self.config.vel_scale,
            alpha_control=self.config.alpha_control,
        )
        model = DriftField(v1_cfg, X_ref=X_ref, V_ref=V_ref_t).to(device)
        opt = torch.optim.AdamW(model.parameters(),
                                lr=self.config.lr,
                                weight_decay=self.config.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config.n_epochs)

        model.train()
        bs = min(self.config.batch_size, n)
        sigma = float(self.config.sigma)
        for epoch in range(self.config.n_epochs):
            idx = torch.as_tensor(rng.integers(0, n, size=bs), dtype=torch.long, device=device)
            x_b, t_b = X_ref[idx], t_t[idx]
            loss_dsm = denoising_score_matching(model, x_b, t_b, sigma=sigma)
            u = model(x_b, t_b)
            loss = loss_dsm + v1_cfg.alpha_control * control_energy(u)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if verbose and (epoch + 1) % max(1, self.config.n_epochs // 10) == 0:
                print(f"[drift] epoch {epoch + 1}/{self.config.n_epochs}  "
                      f"loss={float(loss.detach()):.4f}")

        model.eval()
        self._model = model
        self._fitted = True
        self._device = device
        return self

    # ---------------------------------------------------------- inference
    def _require_fit(self):
        if not self._fitted:
            raise RuntimeError("DriftAdapter.fit() must be called first")

    def f(self, z: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate drift at ``z``. ``t`` defaults to a per-cell scalar at 0.5."""
        self._require_fit()
        import torch
        n = z.shape[0]
        if t is None:
            t = np.full(n, 0.5, dtype=np.float32)
        z_t = torch.as_tensor(z, dtype=torch.float32, device=self._device)
        t_t = torch.as_tensor(t, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            v = self._model(z_t, t_t)
        return v.detach().cpu().numpy()

    def velocity(self, z: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
        return self.f(z, t)

    def jacobian_autodiff(self, z: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
        """Per-cell Jacobian by autodiff of the drift (shape ``(n, d, d)``).

        v2 does not use this inside the pipeline (Step 5 uses a ball regression);
        exposed for comparison tests only.
        """
        self._require_fit()
        import torch
        n = z.shape[0]
        if t is None:
            t = np.full(n, 0.5, dtype=np.float32)
        z_t = torch.as_tensor(z, dtype=torch.float32, device=self._device)
        t_t = torch.as_tensor(t, dtype=torch.float32, device=self._device)
        J = self._model.jacobian(z_t, t_t)
        return J.detach().cpu().numpy()
