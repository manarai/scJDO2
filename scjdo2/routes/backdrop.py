"""The 2-D landscape drawn behind the walkers (display only).

Height U = φ̃ + w·D̃ + wall on a grid over the floor: φ is a potential
fitted so −∇φ matches the drift projected onto the floor, D is scJDO2's
density energy (``EnergyDriftAdapter``) on the floor, both standardised over
cells. The walkers do not use this surface: on LARRY, walkers rolling down
it predicted clonal fate at chance, and on iPSC reprogramming it trapped
every walker in Trophoblast.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def project_velocity(X, V, U, k=30, tau=0.1):
    """kNN correlation-kernel projection of a latent velocity onto a 2-D embedding."""
    nb = cKDTree(X).query(X, k=k + 1)[1][:, 1:]
    out = np.zeros((len(X), 2))
    for i in range(len(X)):
        d = X[nb[i]] - X[i]
        cos = d @ V[i] / (np.linalg.norm(d, axis=1) * np.linalg.norm(V[i]) + 1e-12)
        w = np.exp(cos / tau)
        w /= w.sum()
        du = U[nb[i]] - U[i]
        out[i] = w @ du - du.mean(0)
    return out


def fit_potential(z, v, steps=6000, seed=0):
    """Softplus MLP φ with −∇φ ≈ v; best held-out step. Returns (φ, relative residual)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    val = rng.random(len(z)) < 0.1
    net = nn.Sequential(nn.Linear(2, 96), nn.Softplus(), nn.Linear(96, 96), nn.Softplus(),
                        nn.Linear(96, 96), nn.Softplus(), nn.Linear(96, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Z = torch.tensor(z, dtype=torch.float32)
    Vt = torch.tensor(v, dtype=torch.float32)
    tr, va = np.flatnonzero(~val), torch.tensor(np.flatnonzero(val))

    def grad(x, create):
        x = x.clone().requires_grad_(True)
        return torch.autograd.grad(net(x).sum(), x, create_graph=create)[0]

    best, best_state = np.inf, None
    for step in range(steps):
        b = torch.tensor(rng.choice(tr, 256))
        loss = ((-grad(Z[b], True) - Vt[b]) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0:
            vl = float(((-grad(Z[va], False) - Vt[va]) ** 2).sum(1).mean())
            if vl < best:
                best, best_state = vl, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    resid = best / float((Vt[va] ** 2).sum(1).mean())

    def phi(q):
        with torch.no_grad():
            return net(torch.tensor(q, dtype=torch.float32)).squeeze(-1).numpy()
    return phi, float(resid)


def surface(u, latent, V, w_density=0.5, grid=120, seed=0, potential_steps=6000, density_epochs=1500):
    """Grid surface for the viewer. ``u``: floor (unit SD), ``latent``/``V``: cells' latent position and drift."""
    from ..energy_drift import EnergyDriftAdapter
    v2 = project_velocity(latent, V, u)
    v2 /= np.median(np.linalg.norm(v2, axis=1)) + 1e-12
    phi, resid = fit_potential(u, v2, steps=potential_steps, seed=seed)
    dens = EnergyDriftAdapter(dim=2, hidden=96, depth=3, n_epochs=density_epochs, loss="esm",
                              device="cpu").fit(u.astype(np.float32))
    phi_c, d_c = phi(u), dens.energy(u.astype(np.float32))

    def std(x, ref):
        return (x - np.median(ref)) / ref.std()

    lo, hi = u.min(0) - 0.4, u.max(0) + 0.4
    side, mid = max(hi - lo), (lo + hi) / 2
    lo, hi = mid - side / 2, mid + side / 2
    gs = np.linspace(lo[0], hi[0], grid), np.linspace(lo[1], hi[1], grid)
    gx, gy = np.meshgrid(*gs)
    gp = np.c_[gx.ravel(), gy.ravel()]
    tree = cKDTree(u)
    dist = tree.query(gp, k=8)[0].mean(1)
    r0 = np.quantile(tree.query(u, k=9)[0][:, 1:].mean(1), 0.98)
    Ug = (std(phi(gp), phi_c) + w_density * std(dens.energy(gp.astype(np.float32)), d_c)
          + 40.0 * np.maximum(0, dist - r0) ** 2)
    base = float(np.quantile(std(phi_c, phi_c) + w_density * std(d_c, d_c), 0.01))
    Ug = Ug.reshape(gx.shape) - base
    inside = (dist <= 1.25 * r0).reshape(gx.shape)
    cap = float(Ug[inside].max())
    return {"x": gs[0].round(4).tolist(), "y": gs[1].round(4).tolist(),
            "U": np.clip(Ug, None, cap).round(3).tolist(), "inside": inside.astype(int).tolist(),
            "cap": round(cap, 3), "phi_residual": round(resid, 3), "w_density": w_density}
