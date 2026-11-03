# The mathematics of scJDO2

This page explains what scJDO2 computes, in equations, and what each result can
and cannot tell you. It follows the code: each section names the function that
implements it. You need linear algebra (eigenvalues, Jacobians) and some
familiarity with differential equations; no prior knowledge of optimal transport
or score matching is assumed.

**Contents**

1. [The idea: a local operator per cell](#1-the-idea-a-local-operator-per-cell)
2. [Time course: the flow operator](#2-time-course-the-flow-operator-sjflow)
3. [Snapshot: the energy operator](#3-snapshot-the-energy-operator-sjfit)
4. [Orientation: signs from measured evidence](#4-orientation-signs-from-measured-evidence-sjorient)
5. [What each route can identify](#5-what-each-route-can-identify)
6. [Notation and glossary](#6-notation-and-glossary)

---

## 1. The idea: a local operator per cell

Each cell $c$ has a position $z_c \in \mathbb{R}^d$ in a low-dimensional embedding
(typically the first $d = 10$–$20$ principal-component scores of log-normalised
counts). Assume cells move in that space according to

$
\frac{dz}{dt} = f(z) + \text{noise},
$

where $f$ is the **drift**. Near a cell, the drift is approximately linear:

$
f(z_c + \delta) \approx f(z_c) + J_c\,\delta,
\qquad
J_c = \frac{\partial f}{\partial z}\bigg|_{z_c} \in \mathbb{R}^{d\times d}.
$

$J_c$ is the **local operator** (the Jacobian of the drift). Its eigenvalues
$\lambda_1, \dots, \lambda_d$, ordered by real part, describe what happens to two
cells that start close together:

| leading eigenvalue $\lambda_1$ (largest real part) | nearby cells… | typical place |
|---|---|---|
| $\operatorname{Re}\lambda_1 < 0$ | move closer along every direction | inside a stable state |
| $\operatorname{Re}\lambda_1 > 0$ | separate along at least one direction | a branch point (fate decision) |
| $\operatorname{Re}\lambda_1 \approx 0$ | neither | along a path, or a flat region |

A non-zero imaginary part means rotation (for example, a cycle).

scJDO2 never observes $f$ directly. It estimates it in one of two ways,
depending on the data, and the two estimates mean different things:

* **Time course** (cells sampled at several times): estimate $f$ from how the
  population moves between times. This gives the **flow operator** (section 2).
* **Snapshot** (one time point): estimate the shape of the cell density. This
  gives the **energy operator**, which describes the density's curvature, not
  its dynamics (section 3).

Both are stored in `adata.uns["scjdo2"]["J"]` with shape $(n, d, d)$; the tag
`uns["scjdo2"]["meta"]["operator"]` says which one it is (`"flow"` or `"energy"`).

---

## 2. Time course: the flow operator (`sj.flow`)

Input: positions $z_c$ and a sampling time $t_c$ for every cell, with
$T \ge 2$ distinct times $t^{(1)} < \dots < t^{(T)}$.

### 2.1 Velocity from optimal transport (`ot_velocity`)

Cells are destroyed by sequencing, so no cell is observed twice. Instead, for
each consecutive pair of times, scJDO2 matches the cells of the earlier time,
$x_1, \dots, x_m$, with those of the later time, $y_1, \dots, y_{m'}$, by
**entropic optimal transport**. The transport plan $\pi \in \mathbb{R}_{\ge 0}^{m\times m'}$
solves

$
\pi^\star = \arg\min_{\pi} \sum_{j,k} \pi_{jk}\,\lVert x_j - y_k\rVert^2
\;-\; \varepsilon\, H(\pi)
\quad\text{s.t.}\quad
\sum_k \pi_{jk} = \tfrac{1}{m},\;\;
\sum_j \pi_{jk} = \tfrac{1}{m'},
$

with entropy $H(\pi) = -\sum_{jk}\pi_{jk}(\log \pi_{jk} - 1)$. The plan is the
cheapest way to turn one population into the other while moving cells as little
as possible; the entropy term blurs it so that each early cell maps to a
weighted cloud of later cells. It is computed with the Sinkhorn algorithm
(library POT). The regularisation scales with the data,

$
\varepsilon = \texttt{eps} \times \operatorname{median}_{j,k} \lVert x_j - y_k\rVert^2,
\qquad \texttt{eps} = 0.05 \text{ by default}.
$

Each early cell's **velocity** is the displacement to the centre of its cloud,
divided by the time step:

$
v_j = \frac{1}{t^{(i+1)} - t^{(i)}}
\left( \frac{\sum_k \pi^\star_{jk}\, y_k}{\sum_k \pi^\star_{jk}} - x_j \right).
$

Cells of the last time point have no later time and get no velocity
(`obs["flow_has_velocity"] = False`). Large pairs (more than $64\times 10^6$ cell pairs)
use mini-batch transport.

*Assumption.* The constraints conserve mass: every early cell goes somewhere
and every late cell comes from somewhere. Proliferation and death are not
modelled, so a change in the proportions of cell types is read as movement.

### 2.2 A smooth drift (`drift_jacobian`, `MeasuredDrift`)

The velocities are noisy and exist only at sampled cells. A neural network
$f_\theta : \mathbb{R}^d \to \mathbb{R}^d$ (a multilayer perceptron, hidden layers
96-96-96) is fitted to them by least squares,

$
\min_\theta \; \frac{1}{|\mathcal{C}|}\sum_{c \in \mathcal{C}} \lVert f_\theta(z_c) - v_c \rVert^2
\quad (+\text{ weight decay}),
$

over the cells $\mathcal{C}$ that have a velocity (AdamW optimiser, 1,500 steps of 256 cells).
The fitted $f_\theta$ is then evaluated at every cell, including the last time
point (by extrapolation).

### 2.3 The flow operator

$
J^{\text{flow}}_c = \frac{\partial f_\theta}{\partial z}\bigg|_{z_c},
$

computed exactly by automatic differentiation. $J^{\text{flow}}$ is **not**
symmetric in general, so it can have complex eigenvalues and can represent
rotation. Its leading eigenvalue is stored as `obs["flow_lambda1"]`
$= \operatorname{Re}\lambda_1(J^{\text{flow}}_c)$.

### 2.4 Descriptors (`flow_descriptors`)

Raw matrix entries depend on the coordinates, so scJDO2 summarises each
$J_c$ with eight numbers. With eigenvalues ordered by real part
($\operatorname{Re}\lambda_1 \ge \operatorname{Re}\lambda_2 \ge \dots$):

| name | formula |
|---|---|
| `sign_lead_re` | $\operatorname{sign}(\operatorname{Re}\lambda_1)$ |
| `n_positive_real` | $\#\{k : \operatorname{Re}\lambda_k > 0\}$ |
| `lead_re_over_max_abs_lambda` | $\operatorname{Re}\lambda_1 / \max_k\lvert\lambda_k\rvert$ |
| `max_abs_imag_over_max_abs_lambda` | $\max_k \lvert\operatorname{Im}\lambda_k\rvert / \max_k\lvert\lambda_k\rvert$ (rotation) |
| `spectral_gap_over_max_abs_lambda` | $(\operatorname{Re}\lambda_1 - \operatorname{Re}\lambda_2) / \max_k\lvert\lambda_k\rvert$ |
| `trace_over_sum_abs_lambda` | $\operatorname{tr} J / \sum_k \lvert\lambda_k\rvert$ |
| `henrici` | $\sqrt{\lVert J\rVert_F^2 - \sum_k \lvert\lambda_k\rvert^2} \,/\, \lVert J\rVert_F$ |
| `transient_gain` | $\max_{1\le k\le 30} \lVert J^k\rVert_2$ |

The Henrici index is zero exactly when $J$ is *normal* ($JJ^\top = J^\top J$, for
example symmetric); large values mean that perturbations can grow for a while
even when all eigenvalues are negative. The ranks of all eight within the
dataset are stored as `obsm["flow_descriptor_ranks"]`.

**Invariance.** If the coordinates change linearly, $z \mapsto Az$, the drift
becomes $Af(A^{-1}z)$ and the operator becomes $AJA^{-1}$, which has the
same eigenvalues. So the first six descriptors do not depend on the choice of
linear coordinates. Henrici and transient gain are unchanged only for
orthogonal $A$ (rotations). PCA, scVI and Factor Analysis are not related by
such maps; compare descriptors only within one embedding.

### 2.5 Group summaries and refit check (`flow_groups`, `flow_stability`)

The drift fit is random (initialisation, mini-batches), and per-cell
$\lambda_1$ varies between fits. scJDO2 therefore reads $\lambda_1$ at the group level:

* **`flow_groups`**: for each group $g$, the median of $\lambda_1$ over its
  cells, with a 95% percentile-bootstrap interval obtained by resampling those
  cells (1,000 resamples). The interval reflects cell-to-cell spread for **one**
  fitted operator; it does not include refit variation.
* **`flow_stability`**: refits transport and drift with seeds $s = 0, 1, 2$,
  giving $\lambda_1^{(s)}$, and reports the Spearman rank correlation
  $\rho\big(\lambda_1^{(s)}, \lambda_1^{(s')}\big)$ for every pair of seeds, per cell and
  per group median. `obs["flow_lambda1_rank_sd"]` is the standard deviation of
  each cell's rank quantile across the refits.

In the manuscript's tests the group results were stable across refits, but
per-cell rankings agreed only moderately (median Spearman 0.56). Check
`flow_stability` before reading individual cells.

---

## 3. Snapshot: the energy operator (`sj.fit`)

Input: positions $z_c$ from one time point. Without time, there is no motion to
measure. What can be estimated is the **density** of cells, $p(z)$.

### 3.1 Score matching (`EnergyDriftAdapter`)

Write the density as $p(z) \propto e^{-E(z)}$ for an **energy** $E$, modelled
by a neural network $E_\theta : \mathbb{R}^d \to \mathbb{R}$. The gradient of the
log density, $s(z) = \nabla \log p(z) = -\nabla E(z)$, is called the **score**.
It points toward denser regions.

The normalising constant of $p$ is unknown, but Hyvärinen's score-matching
objective does not need it. Up to a constant independent of $\theta$,

$$
\mathbb{E}_{p}\!\left[\tfrac{1}{2}\lVert \nabla \log p_\theta - \nabla \log p \rVert^2\right]
= \mathbb{E}_{p}\!\left[\tfrac{1}{2}\lVert \nabla E_\theta(z)\rVert^2 - \Delta E_\theta(z)\right] + \text{const},
$$

where $\Delta E = \operatorname{tr} \nabla^2 E$ is the Laplacian. The right-hand
side is averaged over the cells and minimised (`loss="esm"`). The Laplacian is
computed exactly for $d \le 20$ and by random probes
($\operatorname{tr} H \approx \tfrac{1}{K}\sum_k u_k^\top H u_k$, $u_k \in \{\pm 1\}^d$) above.

### 3.2 The operator

The drift is taken to be the score, $f(z) = -\beta\,\nabla E_\theta(z)$
($\beta = 1$ by default), and the operator is its Jacobian:

$$
J^{\text{energy}}_c = -\beta\,\nabla^2 E_\theta(z_c) = \beta\,\nabla^2 \log p(z_c).
$$

A Hessian is **symmetric**, so $J^{\text{energy}}$ has real eigenvalues and orthogonal
eigenvectors. It cannot represent rotation, and its eigenvectors have no sign.

**Example.** For a Gaussian cluster with covariance $\Sigma$,
$\nabla^2 \log p = -\Sigma^{-1}$: every eigenvalue is negative and equals
$-1/\sigma_k^2$, one per principal axis. Between two dense clusters,
$\log p$ curves upward along the line joining them, so one eigenvalue is
positive. `examples/snapshot_quickstart.py` shows this for
$E(x,y) = (x^2-4)^2/4 + y^2/2$: at the two wells $(\pm 2, 0)$,
$\nabla^2 E = \operatorname{diag}(8, 1)$, so $\lambda_1 = -1$; at the bridge $(0,0)$,
$\nabla^2 E = \operatorname{diag}(-4, 1)$, so $\lambda_1 = +4$.

### 3.3 Why this is density curvature, not stability

When is the energy operator equal to the true dynamical operator? Take the
stochastic dynamics

$
dz = f(z)\,dt + \sqrt{2}\,dW, \qquad f(z) = -(I + \gamma Q)\,\nabla U(z),
$

with $Q$ a constant antisymmetric matrix ($Q^\top = -Q$). The steady-state
density is $p \propto e^{-U}$ **for every $\gamma$**, because the rotating part
$-\gamma Q \nabla U$ only moves probability along the level sets of $U$. So:

* the score is $\nabla\log p = -\nabla U$, independent of $\gamma$;
* the energy operator is $-\nabla^2 U$;
* the true operator is $-(I + \gamma Q)\nabla^2 U$.

They agree only when $\gamma = 0$, that is, a gradient system with isotropic
noise at equilibrium (detailed balance). Rotation is invisible in a snapshot,
and the eigenvalues of the two operators differ once $\gamma \ne 0$. If the noise is not
uniform, or the population is not at steady state (development, proliferation,
uneven sampling), then $-\log p$ is not even the potential $U$. A density
can be dense because cells are stable there, or because many cells were made
or sampled there.

The package therefore reads $J^{\text{energy}}$ only as **the curvature of the cell
density**. In synthetic tests with a known landscape, the fitted energy
recovered a multi-well landscape in a 2-dimensional fit but not in 20
dimensions; for $d \ge 4$ treat $J^{\text{energy}}$ as descriptive.

### 3.4 Basin, fork and drift labels (`sj.analyse`, `classify_basin_latent`)

Each cell is labelled from the largest eigenvalue $\lambda_{\max}$ of
$J^{\text{energy}}_c$, against a tolerance $\tau$:

$
\text{label}_c =
\begin{cases}
\text{holder (basin)} & \lambda_{\max} < -\tau \quad\text{(density curves down in every direction: a local peak)}\\
\text{fork} & \lambda_{\max} > +\tau \quad\text{(curves up in some direction: a saddle or trough)}\\
\text{drift} & \text{otherwise (flat in some direction)}
\end{cases}
$

and `obs["op_basin_index"]` $= -\lambda_{\max}$ (larger means a sharper peak).

The tolerance comes from a **Gaussian oracle**: fit a Gaussian to each cluster
$g$ and compute its softest curvature, $|\lambda_{\max}(-\Sigma_g^{-1})| = 1/\sigma_{g,\max}^2$.
Then

$
\tau = 0.1 \times \operatorname{median}_g \frac{1}{\sigma_{g,\max}^2}.
$

Because $\tau$ depends on the cluster shapes and not on the fitted operator,
two datasets classified with the same oracle are comparable.

Other per-cell quantities: eigenvalue-sign fractions
(`op_frac_neg/zero/pos`, the share of eigenvalues below $-\tau$, within $\pm\tau$,
above $+\tau$), well stiffness, and `obs["energy"]` $= E_\theta(z_c)$.

---

## 4. Orientation: signs from measured evidence (`sj.orient`)

The eigenvectors $u_{c,1}, u_{c,2}, \dots$ of a symmetric $J^{\text{energy}}_c$ are
axes without a direction: $u$ and $-u$ are equally valid. The orientation
layer gives each axis a sign from a measured velocity $v_c$ (metabolic
labelling, RNA velocity, a lineage displacement, or a pseudotime gradient) in
the same embedding:

$
\sigma_{c,k} = \operatorname{sign}(v_c \cdot u_{c,k})
\quad\text{if}\quad
\frac{|v_c \cdot u_{c,k}|}{\lVert v_c\rVert} \ge 0.2,
\quad\text{otherwise no sign.}
$

Sources are grouped into two outputs that answer different questions and are
never merged: **flow** (how the cell moves now: labelling, then RNA velocity)
and **developmental** (where development goes over days: lineage barcodes,
then pseudotime). Within each output, the highest-ranked source with a sign
decides; disagreements are recorded as conflicts, not averaged.

* **Rotation** comes only from measured velocity: a ridge regression in the
  $k = 30$ nearest neighbours, $v_{c'} - v_c \approx M_c (z_{c'} - z_c)$, gives a
  local matrix $M_c$ whose antisymmetric part $\Omega_c = \tfrac{1}{2}(M_c - M_c^\top)$
  is stored as `omega`.
* **Reliability check**: RNA velocity is demoted to a cross-check when its
  signs agree with labelling on fewer than 60% of the (cell, axis) pairs both
  sign (with at least 200 such pairs).
* $J$ is never modified; its hash is checked before and after.

The full rules are in [ORIENTATION_SPEC.md](ORIENTATION_SPEC.md).

---

## 5. What each route can identify

| question | flow operator (time course) | energy operator (snapshot) |
|---|---|---|
| Which groups of cells sit at branch points? | yes, by group median $\lambda_1$ (tested on held-out synthetic data and LARRY) | no; it shows where the density is low or saddle-shaped |
| Is a single cell at a branch point? | weakly; check `flow_stability` | no |
| Rotation / cycles | in principle (complex eigenvalues) | never (symmetric) |
| Absolute rates (per hour) | in units of `time_key` only, within a dataset | no |
| The true Jacobian's entries (regulation) | no (median relative error ≈ 1 in tests) | no |
| Shape of the cell density | indirectly | yes (2-d reliably; descriptive for $d \ge 4$) |
| Stability of a steady-state population | not from single-window labelling velocity (tested) | no (section 3.3) |

---

## 6. Notation and glossary

| symbol / term | meaning |
|---|---|
| $z_c$ | position of cell $c$ in the embedding (`obsm[use_rep]`) |
| $d$ | number of embedding dimensions used |
| $f$ | drift: the average velocity at each position |
| $J_c$ | local operator: Jacobian $\partial f/\partial z$ at cell $c$ |
| $\lambda_1$ | eigenvalue of $J_c$ with the largest real part |
| optimal transport (OT) | matching two populations at minimum total squared distance |
| entropic regularisation $\varepsilon$ | blurs the matching; larger means smoother velocities |
| barycentric velocity | displacement to the weighted centre of a cell's matched cells, per unit time |
| score | $\nabla \log p$, the direction of increasing cell density |
| energy $E$ | $-\log p$ up to a constant; low energy = dense |
| score matching | fitting the score without knowing the density's normalising constant |
| holder / fork / drift | density curvature labels: local peak / saddle or trough / flat |
| oracle tolerance $\tau$ | the near-zero band for eigenvalues, from Gaussian fits to the clusters |
| orientation | assigning directions to the operator's axes from measured velocity |
| detailed balance | equilibrium with no net circulating flux; the only case where density curvature equals stability |

## References

* Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. *JMLR* 6, 695–709.
* Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of optimal transport. *NeurIPS*.
* Schiebinger, G. *et al.* (2019). Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. *Cell* 176, 928–943.
* Trefethen, L. N. & Embree, M. (2005). *Spectra and Pseudospectra*. Princeton University Press (non-normality, Henrici departure).
