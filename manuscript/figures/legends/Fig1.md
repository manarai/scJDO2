**Figure 1 | Two operators, one operator space.** Schematic (no data). Time information
determines which local operator can be estimated. A snapshot supports only the
**score operator** *J_score = −β∇²E(z)*, the Hessian of a neural energy fitted by score
matching; it is symmetric and describes density curvature. A **measured** velocity (entropic
optimal transport between sampling days, lineage displacement, metabolic labelling or RNA
velocity) supports the **flow operator** *J_flow = ∂f/∂z* of a smooth drift *f* fitted to that
velocity (MeasuredDrift). Per cell, magnitude-free descriptors, rank-transformed, define an
operator space whose recurrent patterns are archetypes (compared with Leiden clustering at
matched K). Under a change of coordinates *A* the score operator transforms as *A⁻ᵀHA⁻¹*
(congruence) and the flow operator as *AJA⁻¹* (similarity). Spectral descriptors (sign of Re λ₁,
number of eigenvalues with positive real part, rotational index, spectral gap, trace) are
invariant for the flow operator under any invertible *A*; non-normality (Henrici) and transient
gain only under orthogonal *A*; for the score operator only the inertia is invariant, and only
inertia descriptors are compared between the two operators. Different embeddings (PCA, scVI,
FA) are not related by such maps.
Bottom: the six identifiability components and the tasks in which each is tested.
