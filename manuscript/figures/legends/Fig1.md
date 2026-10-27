**Figure 1 | Two operators, one operator space.** Schematic (no data). Time information
determines which local operator can be estimated. A snapshot supports only the
**score operator** *J_score = −β∇²E(z)*, the Hessian of a neural energy fitted by score
matching; it is symmetric and describes density curvature. A **measured** velocity (entropic
optimal transport between sampling days, lineage displacement, metabolic labelling or RNA
velocity) supports the **flow operator** *J_flow = ∂f/∂z* of a smooth drift *f* fitted to that
velocity (MeasuredDrift). Per cell, basis-invariant descriptors (sign of Re λ₁, spectrum,
rotational index, non-normality, transient gain), standardised and rank-transformed, define an
operator space whose recurrent patterns are archetypes (compared with Leiden clustering at
matched K). Under a change of coordinates *A* the score operator transforms as *A⁻ᵀHA⁻¹* and
the flow operator as *AJA⁻¹*, so only descriptors invariant under both are compared directly.
Bottom: the six identifiability components and the tasks in which each is tested.
