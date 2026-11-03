"""Static (matplotlib-only) publication plots for scJDO2 operator-space results.

Public entry points:
    - :func:`figure_operator_space`  — Panels A, B (Panel C pends Phase C).
    - :func:`figure_archetype_sheet` — pends Phase B Step 8 adapter.
    - :func:`figure_v1_vs_v2`        — side-by-side comparison with v1 baseline.
    - :func:`style`                  — context manager enforcing journal style.
    - :func:`figure_behaviour_landscape` — Fisher-weighted density on the
      (lead_re, spectral_gap | henrici) plane with a scrambled-flow null.
    - :func:`figure_behaviour_ternary`   — eigenvalue-sign ternary + per-type
      composition bars split by condition.
"""

from .style import (
    BEHAVIOUR_COLOURS,
    BEHAVIOUR_MARKERS,
    CYCLIC,
    DIVERGING,
    DIVERGING_CMAP,
    OPSPACE_FIG_WIDTH_IN,
    PHASE_CMAP,
    confidence_alpha,
    relative_label,
    sensitive_ring,
    style,
)
from .figures import (
    figure_archetype_sheet,
    figure_ledger_summary,
    figure_operator_space,
    figure_v1_vs_v2,
    plot_drivers,
)
from .landscape import (
    behaviour_composition,
    figure_behaviour_landscape,
    figure_behaviour_landscape_3d,
    figure_behaviour_ternary,
    figure_energy_landscape,
)
from .orientation import figure_orientation
from .figures_v05 import (
    basin_index_by_type,
    ledger_map,
    oracle_agreement,
    perturb_paired,
    soft_axis_map,
    spectrum_ladder,
)

__all__ = [
    "BEHAVIOUR_COLOURS",
    "BEHAVIOUR_MARKERS",
    "CYCLIC",
    "DIVERGING",
    "DIVERGING_CMAP",
    "OPSPACE_FIG_WIDTH_IN",
    "PHASE_CMAP",
    "confidence_alpha",
    "relative_label",
    "sensitive_ring",
    "style",
    "figure_operator_space",
    "figure_archetype_sheet",
    "figure_ledger_summary",
    "figure_v1_vs_v2",
    "plot_drivers",
    # v0.5 symmetric-operator figures (snapshot defaults)
    "spectrum_ladder",
    "soft_axis_map",
    "ledger_map",
    "oracle_agreement",
    "perturb_paired",
    "basin_index_by_type",
    # behaviour-landscape figures (operator-space occupancy)
    "figure_behaviour_landscape",
    "figure_behaviour_landscape_3d",
    "figure_behaviour_ternary",
    "behaviour_composition",
    "figure_energy_landscape",
    "figure_orientation",
    "figure_flow_groups",
]
from .flow import figure_flow_groups  # noqa: E402

# Optional parts (not shipped in every release).
try:
    from .motifs import figure_motif_matrix  # noqa: E402
    __all__.append("figure_motif_matrix")
except ImportError:
    pass
