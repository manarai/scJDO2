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
from .figures_v05 import (
    add_aging_footer,
    aging_vectors,
    basin_index_by_type,
    energy_transect,
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
    "aging_vectors",
    "energy_transect",
    "add_aging_footer",
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
]
