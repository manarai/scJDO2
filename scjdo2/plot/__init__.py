"""Static (matplotlib-only) publication plots for scJDO2 operator-space results.

Public entry points:
    - :func:`figure_operator_space`  — Panels A, B (Panel C pends Phase C).
    - :func:`figure_archetype_sheet` — pends Phase B Step 8 adapter.
    - :func:`figure_v1_vs_v2`        — side-by-side comparison with v1 baseline.
    - :func:`style`                  — context manager enforcing journal style.
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
]
