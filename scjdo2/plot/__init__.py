"""Static (matplotlib-only) publication figures for scJDO2.

One module per topic; each draws the results of the matching analysis
module and never runs analyses itself (the one-call APIs
:mod:`scjdo2.landscape` and :mod:`scjdo2.drivers` do that):

==================  =====================================================
module              figures
==================  =====================================================
``operator_space``  figure_operator_space, figure_archetype_sheet,
                    figure_ledger_summary, plot_drivers (axis-level)
``compare``         figure_v1_vs_v2
``symmetric``       figure_spectrum_ladder, figure_soft_axis_map,
                    figure_ledger_map, figure_oracle_agreement,
                    figure_basin_index  (v0.5 symmetric operator)
``aging``           figure_aging_vectors, figure_energy_transect,
                    add_aging_footer
``perturbation``    figure_perturb_paired
``landscape``       figure_behaviour_landscape(_3d), figure_behaviour_ternary,
                    figure_energy_landscape, behaviour_composition (table)
``motifs``          figure_motif_matrix
``drivers``         figure_influence_network, figure_reference_auroc
``timing``          figure_expression_cascade, figure_branch_timing,
                    figure_network_expression
``style``           journal style, colour keys, label helpers
==================  =====================================================

Conventions: every ``figure_*`` returns a :class:`~matplotlib.figure.Figure`
and takes ``out`` — a path stem; PDF + PNG are written via
:func:`scjdo2.plot._save.save_figure`.

The pre-reorganisation names (``spectrum_ladder``, ``ledger_map``,
``perturb_paired`` …) remain importable here as aliases.
"""

from ._save import save_figure
from .aging import add_aging_footer, figure_aging_vectors, figure_energy_transect
from .compare import figure_v1_vs_v2
from .drivers import figure_influence_network, figure_reference_auroc
from .landscape import (
    behaviour_composition,
    figure_behaviour_landscape,
    figure_behaviour_landscape_3d,
    figure_behaviour_ternary,
    figure_energy_landscape,
)
from .motifs import figure_motif_matrix
from .operator_space import (
    figure_archetype_sheet,
    figure_ledger_summary,
    figure_operator_space,
    plot_drivers,
)
from .perturbation import figure_perturb_paired
from .style import (
    BEHAVIOUR_COLOURS,
    BEHAVIOUR_MARKERS,
    CYCLIC,
    DIVERGING,
    DIVERGING_CMAP,
    OPSPACE_FIG_WIDTH_IN,
    PHASE_CMAP,
    ROLE_COLOURS,
    confidence_alpha,
    relative_label,
    sensitive_ring,
    style,
)
from .symmetric import (
    figure_basin_index,
    figure_ledger_map,
    figure_oracle_agreement,
    figure_soft_axis_map,
    figure_spectrum_ladder,
)
from .timing import (
    figure_branch_timing,
    figure_expression_cascade,
    figure_network_expression,
)

# Aliases for the pre-reorganisation names (same objects).
spectrum_ladder = figure_spectrum_ladder
soft_axis_map = figure_soft_axis_map
ledger_map = figure_ledger_map
oracle_agreement = figure_oracle_agreement
basin_index_by_type = figure_basin_index
perturb_paired = figure_perturb_paired
aging_vectors = figure_aging_vectors
energy_transect = figure_energy_transect

__all__ = [
    # style
    "BEHAVIOUR_COLOURS",
    "BEHAVIOUR_MARKERS",
    "ROLE_COLOURS",
    "CYCLIC",
    "DIVERGING",
    "DIVERGING_CMAP",
    "OPSPACE_FIG_WIDTH_IN",
    "PHASE_CMAP",
    "confidence_alpha",
    "relative_label",
    "sensitive_ring",
    "style",
    "save_figure",
    # operator space
    "figure_operator_space",
    "figure_archetype_sheet",
    "figure_ledger_summary",
    "plot_drivers",
    # compare
    "figure_v1_vs_v2",
    # v0.5 symmetric operator
    "figure_spectrum_ladder",
    "figure_soft_axis_map",
    "figure_ledger_map",
    "figure_oracle_agreement",
    "figure_basin_index",
    # aging
    "figure_aging_vectors",
    "figure_energy_transect",
    "add_aging_footer",
    # perturbation
    "figure_perturb_paired",
    # behaviour landscape
    "figure_behaviour_landscape",
    "figure_behaviour_landscape_3d",
    "figure_behaviour_ternary",
    "figure_energy_landscape",
    "behaviour_composition",
    # motifs
    "figure_motif_matrix",
    # drivers / responders
    "figure_influence_network",
    "figure_reference_auroc",
    # timing QC
    "figure_expression_cascade",
    "figure_branch_timing",
    "figure_network_expression",
    # aliases (pre-reorganisation names)
    "spectrum_ladder",
    "soft_axis_map",
    "ledger_map",
    "oracle_agreement",
    "basin_index_by_type",
    "perturb_paired",
    "aging_vectors",
    "energy_transect",
]
