"""Synthetic datasets for scJDO2 tests and benchmarks.

In-repo generators so scJDO2's acceptance tests do not depend on v1's
``scratchpad_run/`` directory. The source for the pitchfork system is the
mid-interval saddle control at
``scJDO/scratchpad_run/saddle_readouts_midinterval.py``, resolving risk R2
from V2_PLAN.md.
"""

from .dynamo_hematopoiesis import load_dynamo_hematopoiesis
from .larry import clone_pairs, clone_split_positions, load_larry
from .synthetic_pitchfork import (
    ALPHA_CRIT,
    DEFAULT_WIDTH,
    analytic_jacobian,
    generate_pitchfork,
    generate_pitchfork_equilibrated,
)

__all__ = [
    "generate_pitchfork",
    "generate_pitchfork_equilibrated",
    "analytic_jacobian",
    "ALPHA_CRIT",
    "DEFAULT_WIDTH",
    "load_larry",
    "clone_pairs",
    "clone_split_positions",
    "load_dynamo_hematopoiesis",
]
