"""Synthetic datasets for scJDO2 tests and benchmarks.

In-repo generators so scJDO2's acceptance tests do not depend on v1's
``scratchpad_run/`` directory. The source for the pitchfork system is the
mid-interval saddle control at
``scJDO/scratchpad_run/saddle_readouts_midinterval.py``, resolving risk R2
from V2_PLAN.md.
"""

from .synthetic_pitchfork import generate_pitchfork, ALPHA_CRIT, DEFAULT_WIDTH

__all__ = ["generate_pitchfork", "ALPHA_CRIT", "DEFAULT_WIDTH"]
