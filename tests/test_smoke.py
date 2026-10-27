"""Phase A smoke tests — only verify that the package layout imports cleanly.

Phase B/C acceptance tests live in test_field_*.py / test_rates_*.py once the
corresponding implementations land.
"""

from __future__ import annotations


def test_package_imports():
    import scjdo2
    import scjdo2.pipeline  # noqa: F401
    import scjdo2.plot  # noqa: F401

    import re
    from pathlib import Path

    toml = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert scjdo2.__version__ == re.search(r'^version = "([^"]+)"', toml, re.M).group(1)


def test_all_modules_importable():
    import importlib

    for mod in (
        "scjdo2.embed",
        "scjdo2.rates",
        "scjdo2.field",
        "scjdo2.fisher",
        "scjdo2.opspace",
        "scjdo2.archetypes",
        "scjdo2.grammar",
        "scjdo2.io",
        "scjdo2.cli",
    ):
        importlib.import_module(mod)


def test_cli_help(capsys):
    import pytest

    from scjdo2 import cli

    with pytest.raises(SystemExit):
        cli.main(["--help"])
