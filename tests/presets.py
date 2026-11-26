"""Locate real `.SerumPreset` files for the integration tests.

No presets ship with this repository. Tests that exercise the full pipeline
against real binary data look in `$SERUMGEN_TEST_PRESETS` (any folder of
`.SerumPreset` files, searched recursively), falling back to your local
`data/library/bass/`. When too few presets are found, those tests are
skipped rather than failed.
"""
import os
from pathlib import Path

import pytest

PRESET_DIR = Path(os.environ.get("SERUMGEN_TEST_PRESETS", "data/library/bass"))
ALL = sorted(PRESET_DIR.rglob("*.SerumPreset")) if PRESET_DIR.is_dir() else []


def _reason(n: int) -> str:
    return (f"needs >= {n} real .SerumPreset files in {PRESET_DIR} "
            f"(found {len(ALL)}); set SERUMGEN_TEST_PRESETS to a preset folder")


def sample(n: int) -> list[Path]:
    """`n` presets spread evenly across the pool, or skip the calling test."""
    if len(ALL) < n:
        pytest.skip(_reason(n))
    step = len(ALL) // n
    return ALL[::step][:n]


def requires(n: int):
    """Mark a test as needing at least `n` real presets."""
    return pytest.mark.skipif(len(ALL) < n, reason=_reason(n))
