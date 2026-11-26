"""Thin wrapper around the serum-preset-packager submodule."""
import json
import sys
import tempfile
from pathlib import Path

_PACKAGER_DIR = Path(__file__).resolve().parent.parent / "serum-preset-packager"
if str(_PACKAGER_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGER_DIR))

import cli as _packager  # noqa: E402  (the submodule's module is named cli)


def unpack(src: Path, dst: Path) -> None:
    _packager.unpack(Path(src), Path(dst))


def pack(src: Path, dst: Path) -> None:
    _packager.pack(Path(src), Path(dst))


def load(src: Path) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        unpack(Path(src), tmp_path)
        with open(tmp_path) as f:
            return json.load(f)
    finally:
        tmp_path.unlink(missing_ok=True)
