import json
import shutil
from pathlib import Path

import pytest

from serumgen.importer import classify, import_folder
import presets as real  # tests/presets.py — locates real presets, skips if absent

CASES = [
    ("Factory/Bass/Growler.SerumPreset", "bass"),
    ("Factory/Lead/Screamer.SerumPreset", "lead"),
    ("Factory/E Piano/Soft.SerumPreset", "keys"),
    ("Packs/SomePack/BA Deep [XY].SerumPreset", "bass"),
    ("Packs/SomePack/LD - Example Lead.SerumPreset", "lead"),
    ("Packs/Acme/AC_SND_Pluck_wobbly_pluck.SerumPreset", "pluck"),
    ("Packs/Acme/AC_SND_Pad_deep.SerumPreset", "pad"),
    ("Packs/X/PD Warm.SerumPreset", "pad"),
    ("Packs/X/Mystery Sound.SerumPreset", "unknown"),
    ("Factory/Bass/LD Not Really.SerumPreset", "bass"),  # folder wins over filename
    ("Packs/X/BS - Rumble.SerumPreset", "bass"),         # BS prefix = bass
    ("Packs/X/BS Thing 01.SerumPreset", "bass"),
    ("Packs/X/XX_YY_BS_Quiet_One.SerumPreset", "bass"),  # code mid-filename
    ("Packs/X/XX_ZZ_LD_Unison.SerumPreset", "lead"),
    ("Packs/X/Odd Noise.SerumPreset", "unknown"),        # no type signal
]


@pytest.mark.parametrize("relpath,expected", CASES)
def test_classify(relpath, expected):
    assert classify(Path(relpath)) == expected


def test_classify_metadata_fallback():
    meta = {"presetDescription": "fat bass patch", "tags": []}
    assert classify(Path("X/Mystery.SerumPreset"), metadata=meta) == "bass"




def _make_source(tmp_path):
    src = tmp_path / "src" / "Packs" / "Bass"
    src.mkdir(parents=True)
    for f in real.sample(3):
        shutil.copy(f, src / Path(f).name)
    (src / "legacy.fxp").write_bytes(b"not a real fxp")
    return tmp_path / "src"


def test_import_folder_builds_library(tmp_path):
    src = _make_source(tmp_path)
    lib = tmp_path / "library"
    stats = import_folder(src, lib)
    assert stats["imported"] == 3
    assert stats["fxp_found"] == 1
    assert stats["by_type"] == {"bass": 3}
    presets = list((lib / "bass").glob("*.SerumPreset"))
    params = list((lib / "bass").glob("*.json"))
    assert len(presets) == 3 and len(params) == 3
    index = json.loads((lib / "index.json").read_text())
    assert len(index) == 3
    entry = next(iter(index.values()))
    assert set(entry) == {"name", "type", "author", "source", "preset", "params"}
    # params cache is non-empty numeric features
    feats = json.loads((lib / entry["params"]).read_text())
    assert feats and all(isinstance(v, float) for v in feats.values())


def test_import_is_idempotent(tmp_path):
    src = _make_source(tmp_path)
    lib = tmp_path / "library"
    import_folder(src, lib)
    stats2 = import_folder(src, lib)
    assert stats2["imported"] == 0
    assert stats2["skipped_dupes"] == 3


def test_corrupt_preset_does_not_abort(tmp_path):
    src = _make_source(tmp_path)
    (src / "Packs" / "Bass" / "broken.SerumPreset").write_bytes(b"garbage")
    lib = tmp_path / "library"
    stats = import_folder(src, lib)
    assert stats["imported"] == 3
    assert stats["failed"] == 1


def test_import_collision_rename(tmp_path):
    """Test that same filename with different content gets renamed with hash suffix."""
    # Get two different bass presets to create content collision
    all_fixtures = real.sample(2)

    # Create source with two subfolders, each containing a file named "BA same.SerumPreset"
    # but with different content from different source files
    src = tmp_path / "src"
    src_one = src / "one"
    src_two = src / "two"
    src_one.mkdir(parents=True)
    src_two.mkdir(parents=True)

    collision_name = "BA same.SerumPreset"
    shutil.copy(all_fixtures[0], src_one / collision_name)
    shutil.copy(all_fixtures[1], src_two / collision_name)

    # Import and verify collision handling
    lib = tmp_path / "library"
    stats = import_folder(src, lib)

    assert stats["imported"] == 2
    assert stats["by_type"] == {"bass": 2}

    # Check that both presets and params files exist in library/bass
    bass_dir = lib / "bass"
    presets = sorted(bass_dir.glob("*.SerumPreset"))
    params = list(bass_dir.glob("*.json"))

    assert len(presets) == 2, f"Expected 2 presets, got {len(presets)}"
    assert len(params) == 2, f"Expected 2 params files, got {len(params)}"

    # Verify one has original name and one has hash suffix
    preset_names = [p.name for p in presets]
    assert "BA same.SerumPreset" in preset_names

    # The second file should have hash-suffixed name
    hash_suffixed = [n for n in preset_names if n != "BA same.SerumPreset"]
    assert len(hash_suffixed) == 1
    hash_name = hash_suffixed[0]

    # Verify format: BA same_<8 hex chars>.SerumPreset
    assert hash_name.startswith("BA same_")
    assert hash_name.endswith(".SerumPreset")
    hash_part = hash_name.replace("BA same_", "").replace(".SerumPreset", "")
    assert len(hash_part) == 8, f"Expected 8-char hash, got '{hash_part}'"
    # Verify it's hex
    try:
        int(hash_part, 16)
    except ValueError:
        pytest.fail(f"Hash part '{hash_part}' is not valid hex")
