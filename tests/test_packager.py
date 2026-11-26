import json

import pytest

from serumgen import packager
import presets as real  # tests/presets.py — locates real presets, skips if absent


@pytest.fixture
def fixture_preset():
    return real.sample(1)[0]


def test_load_returns_metadata_and_data(fixture_preset):
    data = packager.load(fixture_preset)
    assert "metadata" in data
    assert "data" in data
    assert isinstance(data["metadata"].get("presetName"), str)


def test_pack_roundtrip(tmp_path, fixture_preset):
    data = packager.load(fixture_preset)
    json_path = tmp_path / "preset.json"
    json_path.write_text(json.dumps(data))
    out = tmp_path / "out.SerumPreset"
    packager.pack(json_path, out)
    assert out.exists() and out.stat().st_size > 1000
    # reload the packed file to prove it is still parseable
    reloaded = packager.load(out)
    assert reloaded["metadata"]["presetName"] == data["metadata"]["presetName"]
