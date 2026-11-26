import json
import random
from pathlib import Path

import pytest
import torch

from serumgen.generate import (decode_to_features, encode_preset,
                               inject_features, load_model, generate, interpolate)
from serumgen.train import train_type
import presets as real  # tests/presets.py — locates real presets, skips if absent



def test_inject_features_writes_matching_params():
    data = {"data": {
        "Env0": {"plainParams": {"kParamAttack": 0.0}},
        "Oscillator0": {"WTOsc0": {"plainParams": {"kParamTablePos": 0.0}}},
    }}
    inject_features(data, {"Env0/kParamAttack": 0.5,
                           "Oscillator0/WTOsc0/kParamTablePos": 88.0,
                           "Env9/kParamMissing": 1.0,
                           "Env0/kParamUnknown": 1.0})
    assert data["data"]["Env0"]["plainParams"]["kParamAttack"] == 0.5
    assert data["data"]["Oscillator0"]["WTOsc0"]["plainParams"]["kParamTablePos"] == 88.0
    assert "kParamUnknown" not in data["data"]["Env0"]["plainParams"]


def test_nearest_template_picks_closest():
    import numpy as np

    from serumgen.generate import _nearest_template
    paths = ["a", "b", "c"]
    mus = np.array([[0.0, 0.0], [5.0, 5.0], [-3.0, -3.0]])
    assert _nearest_template(np.array([4.5, 4.6]), paths, mus) == "b"
    assert _nearest_template(np.array([0.1, -0.1]), paths, mus) == "a"


def test_sample_near_library_stays_near_real_presets():
    """Collapsed latent dims (library std ~0.03) must not get prior-sized noise."""
    import numpy as np

    from serumgen.generate import sample_near_library
    rng = np.random.default_rng(0)
    # 200 presets: 2 informative dims, 14 near-constant (posterior collapse)
    mus = np.zeros((200, 16))
    mus[:, 0] = rng.normal(0, 1.0, 200)
    mus[:, 1] = rng.normal(0, 0.5, 200)
    mus[:, 2:] = rng.normal(0, 0.03, (200, 14))

    def nn_dist(z):
        return np.sqrt(((mus - z) ** 2).sum(1)).min()

    low = [nn_dist(sample_near_library(mus, 0.2, rng).numpy()[0]) for _ in range(200)]
    high = [nn_dist(sample_near_library(mus, 1.8, rng).numpy()[0]) for _ in range(200)]
    prior = [nn_dist(rng.normal(0, 0.2, 16)) for _ in range(200)]
    assert np.median(low) < 0.2 * np.median(prior)   # far closer than randn*T
    assert np.median(high) > 2 * np.median(low)      # temperature still spreads


def test_sample_near_library_shape_and_dtype():
    import numpy as np

    from serumgen.generate import sample_near_library
    z = sample_near_library(np.zeros((5, 8)), 0.8, np.random.default_rng(1))
    assert z.shape == (1, 8) and z.dtype == torch.float32


def test_load_model_missing_exits(tmp_path):
    with pytest.raises(SystemExit):
        load_model(tmp_path, "bass")


@pytest.fixture(scope="module")
def real_library(tmp_path_factory):
    """Real presets + a real (tiny) trained model."""
    from serumgen.importer import import_folder
    import shutil
    tmp = tmp_path_factory.mktemp("gen")
    src = tmp / "src" / "Bass"
    src.mkdir(parents=True)
    for f in real.sample(45):
        shutil.copy(f, src / Path(f).name)
    lib = tmp / "library"
    import_folder(tmp / "src", lib)
    models = tmp / "models"
    train_type("bass", lib, models, epochs=5, latent_dim=4)
    return lib, models, tmp


def test_encode_decode_roundtrip(real_library):
    lib, models, tmp = real_library
    model, scaler, keys, meta = load_model(models, "bass")
    preset = next((lib / "bass").glob("*.SerumPreset"))
    mu = encode_preset(preset, model, scaler, keys)
    assert mu.shape == (1, meta["latent_dim"])
    feats = decode_to_features(mu, model, scaler, keys)
    assert set(feats) == set(keys)


def test_generate_produces_loadable_presets(real_library):
    lib, models, tmp = real_library
    out = tmp / "generated"
    paths = generate("bass", lib, models, out, n=2, temperature=0.5)
    assert len(paths) == 2
    from serumgen import packager
    data = packager.load(paths[0])
    meta = data["metadata"]
    assert meta["presetAuthor"] == "audiomimi experimental labs"
    assert meta["vendor"] == "audiomimi experimental labs"
    # nothing identifying the donor preset survives into the generated file
    assert meta["url"] == ""
    assert meta["tags"] == []
    assert meta["hash"] == ""
    assert "audiomimi" in meta["presetDescription"].lower()


def test_sanitize_metadata_strips_donor_identity():
    from serumgen.generate import _sanitize_metadata
    donor_meta = {
        "fileType": "SerumPreset", "product": "Serum2", "productVersion": "2.0.11",
        "version": 4.0, "hash": "abc123", "presetAuthor": "Jane Doe",
        "presetDescription": "Donor description", "presetName": "BA - Old Name",
        "tags": ["Wavetable", "Poly", "Preview"], "url": "https://example.com/",
        "vendor": "Example Sounds",
    }
    _sanitize_metadata(donor_meta, "Bass_abc123")
    assert donor_meta["presetAuthor"] == "audiomimi experimental labs"
    assert donor_meta["vendor"] == "audiomimi experimental labs"
    assert donor_meta["presetName"] == "Bass_abc123"
    assert donor_meta["url"] == ""
    assert donor_meta["tags"] == []
    assert donor_meta["hash"] == ""
    # format-schema fields (not donor identity) are left alone
    assert donor_meta["fileType"] == "SerumPreset"
    assert donor_meta["product"] == "Serum2"
    assert donor_meta["productVersion"] == "2.0.11"
    assert donor_meta["version"] == 4.0


def test_generate_like(real_library):
    """Test generate with like parameter (gaussian around encoded preset)."""
    lib, models, tmp = real_library
    out = tmp / "gen_like"

    # Pick a real preset from the library
    preset = next((lib / "bass").glob("*.SerumPreset"))

    # Generate around that preset with spread=0.1
    paths = generate("bass", lib, models, out, n=2, like=str(preset), spread=0.1)

    # Assert 2 files written and exist
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_generate_template_filter(real_library):
    """Test generate with template substring filter and SystemExit on no match."""
    lib, models, tmp = real_library
    out = tmp / "gen_template"

    # Get a real template name and use substring of it
    template_file = next((lib / "bass").glob("*.SerumPreset"))
    template_substr = template_file.stem[:5]  # First 5 chars of filename

    # Generate with matching template substring
    paths = generate("bass", lib, models, out, n=1, template=template_substr)
    assert len(paths) == 1
    assert paths[0].exists()

    # Assert SystemExit for non-matching template
    out_fail = tmp / "gen_template_fail"
    with pytest.raises(SystemExit):
        generate("bass", lib, models, out_fail, n=1, template="zzz-no-such-template")


def test_generate_samples_near_whole_library(real_library, monkeypatch):
    """CLI generate anchors on the full library, even when --template narrows donors."""
    import serumgen.generate as gen
    lib, models, tmp = real_library
    n_lib = len(list((lib / "bass").glob("*.SerumPreset")))
    seen = []
    real = gen.sample_near_library

    def spy(mus, temperature, rng=None):
        seen.append((len(mus), temperature))
        return real(mus, temperature, rng)

    monkeypatch.setattr(gen, "sample_near_library", spy)
    stem = next((lib / "bass").glob("*.SerumPreset")).stem
    gen.generate("bass", lib, models, tmp / "gen_near", n=2, temperature=0.3,
                 template=stem)
    assert seen == [(n_lib, 0.3), (n_lib, 0.3)]


def test_generate_like_does_not_use_library_sampling(real_library, monkeypatch):
    import serumgen.generate as gen
    lib, models, tmp = real_library
    monkeypatch.setattr(gen, "sample_near_library",
                        lambda *a, **k: pytest.fail("--like has its own centre"))
    like = next((lib / "bass").glob("*.SerumPreset"))
    gen.generate("bass", lib, models, tmp / "gen_like2", n=1, like=like)


def test_interpolate(real_library):
    """Test interpolate function creating morph sequence."""
    lib, models, tmp = real_library
    out = tmp / "interpolate"

    # Pick two different presets
    presets = list((lib / "bass").glob("*.SerumPreset"))
    assert len(presets) >= 2, "Need at least 2 presets for interpolation"
    preset_a = str(presets[0])
    preset_b = str(presets[1])

    # Interpolate with 3 steps
    steps = 3
    paths = interpolate("bass", lib, models, out, preset_a, preset_b, steps=steps)

    # Assert correct number of files
    assert len(paths) == steps

    # Assert correct filenames and all exist
    expected_names = [
        f"Bass_morph_01of{steps}.SerumPreset",
        f"Bass_morph_02of{steps}.SerumPreset",
        f"Bass_morph_03of{steps}.SerumPreset",
    ]
    for i, expected_name in enumerate(expected_names):
        assert paths[i].name == expected_name
        assert paths[i].exists()
        assert paths[i].stat().st_size > 0
