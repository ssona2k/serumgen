import json
import random
from pathlib import Path

import numpy as np
import pytest

from serumgen.train import build_matrix, load_features, train_type


def _fake_library(tmp_path, n=50, ptype="bass"):
    tdir = tmp_path / "library" / ptype
    tdir.mkdir(parents=True)
    random.seed(0)
    for i in range(n):
        feats = {f"Env0_kParam{j}": random.random() for j in range(20)}
        (tdir / f"preset_{i:03d}.json").write_text(json.dumps(feats))
    return tmp_path / "library"


def test_load_features(tmp_path):
    lib = _fake_library(tmp_path)
    feats = load_features(lib, "bass")
    assert len(feats) == 50
    assert all(len(v) == 20 for v in feats.values())


def test_load_features_missing_type_exits(tmp_path):
    with pytest.raises(SystemExit):
        load_features(tmp_path / "library", "pad")


def test_train_type_saves_artifacts(tmp_path):
    lib = _fake_library(tmp_path)
    models = tmp_path / "models"
    out = train_type("bass", lib, models, epochs=5, latent_dim=4)
    assert (out / "vae.pth").exists()
    assert (out / "scaler.pkl").exists()
    assert (out / "keys.json").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["latent_dim"] == 4
    assert meta["input_dim"] == 20
    assert meta["num_presets"] == 50


def test_train_type_refuses_small_dataset(tmp_path):
    lib = _fake_library(tmp_path, n=10)
    with pytest.raises(SystemExit):
        train_type("bass", lib, tmp_path / "models", epochs=1)
    # --force overrides
    out = train_type("bass", lib, tmp_path / "models", epochs=1, force=True)
    assert (out / "vae.pth").exists()


def test_build_matrix_heterogeneous_keys():
    """Test that build_matrix handles missing keys by filling with 0.0."""
    features_by_name = {
        "p1": {"b": 1.0, "a": 2.0},
        "p2": {"b": 3.0, "c": 4.0},
    }
    names, keys, matrix = build_matrix(features_by_name)

    # Keys should be sorted union
    assert keys == ["a", "b", "c"]
    # Names should be sorted
    assert names == ["p1", "p2"]
    # Matrix should have 0.0 for missing keys
    # p1 row: a=2.0, b=1.0, c=0.0
    # p2 row: a=0.0, b=3.0, c=4.0
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix, np.array([
        [2.0, 1.0, 0.0],
        [0.0, 3.0, 4.0],
    ], dtype=np.float32))
