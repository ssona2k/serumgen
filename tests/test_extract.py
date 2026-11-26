from serumgen.extract import extract_numeric_features, extract_author

SAMPLE = {
    "metadata": {"presetAuthor": "Jane Doe", "presetName": "BA Test"},
    "data": {
        "Env0": {"plainParams": {"kParamAttack": 0.05, "kParamLabel": "text-ignored"}},
        "Osc0": {"plainParams": {"kParamLevel": 0.8, "kParamOn": True}},
        "Meta": {"noPlainParams": True},
    },
}


def test_flattens_numeric_params():
    feats = extract_numeric_features(SAMPLE)
    assert feats == {
        "Env0/kParamAttack": 0.05,
        "Osc0/kParamLevel": 0.8,
        "Osc0/kParamOn": 1.0,
    }


def test_captures_nested_submodule_params():
    """Params inside a nested dict sub-module (e.g. the wavetable engine)."""
    data = {"data": {"Oscillator0": {
        "plainParams": {"kParamLevel": 0.7},
        "WTOsc0": {"plainParams": {"kParamTablePos": 105.0, "kParamWarp": 0.6}},
    }}}
    feats = extract_numeric_features(data)
    assert feats == {
        "Oscillator0/kParamLevel": 0.7,
        "Oscillator0/WTOsc0/kParamTablePos": 105.0,
        "Oscillator0/WTOsc0/kParamWarp": 0.6,
    }


def test_lists_are_skipped():
    """FX racks are lists — intentionally not traversed."""
    data = {"data": {"FXRack0": {"FX": [
        {"FXBode": {"plainParams": {"kParamDrive": 0.5}}},
    ]}}}
    assert extract_numeric_features(data) == {}


def test_empty_when_no_data():
    assert extract_numeric_features({"metadata": {}}) == {}


def test_author_from_metadata():
    assert extract_author(SAMPLE, "whatever.SerumPreset") == "Jane Doe"


def test_author_from_filename_tag():
    assert extract_author({"metadata": {}}, "BA Deep [XY].SerumPreset") == "XY"


def test_author_empty_fallback():
    assert extract_author({}, "BA Deep.SerumPreset") == ""
