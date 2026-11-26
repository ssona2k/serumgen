import json
import pytest

from serumgen.cli import main


@pytest.mark.parametrize("cmd", [
    ["--help"], ["import", "--help"], ["library", "--help"],
    ["extract", "--help"], ["train", "--help"], ["generate", "--help"],
    ["interpolate", "--help"], ["map", "--help"],
])
def test_help_exits_zero(cmd):
    with pytest.raises(SystemExit) as exc:
        main(cmd)
    assert exc.value.code == 0


def test_library_command_empty(tmp_path, capsys):
    assert main(["library", "--library-dir", str(tmp_path / "library")]) == 0
    out = capsys.readouterr().out
    assert "empty" in out.lower()


def test_library_command_lists_types(tmp_path, capsys):
    lib = tmp_path / "library"
    (lib / "bass").mkdir(parents=True)
    (lib / "bass" / "a.json").write_text("{}")
    (lib / "bass" / "a.SerumPreset").write_bytes(b"x")
    (lib / "index.json").write_text(json.dumps(
        {"h1": {"name": "a", "type": "bass", "author": "Jane Doe",
                "source": "s", "preset": "bass/a.SerumPreset",
                "params": "bass/a.json"}}))
    assert main(["library", "--library-dir", str(lib)]) == 0
    out = capsys.readouterr().out
    assert "bass" in out and "1" in out
