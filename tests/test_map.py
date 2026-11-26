import shutil
from pathlib import Path

import pytest

from serumgen.importer import import_folder
from serumgen.train import train_type
from serumgen.generate import generate
from serumgen.map import build_map
import presets as real  # tests/presets.py — locates real presets, skips if absent



@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("map")
    src = tmp / "src" / "Bass"
    src.mkdir(parents=True)
    for f in real.sample(45):
        shutil.copy(f, src / Path(f).name)
    lib = tmp / "library"
    import_folder(tmp / "src", lib)
    models = tmp / "models"
    train_type("bass", lib, models, epochs=5, latent_dim=4)
    gen = tmp / "generated" / "bass"
    generate("bass", lib, models, gen, n=1)
    return tmp, lib, models


def test_build_map_writes_html(trained):
    tmp, lib, models = trained
    out = build_map("bass", lib, models, tmp / "generated" / "bass",
                    tmp / "map.html")
    html = out.read_text()
    assert "<svg" in html
    # every training preset appears by name (hover titles)
    a_preset = next((lib / "bass").glob("*.json")).stem
    assert a_preset in html
    # generated preset overlaid
    assert "generated" in html
    # worst-reconstruction section present
    assert "Worst reconstruction" in html


def test_build_map_tsne(trained):
    """Test t-SNE dimensionality reduction method."""
    tmp, lib, models = trained
    out = build_map("bass", lib, models, tmp / "generated" / "bass",
                    tmp / "map_tsne.html", method="tsne")
    html = out.read_text()
    assert out.exists()
    assert "<svg" in html


def test_build_map_missing_generated_dir(trained):
    """Test graceful handling of missing generated directory."""
    tmp, lib, models = trained
    out = build_map("bass", lib, models, tmp / "no-such-dir",
                    tmp / "map_nogen.html")
    html = out.read_text()
    assert out.exists()
    assert "<svg" in html
    # generated section should not be in HTML when dir doesn't exist
    assert 'class="generated"' not in html


def test_build_map_escapes_names(trained):
    """Test HTML escaping of special characters in preset names."""
    tmp, lib, models = trained
    # Copy an existing preset with a name containing special chars
    src_preset = next((lib / "bass").glob("*.json"))
    special_name = "BA <Evil> & Co.json"
    dest = lib / "bass" / special_name
    shutil.copy(src_preset, dest)

    # Rebuild map
    out = build_map("bass", lib, models, tmp / "generated" / "bass",
                    tmp / "map_escaped.html")
    html = out.read_text()

    # Check that special chars are properly escaped
    assert "BA &lt;Evil&gt; &amp; Co" in html
