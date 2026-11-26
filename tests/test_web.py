import math
import shutil
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from serumgen import web
from serumgen.importer import import_folder
from serumgen.train import train_type
import presets as real  # tests/presets.py — locates real presets, skips if absent



@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("web")
    src = tmp / "src" / "Bass"
    src.mkdir(parents=True)
    for f in real.sample(45):
        shutil.copy(f, src / Path(f).name)
    lib = tmp / "library"
    import_folder(tmp / "src", lib)
    models = tmp / "models"
    train_type("bass", lib, models, epochs=5, latent_dim=4)
    web._CACHE.clear()
    web._DOWNLOADS.clear()
    return lib, models, tmp


def test_available_types_lists_trained(trained):
    lib, models, tmp = trained
    types = web.available_types(lib, models)
    assert [t["type"] for t in types] == ["bass"]
    assert types[0]["count"] == 45
    assert types[0]["color"].startswith("#")


def test_projection_one_point_per_preset(trained):
    lib, models, tmp = trained
    proj = web.get_projection("bass", lib, models)
    assert len(proj["points"]) == 45
    for p in proj["points"]:
        assert math.isfinite(p["x"]) and math.isfinite(p["y"])
        assert isinstance(p["name"], str)


def test_projection_is_cached(trained):
    lib, models, tmp = trained
    first = web.get_projection("bass", lib, models)
    assert web.get_projection("bass", lib, models) is first


def test_generate_one_writes_parseable_preset(trained):
    lib, models, tmp = trained
    from serumgen import packager
    result = web.generate_one("bass", lib, models, tmp / "generated", 0.8)
    assert math.isfinite(result["x"]) and math.isfinite(result["y"])
    path = web._DOWNLOADS[result["download_id"]]
    assert path.exists()
    data = packager.load(path)
    assert data["metadata"]["presetAuthor"] == "audiomimi experimental labs"


def test_generate_one_uses_nearest_donor(trained, monkeypatch):
    """The web path must pick the donor nearest the sample, not a random one."""
    lib, models, tmp = trained
    from serumgen import generate as gen
    entry = web.get_projection("bass", lib, models)
    z = torch.randn(1, entry["meta"]["latent_dim"])
    monkeypatch.setattr(gen, "sample_near_library", lambda *a, **k: z.clone())
    expected = gen._nearest_template(z.numpy()[0], entry["templates"],
                                     entry["template_mus"])
    result = web.generate_one("bass", lib, models, tmp / "generated", 0.8)
    assert result["donor"] == expected.stem


def test_generate_one_low_temperature_lands_near_library(trained):
    """At the lowest temperature a sample should sit among real presets on the map."""
    import numpy as np
    lib, models, tmp = trained
    proj = web.get_projection("bass", lib, models)
    pts = np.array([[p["x"], p["y"]] for p in proj["points"]])
    spread = pts.std(0).mean()
    # median over many draws: single random samples occasionally land further
    # out on the tiny test model, which made a per-sample assert flaky
    ds = []
    for _ in range(20):
        r = web.generate_one("bass", lib, models, tmp / "generated", 0.2)
        ds.append(np.sqrt(((pts - [r["x"], r["y"]]) ** 2).sum(1)).min())
    assert np.median(ds) < 0.15 * spread


def test_generate_missing_type_raises(trained):
    lib, models, tmp = trained
    with pytest.raises(SystemExit):
        web.generate_one("lead", lib, models, tmp / "generated", 0.8)


def test_zip_downloads_single_type_is_flat(trained):
    import io
    import zipfile
    lib, models, tmp = trained
    rs = [web.generate_one("bass", lib, models, tmp / "generated", 0.5) for _ in range(3)]
    data, count, types = web.zip_downloads([r["download_id"] for r in rs] + ["nope"])
    assert count == 3 and types == {"bass"}             # unknown ids are skipped
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert sorted(z.namelist()) == sorted(f"{r['name']}.SerumPreset" for r in rs)
        for r in rs:
            src = web._DOWNLOADS[r["download_id"]].read_bytes()
            assert z.read(f"{r['name']}.SerumPreset") == src


def test_zip_downloads_mixed_types_use_folders(trained, tmp_path):
    import io
    import zipfile
    lib, models, tmp = trained
    r = web.generate_one("bass", lib, models, tmp / "generated", 0.5)
    lead = tmp_path / "lead" / "Lead_abc123.SerumPreset"     # stand-in second type
    lead.parent.mkdir()
    lead.write_bytes(b"x")
    web._DOWNLOADS["lead-id"] = lead
    data, count, types = web.zip_downloads([r["download_id"], "lead-id"])
    assert types == {"bass", "lead"}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert sorted(z.namelist()) == sorted([f"bass/{r['name']}.SerumPreset",
                                               "lead/Lead_abc123.SerumPreset"])


def test_zip_filename_names_the_type():
    assert web.zip_filename({"bass"}, "20261126-1344") == "serumgen-bass-20261126-1344.zip"
    assert (web.zip_filename({"bass", "lead"}, "20261126-1344")
            == "serumgen-session-20261126-1344.zip")


def test_zip_downloads_nothing_valid(trained):
    data, count, types = web.zip_downloads(["nope", ""])
    assert count == 0 and types == set()


def _start_server(library_dir, models_dir, out_dir, password=None):
    """Spin up a real _Handler-backed server on an ephemeral port."""
    web._Handler.library_dir = str(library_dir)
    web._Handler.models_dir = str(models_dir)
    web._Handler.out_dir = str(out_dir)
    web._Handler.password = password
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web._Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def _get(port, path, user=None, password=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    if user is not None:
        import base64
        creds = base64.b64encode(f"{user}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {creds}")
    return urllib.request.urlopen(req, timeout=5)


def test_download_all_route_serves_zip(trained):
    import io
    import zipfile
    lib, models, tmp = trained
    rs = [web.generate_one("bass", lib, models, tmp / "generated", 0.5) for _ in range(2)]
    httpd, port = _start_server(lib, models, tmp / "generated")
    try:
        ids = ",".join(r["download_id"] for r in rs)
        resp = _get(port, f"/api/download-all?ids={ids}")
        assert resp.headers["Content-Type"] == "application/zip"
        assert 'filename="serumgen-bass-' in resp.headers["Content-Disposition"]
        with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
            assert len(z.namelist()) == 2
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(port, "/api/download-all?ids=nope")
        assert exc.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_no_password_configured_allows_unauthenticated_access(trained):
    lib, models, tmp = trained
    httpd, port = _start_server(lib, models, tmp / "generated", password=None)
    try:
        resp = _get(port, "/api/types")
        assert resp.status == 200
    finally:
        httpd.shutdown(); httpd.server_close()


def test_password_configured_rejects_missing_auth(trained):
    lib, models, tmp = trained
    httpd, port = _start_server(lib, models, tmp / "generated", password="secret123")
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(port, "/api/types")
        assert exc.value.code == 401
        assert exc.value.headers.get("WWW-Authenticate", "").startswith("Basic")
    finally:
        httpd.shutdown(); httpd.server_close()


def test_password_configured_rejects_wrong_credentials(trained):
    lib, models, tmp = trained
    httpd, port = _start_server(lib, models, tmp / "generated", password="secret123")
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(port, "/api/types", user="serumgen", password="wrong")
        assert exc.value.code == 401
    finally:
        httpd.shutdown(); httpd.server_close()


def test_password_configured_accepts_correct_credentials(trained):
    lib, models, tmp = trained
    httpd, port = _start_server(lib, models, tmp / "generated", password="secret123")
    try:
        resp = _get(port, "/api/types", user="serumgen", password="secret123")
        assert resp.status == 200
    finally:
        httpd.shutdown(); httpd.server_close()
        web._Handler.password = None  # reset for other tests in this module


def test_brand_svgs_are_served(trained):
    lib, models, tmp = trained
    httpd, port = _start_server(lib, models, tmp / "generated")
    try:
        for name in ("serumgen-wordmark.svg", "favicon.svg"):
            resp = _get(port, f"/static/brand/{name}")
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "image/svg+xml"
            assert resp.read().startswith(b"<svg")
    finally:
        httpd.shutdown(); httpd.server_close()


@pytest.mark.parametrize("path", [
    "/static/../web.py",
    "/static/%2e%2e/web.py",
    "/static/brand/../../web.py",
    "/static/index.html",          # only .svg is served from /static/
    "/static/brand/missing.svg",
])
def test_static_route_refuses_everything_else(trained, path):
    lib, models, tmp = trained
    httpd, port = _start_server(lib, models, tmp / "generated")
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(port, path)
        assert exc.value.code == 404
    finally:
        httpd.shutdown(); httpd.server_close()
